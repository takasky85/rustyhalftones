//! GPU halftone preview: the engine's per-pixel evaluation ported to a WGSL fragment shader,
//! drawn through an egui paint callback. Changing dot size, angle, shape, lpi, dot gain and
//! friends only rewrites a small uniform buffer, so the preview updates in real time.
//! (Tone adjustments still rebuild the ink maps on the CPU, then upload them.)
use std::borrow::Cow;
use std::sync::Arc;

use eframe::egui_wgpu::{self, CallbackResources, CallbackTrait, ScreenDescriptor};
use eframe::wgpu;

use crate::engine::layout::Rect;
use crate::engine::spot::SpotTable;
use crate::engine::tonemap::ToneMaps;

pub const SHADER: &str = include_str!("gpu.wgsl");
/// The shader holds up to this many plates (CMYK).
pub const MAX_PLATES: usize = 4;
const MAX_PATCHES: usize = 12;
const LAYER_FLOATS: usize = 80;
const FRAME_FLOATS: usize = 16 + MAX_PLATES * LAYER_FLOATS;
/// Floats per plate in the spot table (1025 used, padded to a multiple of 4).
pub const SPOT_STRIDE: usize = 1028;
/// The largest ink map the GPU path accepts (matches the biggest working-resolution choice).
pub const MIN_TEXTURE_DIM: u32 = 6144;

// ---------------------------------------------------------------------------------------------
// Ink maps as half-float mip chains

pub fn mip_dims(w: usize, h: usize, level: usize) -> (usize, usize) {
    ((w >> level).max(1), (h >> level).max(1))
}

pub fn mip_count(w: usize, h: usize) -> usize {
    (usize::BITS - w.max(h).max(1).leading_zeros()) as usize
}

pub fn f32_to_f16(f: f32) -> u16 {
    let b = f.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let man = b & 0x007f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man != 0 { 0x0200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = man | 0x0080_0000;
        let shift = (14 - e) as u32;
        let half = m >> shift;
        let rem = m & ((1u32 << shift) - 1);
        let halfway = 1u32 << (shift - 1);
        let up = (rem > halfway || (rem == halfway && (half & 1) == 1)) as u32;
        return sign | (half + up) as u16;
    }
    let half = ((e as u32) << 10) | (man >> 13);
    let rem = man & 0x1fff;
    let up = (rem > 0x1000 || (rem == 0x1000 && (half & 1) == 1)) as u32;
    sign | (half + up) as u16
}

pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h as u32) & 0x8000) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if man == 0 {
            sign
        } else {
            // subnormal half: normalise
            let mut e = 127 - 15 + 1;
            let mut m = man;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | ((e as u32) << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (man << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (man << 13)
    };
    f32::from_bits(bits)
}

/// All plates' ink maps as half floats with a full mip chain (sizes are floor(w / 2^level)).
pub struct GpuInk {
    pub w: usize,
    pub h: usize,
    pub plates: usize,
    /// `[plate][level]`
    pub mips: Vec<Vec<Vec<u16>>>,
}

impl GpuInk {
    pub fn from_maps(maps: &ToneMaps) -> Option<GpuInk> {
        let first = maps.plates.first()?;
        let (w, h) = first.base_size();
        let n = maps.plates.len().min(MAX_PLATES);
        let mips: Vec<Vec<Vec<u16>>> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..n)
                .map(|i| {
                    let plane = maps.plates[i].base_data();
                    s.spawn(move || build_chain(plane, w, h))
                })
                .collect();
            handles.into_iter().map(|t| t.join().unwrap_or_default()).collect()
        });
        if mips.iter().any(|m| m.is_empty()) {
            return None;
        }
        Some(GpuInk { w, h, plates: n, mips })
    }
}

fn build_chain(base: &[f32], w: usize, h: usize) -> Vec<Vec<u16>> {
    let levels = mip_count(w, h);
    let mut out = Vec::with_capacity(levels);
    let mut cur: Vec<f32> = base.to_vec();
    let (mut cw, mut ch) = (w, h);
    for l in 0..levels {
        out.push(cur.iter().map(|&v| f32_to_f16(v)).collect());
        if l + 1 == levels {
            break;
        }
        let (nw, nh) = mip_dims(w, h, l + 1);
        let mut next = vec![0.0f32; nw * nh];
        for y in 0..nh {
            let y0 = (2 * y).min(ch - 1);
            let y1 = (2 * y + 1).min(ch - 1);
            for x in 0..nw {
                let x0 = (2 * x).min(cw - 1);
                let x1 = (2 * x + 1).min(cw - 1);
                next[y * nw + x] =
                    0.25 * (cur[y0 * cw + x0] + cur[y0 * cw + x1] + cur[y1 * cw + x0] + cur[y1 * cw + x1]);
            }
        }
        cur = next;
        cw = nw;
        ch = nh;
    }
    out
}

// ---------------------------------------------------------------------------------------------
// Uniform data

/// Spot tables for every plate, flattened for the shader. Returns the table plus `(min, inv)` per plate.
pub fn spot_tables(tables: &[SpotTable]) -> (Vec<f32>, Vec<(f32, f32)>) {
    let mut data = vec![0.0f32; MAX_PLATES * SPOT_STRIDE];
    let mut params = Vec::new();
    for (i, t) in tables.iter().enumerate().take(MAX_PLATES) {
        let (min, inv, cum) = t.gpu_data();
        data[i * SPOT_STRIDE..i * SPOT_STRIDE + cum.len()].copy_from_slice(cum);
        params.push((min, inv));
    }
    (data, params)
}

#[derive(Clone)]
pub struct LayerDesc {
    /// Index of the plate: selects the ink map layer and the spot table.
    pub plate: usize,
    pub ink: [f32; 3],
    pub angle_deg: f32,
    pub lpi: f32,
    pub dot_gain: f32,
    pub min_dot: f32,
    pub max_dot: f32,
    pub dropout: f32,
    pub invert: bool,
    pub shape_id: u32,
    pub elong: f32,
    pub spot_min: f32,
    pub spot_inv: f32,
    pub patches: Vec<(Rect, f32)>,
}

#[derive(Clone)]
pub struct FrameDesc {
    /// Page top-left in framebuffer pixels.
    pub origin_px: [f32; 2],
    pub px_per_mm: f32,
    pub image: Rect,
    pub page: (f32, f32),
    pub dots: bool,
    pub aa: bool,
    pub max_lod: f32,
    pub base_w: f32,
    pub srgb_target: bool,
    pub layers: Vec<LayerDesc>,
}

impl FrameDesc {
    pub fn to_floats(&self) -> Vec<f32> {
        let mut v = Vec::with_capacity(FRAME_FLOATS);
        let n = self.layers.len().min(MAX_PLATES);
        v.extend_from_slice(&[self.origin_px[0], self.origin_px[1], self.px_per_mm, n as f32]);
        v.extend_from_slice(&[self.image.x, self.image.y, self.image.w, self.image.h]);
        v.extend_from_slice(&[self.page.0, self.page.1, self.dots as u8 as f32, self.aa as u8 as f32]);
        v.extend_from_slice(&[self.max_lod, self.base_w, self.srgb_target as u8 as f32, 0.0]);
        for k in 0..MAX_PLATES {
            let Some(l) = self.layers.get(k) else {
                v.extend(std::iter::repeat(0.0).take(LAYER_FLOATS));
                continue;
            };
            let pitch = 25.4 / l.lpi.max(1.0);
            let a = l.angle_deg.to_radians();
            let cell_px = pitch * self.base_w / self.image.w.max(1e-3);
            v.extend_from_slice(&[l.ink[0], l.ink[1], l.ink[2], 0.0]);
            v.extend_from_slice(&[a.cos(), a.sin(), pitch, cell_px]);
            v.extend_from_slice(&[l.dot_gain, l.min_dot, l.max_dot, l.dropout]);
            v.extend_from_slice(&[l.invert as u8 as f32, l.shape_id as f32, l.elong, l.spot_min]);
            let np = l.patches.len().min(MAX_PATCHES);
            v.extend_from_slice(&[l.spot_inv, l.plate as f32, np as f32, 0.0]);
            for i in 0..MAX_PATCHES {
                match l.patches.get(i) {
                    Some((r, _)) => v.extend_from_slice(&[r.x, r.y, r.w, r.h]),
                    None => v.extend_from_slice(&[0.0; 4]),
                }
            }
            for i in 0..MAX_PATCHES {
                v.push(l.patches.get(i).map(|p| p.1).unwrap_or(0.0));
            }
        }
        debug_assert_eq!(v.len(), FRAME_FLOATS);
        v
    }
}

fn bytes_of(f: &[f32]) -> Vec<u8> {
    f.iter().flat_map(|x| x.to_le_bytes()).collect()
}

// ---------------------------------------------------------------------------------------------
// wgpu resources

struct InkTex {
    _tex: wgpu::Texture,
    view: wgpu::TextureView,
    rev: u64,
}

struct PageGpu {
    buf: wgpu::Buffer,
    bind: Option<wgpu::BindGroup>,
    tex_rev: u64,
}

pub struct GpuResources {
    pipeline: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    spots_buf: wgpu::Buffer,
    spots_ver: u64,
    tex: Option<InkTex>,
    pages: Vec<PageGpu>,
}

impl GpuResources {
    /// Build the pipeline. Any validation problem (bad shader, unsupported limits) comes back
    /// as an error so the app can fall back to the CPU preview instead of crashing.
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Result<Self, String> {
        let limits = device.limits();
        if limits.max_texture_dimension_2d < MIN_TEXTURE_DIM {
            return Err(format!("GPU texture limit {} is too small", limits.max_texture_dimension_2d));
        }
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("halftone shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SHADER)),
        });
        let uniform = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("halftone bgl"),
            entries: &[
                uniform(0),
                uniform(1),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("halftone pipeline layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("halftone pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL })],
            }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleStrip, ..Default::default() },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("ink sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let spots_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("spot tables"),
            size: (MAX_PLATES * SPOT_STRIDE * 4) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if let Some(e) = pollster::block_on(device.pop_error_scope()) {
            return Err(e.to_string());
        }
        Ok(GpuResources { pipeline, bgl, sampler, spots_buf, spots_ver: u64::MAX, tex: None, pages: Vec::new() })
    }

    pub fn set_spots(&mut self, queue: &wgpu::Queue, data: &[f32], ver: u64) {
        if self.spots_ver != ver {
            queue.write_buffer(&self.spots_buf, 0, &bytes_of(data));
            self.spots_ver = ver;
        }
    }

    pub fn set_ink(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, ink: &GpuInk, rev: u64) {
        if self.tex.as_ref().map(|t| t.rev) == Some(rev) {
            return;
        }
        let levels = mip_count(ink.w, ink.h);
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ink maps"),
            size: wgpu::Extent3d {
                width: ink.w as u32,
                height: ink.h as u32,
                depth_or_array_layers: MAX_PLATES as u32,
            },
            mip_level_count: levels as u32,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        for (layer, chain) in ink.mips.iter().enumerate() {
            for (level, data) in chain.iter().enumerate() {
                let (mw, mh) = mip_dims(ink.w, ink.h, level);
                let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
                queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &tex,
                        mip_level: level as u32,
                        origin: wgpu::Origin3d { x: 0, y: 0, z: layer as u32 },
                        aspect: wgpu::TextureAspect::All,
                    },
                    &bytes,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(mw as u32 * 2),
                        rows_per_image: Some(mh as u32),
                    },
                    wgpu::Extent3d { width: mw as u32, height: mh as u32, depth_or_array_layers: 1 },
                );
            }
        }
        let view = tex.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        self.tex = Some(InkTex { _tex: tex, view, rev });
    }

    pub fn has_ink(&self) -> bool {
        self.tex.is_some()
    }

    pub fn write_page(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, slot: usize, frame: &[f32]) {
        while self.pages.len() <= slot {
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("halftone frame"),
                size: (FRAME_FLOATS * 4) as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.pages.push(PageGpu { buf, bind: None, tex_rev: u64::MAX });
        }
        let Some(tex) = &self.tex else { return };
        let page = &mut self.pages[slot];
        queue.write_buffer(&page.buf, 0, &bytes_of(frame));
        if page.bind.is_none() || page.tex_rev != tex.rev {
            page.bind = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("halftone bind group"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: page.buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: self.spots_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&tex.view) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                ],
            }));
            page.tex_rev = tex.rev;
        }
    }

    pub fn draw(&self, slot: usize, pass: &mut wgpu::RenderPass<'_>) {
        if let Some(PageGpu { bind: Some(bind), .. }) = self.pages.get(slot) {
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, bind, &[]);
            pass.draw(0..4, 0..1);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// egui paint callback

pub struct PageCallback {
    pub slot: usize,
    pub frame: Vec<f32>,
    pub ink: Arc<GpuInk>,
    pub ink_rev: u64,
    pub spots: Arc<Vec<f32>>,
    pub spots_ver: u64,
}

impl CallbackTrait for PageCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen: &ScreenDescriptor,
        _encoder: &mut wgpu::CommandEncoder,
        resources: &mut CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        if let Some(r) = resources.get_mut::<GpuResources>() {
            r.set_ink(device, queue, &self.ink, self.ink_rev);
            r.set_spots(queue, &self.spots, self.spots_ver);
            r.write_page(device, queue, self.slot, &self.frame);
        }
        Vec::new()
    }

    fn paint(
        &self,
        _info: eframe::egui::PaintCallbackInfo,
        pass: &mut wgpu::RenderPass<'static>,
        resources: &CallbackResources,
    ) {
        if let Some(r) = resources.get::<GpuResources>() {
            r.draw(self.slot, pass);
        }
    }
}

pub fn callback_shape(rect: eframe::egui::Rect, cb: PageCallback) -> eframe::egui::Shape {
    eframe::egui::Shape::Callback(egui_wgpu::Callback::new_paint_callback(rect, cb))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_float_round_trip() {
        for i in 0..=2000 {
            let v = i as f32 / 2000.0;
            let back = f16_to_f32(f32_to_f16(v));
            assert!((back - v).abs() <= 0.001 * v.max(0.0005), "{} -> {}", v, back);
        }
        assert_eq!(f32_to_f16(0.0), 0);
        assert_eq!(f16_to_f32(f32_to_f16(1.0)), 1.0);
        assert_eq!(f16_to_f32(f32_to_f16(0.5)), 0.5);
    }

    #[test]
    fn mip_chain_sizes() {
        assert_eq!(mip_count(1, 1), 1);
        assert_eq!(mip_count(4096, 3000), 13);
        assert_eq!(mip_dims(3073, 2000, 1), (1536, 1000));
        assert_eq!(mip_dims(3, 1, 5), (1, 1));
        let base = vec![0.5f32; 7 * 5];
        let chain = build_chain(&base, 7, 5);
        assert_eq!(chain.len(), mip_count(7, 5));
        for (l, d) in chain.iter().enumerate() {
            let (w, h) = mip_dims(7, 5, l);
            assert_eq!(d.len(), w * h);
            assert!(d.iter().all(|&v| f16_to_f32(v) == 0.5));
        }
    }

    #[test]
    fn frame_layout_matches_shader() {
        let layer = LayerDesc {
            plate: 0,
            ink: [0.0; 3],
            angle_deg: 45.0,
            lpi: 45.0,
            dot_gain: 0.0,
            min_dot: 0.03,
            max_dot: 1.0,
            dropout: 0.005,
            invert: false,
            shape_id: 0,
            elong: 1.4,
            spot_min: 0.0,
            spot_inv: 1.0,
            patches: Vec::new(),
        };
        let f = FrameDesc {
            origin_px: [0.0; 2],
            px_per_mm: 4.0,
            image: Rect::new(0.0, 0.0, 10.0, 10.0),
            page: (10.0, 10.0),
            dots: true,
            aa: true,
            max_lod: 3.0,
            base_w: 100.0,
            srgb_target: false,
            layers: vec![layer],
        };
        assert_eq!(f.to_floats().len(), FRAME_FLOATS);
        assert_eq!(FRAME_FLOATS * 4, 16 * 4 + 4 * 320);
        // the shader declares Frame as 4 vec4 + 4 layers of (5 + 12 + 3) vec4
        assert!(SHADER.contains("array<Layer, 4>") && SHADER.contains("array<vec4<f32>, 1028>"));
        assert_eq!(SPOT_STRIDE * 4, 1028 * 4);
    }
}

/// Renders the shader headless and compares it with the CPU renderer. Needs a GPU or a software
/// adapter, so it only runs with `--features gpu-tests` (and passes quietly if there is none).
#[cfg(all(test, feature = "gpu-tests"))]
mod gpu_tests {
    use super::*;
    use crate::engine::layout::MarkSettings;
    use crate::engine::render::{render_composite, PlateEval, PlateParams, Region};
    use crate::engine::spot::DotShape;
    use crate::engine::tonemap::{build_tone_maps, SepMode, SourceImage, ToneSettings};
    use std::sync::atomic::AtomicBool;

    fn adapter_and_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::default();
        for fallback in [false, true] {
            let opts = wgpu::RequestAdapterOptions { force_fallback_adapter: fallback, ..Default::default() };
            let Ok(adapter) = pollster::block_on(instance.request_adapter(&opts)) else { continue };
            let desc = wgpu::DeviceDescriptor { required_limits: adapter.limits(), ..Default::default() };
            if let Ok(dq) = pollster::block_on(adapter.request_device(&desc)) {
                println!("using adapter: {:?}", adapter.get_info());
                return Some(dq);
            }
        }
        None
    }

    fn render_gpu(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        res: &mut GpuResources,
        frame: &FrameDesc,
        w: u32,
        h: u32,
    ) -> Vec<u8> {
        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = target.create_view(&Default::default());
        res.write_page(device, queue, 0, &frame.to_floats());
        let mut enc = device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::RED), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            res.draw(0, &mut pass);
        }
        let bpr = w * 4;
        assert_eq!(bpr % 256, 0, "test width must give 256-aligned rows");
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (bpr * h) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buf,
                layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(bpr), rows_per_image: Some(h) },
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        queue.submit(Some(enc.finish()));
        let slice = buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
        device.poll(wgpu::PollType::Wait).expect("poll");
        let data = slice.get_mapped_range().to_vec();
        data
    }

    fn compare(shape: DotShape, ppmm: f32, dots: bool, max_bad_fraction: f32, max_mean_diff: f32) {
        let Some((device, queue)) = adapter_and_device() else {
            println!("no GPU adapter, skipping");
            return;
        };
        let mut res = GpuResources::new(&device, wgpu::TextureFormat::Rgba8Unorm).expect("pipeline");

        // a smooth gradient with some structure
        let (iw, ih) = (160usize, 120usize);
        let mut rgb = Vec::with_capacity(iw * ih * 3);
        for y in 0..ih {
            for x in 0..iw {
                let v = ((x as f32 / iw as f32) * 0.8 + (y as f32 / ih as f32) * 0.2) * 255.0;
                let w = if (x / 20 + y / 20) % 2 == 0 { v } else { 255.0 - v };
                rgb.extend_from_slice(&[w as u8, w as u8, w as u8]);
            }
        }
        let src = SourceImage::new(iw, ih, rgb);
        let ts = ToneSettings { mode: SepMode::Mono, ..ToneSettings::default() };
        let maps = build_tone_maps(&src, &ts, 1024);
        let ink = GpuInk::from_maps(&maps).expect("ink");
        res.set_ink(&device, &queue, &ink, 1);

        let params = PlateParams {
            angle_deg: 30.0,
            lpi: 20.0,
            shape,
            elong: 1.4,
            dot_gain: 0.08,
            min_dot: 0.03,
            max_dot: 0.98,
            dropout: 0.01,
            invert: false,
        };
        let spot = SpotTable::new(shape, params.elong);
        let (data, sp) = spot_tables(std::slice::from_ref(&spot));
        res.set_spots(&queue, &data, 1);

        let image = Rect::new(8.0, 8.0, 64.0, 48.0);
        let marks = MarkSettings::default();
        let patches = marks.wedge_patches(image, 0);
        let (w, h) = (256u32, 256u32);
        let page = (w as f32 / ppmm, h as f32 / ppmm);

        let eval = PlateEval::new(params.clone(), &spot, &maps.plates[0], image, &patches);
        let reg = Region { x0: 0.0, y0: 0.0, px: 1.0 / ppmm, w: w as usize, h: h as usize };
        let stale = AtomicBool::new(false);
        let cpu = render_composite(&[(eval, [0.1, 0.1, 0.1])], &reg, dots, true, &stale);

        let layer = LayerDesc {
            plate: 0,
            ink: [0.1, 0.1, 0.1],
            angle_deg: params.angle_deg,
            lpi: params.lpi,
            dot_gain: params.dot_gain,
            min_dot: params.min_dot,
            max_dot: params.max_dot,
            dropout: params.dropout,
            invert: false,
            shape_id: shape.id(),
            elong: params.elong,
            spot_min: sp[0].0,
            spot_inv: sp[0].1,
            patches,
        };
        let frame = FrameDesc {
            origin_px: [0.0, 0.0],
            px_per_mm: ppmm,
            image,
            page,
            dots,
            aa: true,
            max_lod: (mip_count(ink.w, ink.h) - 1) as f32,
            base_w: ink.w as f32,
            srgb_target: false,
            layers: vec![layer],
        };
        let gpu = render_gpu(&device, &queue, &mut res, &frame, w, h);

        let mut bad = 0usize;
        let mut sum = 0.0f64;
        for i in 0..(w * h) as usize {
            let d = (gpu[i * 4] as f32 - cpu[i * 4] as f32).abs() / 255.0;
            sum += d as f64;
            if d > 0.25 {
                bad += 1;
            }
        }
        let n = (w * h) as f32;
        let (frac, mean) = (bad as f32 / n, (sum / n as f64) as f32);
        println!("{:?} dots={} bad={:.4} mean diff={:.4}", shape, dots, frac, mean);
        assert!(frac <= max_bad_fraction, "{:?}: {:.2}% of pixels differ", shape, frac * 100.0);
        assert!(mean <= max_mean_diff, "{:?}: mean difference {:.3}", shape, mean);
    }

    #[test]
    fn gpu_dots_match_cpu_for_every_shape() {
        for s in DotShape::ALL {
            compare(s, 8.0, true, 0.01, 0.01);
        }
    }

    #[test]
    fn gpu_tone_matches_cpu() {
        compare(DotShape::Round, 1.0, false, 0.001, 0.01);
    }
}
