//! Per-pixel halftone evaluation, the screen preview compositor and the high-resolution
//! 1-bit plate renderer used for export.
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use super::layout::{MarkSettings, Rect};
use super::par::par_rows;
use super::spot::{blue_noise, DotShape, SpotTable, MASK_N};
use super::tonemap::{Pyramid, ToneMaps};

#[derive(Clone, Debug, PartialEq)]
pub struct PlateParams {
    pub angle_deg: f32,
    pub lpi: f32,
    pub shape: DotShape,
    pub elong: f32,
    /// Dot gain to compensate for, as a fraction at the 50% point (0..0.25).
    pub dot_gain: f32,
    /// Smallest dot that will be printed (fraction).
    pub min_dot: f32,
    /// Largest dot (fraction); below 1.0 keeps shadows open.
    pub max_dot: f32,
    /// Tones at or below this drop out entirely (fraction).
    pub dropout: f32,
    pub invert: bool,
}

impl Default for PlateParams {
    fn default() -> Self {
        PlateParams {
            angle_deg: 45.0,
            lpi: 45.0,
            shape: DotShape::Round,
            elong: 1.4,
            dot_gain: 0.0,
            min_dot: 0.03,
            max_dot: 1.0,
            dropout: 0.005,
            invert: false,
        }
    }
}

/// Invert `printed = x + 4g·x(1-x)` so the dot printed for `t` lands on `t` after press gain.
pub fn dot_gain_compensate(t: f32, g: f32) -> f32 {
    let g = g.clamp(-0.2, 0.25);
    if g.abs() < 1e-5 {
        return t;
    }
    let a = 1.0 + 4.0 * g;
    let disc = (a * a - 16.0 * g * t).max(0.0);
    ((a - disc.sqrt()) / (8.0 * g)).clamp(0.0, 1.0)
}

/// Cache of the last cell looked up, so consecutive pixels in a cell share one tone sample.
#[derive(Clone, Copy, Default)]
pub struct Cell {
    iu: i64,
    iv: i64,
    region: i32,
    t: f32,
    valid: bool,
}

pub struct PlateEval<'a> {
    params: PlateParams,
    pitch: f64,
    cos: f64,
    sin: f64,
    spot: &'a SpotTable,
    pyr: &'a Pyramid,
    image: Rect,
    patches: &'a [(Rect, f32)],
    patch_bounds: Option<Rect>,
    cell_px: f32,
    base_w: f32,
}

impl<'a> PlateEval<'a> {
    pub fn new(
        params: PlateParams,
        spot: &'a SpotTable,
        pyr: &'a Pyramid,
        image: Rect,
        patches: &'a [(Rect, f32)],
    ) -> Self {
        let pitch = 25.4 / params.lpi.max(1.0) as f64;
        let a = (params.angle_deg as f64).to_radians();
        let base_w = pyr.base_size().0 as f32;
        let cell_px = pitch as f32 * (base_w / image.w.max(1e-3));
        let patch_bounds = patches.iter().fold(None, |acc: Option<Rect>, (r, _)| {
            Some(match acc {
                Some(b) => b.union(r),
                None => *r,
            })
        });
        PlateEval {
            params,
            pitch,
            cos: a.cos(),
            sin: a.sin(),
            spot,
            pyr,
            image,
            patches,
            patch_bounds,
            cell_px,
            base_w,
        }
    }

    /// Bounding box of everything this plate prints (image + wedge patches).
    pub fn content_bounds(&self) -> Rect {
        match self.patch_bounds {
            Some(b) => self.image.union(&b),
            None => self.image,
        }
    }

    pub fn pitch_mm(&self) -> f32 {
        self.pitch as f32
    }

    #[inline]
    fn region_at(&self, x: f32, y: f32) -> i32 {
        if self.image.contains(x, y) {
            return 0;
        }
        if let Some(b) = self.patch_bounds {
            if b.contains(x, y) {
                for (i, (r, _)) in self.patches.iter().enumerate() {
                    if r.contains(x, y) {
                        return 1 + i as i32;
                    }
                }
            }
        }
        -1
    }

    /// Apply invert, dot-gain compensation and min/max/dropout to a raw ink value.
    fn adjust(&self, raw: f32) -> f32 {
        let p = &self.params;
        let mut t = raw.clamp(0.0, 1.0);
        if p.invert {
            t = 1.0 - t;
        }
        t = dot_gain_compensate(t, p.dot_gain);
        if t <= p.dropout.max(1e-4) {
            return 0.0;
        }
        t.max(p.min_dot).min(p.max_dot)
    }

    fn tone_at(&self, region: i32, cx: f32, cy: f32) -> f32 {
        if region == 0 {
            let r = self.image;
            let u = ((cx - r.x) / r.w).clamp(0.0, 1.0);
            let v = ((cy - r.y) / r.h).clamp(0.0, 1.0);
            self.adjust(self.pyr.sample(u, v, self.cell_px))
        } else {
            // Wedge patches are nominal dot areas: no curves, so press gain can be measured.
            self.patches[(region - 1) as usize].1
        }
    }

    /// Is the page point (x, y) mm covered by ink in a hard-threshold (film) rendering?
    #[inline]
    pub fn is_on(&self, x: f32, y: f32, cell: &mut Cell) -> bool {
        let region = self.region_at(x, y);
        if region < 0 {
            return false;
        }
        let (xd, yd) = (x as f64, y as f64);
        let u = xd * self.cos - yd * self.sin;
        let v = xd * self.sin + yd * self.cos;
        let su = u / self.pitch;
        let sv = v / self.pitch;
        let iu = su.floor();
        let iv = sv.floor();
        let (iu_i, iv_i) = (iu as i64, iv as i64);
        if !(cell.valid && cell.iu == iu_i && cell.iv == iv_i && cell.region == region) {
            let cu = (iu + 0.5) * self.pitch;
            let cv = (iv + 0.5) * self.pitch;
            let cx = (cu * self.cos + cv * self.sin) as f32;
            let cy = (-cu * self.sin + cv * self.cos) as f32;
            *cell = Cell { iu: iu_i, iv: iv_i, region, t: self.tone_at(region, cx, cy), valid: true };
        }
        let t = cell.t;
        if t <= 0.0 {
            return false;
        }
        if t >= 0.9999 {
            return true;
        }
        if self.params.shape.uses_mask() {
            // Stochastic screen: each lattice cell is one grain, switched on by the blue-noise mask.
            let n = MASK_N as i64;
            let mx = iu_i.rem_euclid(n) as usize;
            let my = iv_i.rem_euclid(n) as usize;
            return blue_noise()[my * MASK_N + mx] <= t;
        }
        let fx = (su - iu - 0.5) as f32;
        let fy = (sv - iv - 0.5) as f32;
        self.spot.rank(fx, fy) <= t
    }

    /// Continuous-tone stand-in for the dots, used when zoomed out too far to resolve them.
    pub fn tone_preview(&self, x: f32, y: f32, footprint_mm: f32) -> f32 {
        match self.region_at(x, y) {
            -1 => 0.0,
            0 => {
                let r = self.image;
                let u = ((x - r.x) / r.w).clamp(0.0, 1.0);
                let v = ((y - r.y) / r.h).clamp(0.0, 1.0);
                let fp = footprint_mm.max(self.pitch as f32) * (self.base_w / r.w.max(1e-3));
                self.adjust(self.pyr.sample(u, v, fp))
            }
            i => self.patches[(i - 1) as usize].1,
        }
    }
}

/// Everything that defines the page: geometry, image rectangle, marks and plates.
#[derive(Clone, Debug, PartialEq)]
pub struct PlateSetup {
    pub name: String,
    pub params: PlateParams,
    /// Preview ink colour (sRGB 0..1).
    pub ink: [f32; 3],
    pub visible: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Scene {
    pub page_w: f32,
    pub page_h: f32,
    pub image: Rect,
    pub marks: MarkSettings,
    pub plates: Vec<PlateSetup>,
}

pub struct Prepared {
    spots: Vec<SpotTable>,
    patches: Vec<Vec<(Rect, f32)>>,
}

impl Scene {
    pub fn prepare(&self) -> Prepared {
        let n = self.plates.len();
        Prepared {
            spots: self.plates.iter().map(|p| SpotTable::new(p.params.shape, p.params.elong)).collect(),
            patches: (0..n)
                .map(|i| if self.marks.wedge { self.marks.wedge_patches(self.image, i) } else { Vec::new() })
                .collect(),
        }
    }

    /// One evaluator per plate. `maps` must have the same number of plates as the scene.
    pub fn evals<'a>(&'a self, prep: &'a Prepared, maps: &'a ToneMaps) -> Vec<PlateEval<'a>> {
        self.plates
            .iter()
            .enumerate()
            .map(|(i, p)| {
                PlateEval::new(p.params.clone(), &prep.spots[i], &maps.plates[i], self.image, &prep.patches[i])
            })
            .collect()
    }

    pub fn label(&self, i: usize, job: &str) -> String {
        let p = &self.plates[i];
        format!(
            "{}  {:.1} deg  {:.0} lpi  {}  {}",
            p.name,
            p.params.angle_deg,
            p.params.lpi,
            p.params.shape.name(),
            job
        )
    }
}

/// A pixel grid over part of the page. `px` is millimetres per pixel.
#[derive(Clone, Copy, Debug)]
pub struct Region {
    pub x0: f32,
    pub y0: f32,
    pub px: f32,
    pub w: usize,
    pub h: usize,
}

/// Multiply-blend all layers onto white paper. Returns RGBA bytes. If `stale` becomes true
/// the render stops early and the result should be discarded.
pub fn render_composite(
    layers: &[(PlateEval, [f32; 3])],
    reg: &Region,
    dots: bool,
    aa: bool,
    stale: &AtomicBool,
) -> Vec<u8> {
    let mut out = vec![255u8; reg.w * reg.h * 4];
    par_rows(&mut out, reg.w * 4, |j, row| {
        if stale.load(Ordering::Relaxed) {
            return;
        }
        let y = reg.y0 + (j as f32 + 0.5) * reg.px;
        let mut cells = vec![Cell::default(); layers.len()];
        let d = reg.px * 0.25;
        for i in 0..reg.w {
            let x = reg.x0 + (i as f32 + 0.5) * reg.px;
            let mut col = [1.0f32; 3];
            for (k, (ev, ink)) in layers.iter().enumerate() {
                let cov = if !dots {
                    ev.tone_preview(x, y, reg.px)
                } else if aa {
                    let c = &mut cells[k];
                    let n = ev.is_on(x - d, y - d, c) as u8
                        + ev.is_on(x + d, y - d, c) as u8
                        + ev.is_on(x - d, y + d, c) as u8
                        + ev.is_on(x + d, y + d, c) as u8;
                    n as f32 * 0.25
                } else if ev.is_on(x, y, &mut cells[k]) {
                    1.0
                } else {
                    0.0
                };
                if cov > 0.0 {
                    for c in 0..3 {
                        col[c] *= 1.0 - cov * (1.0 - ink[c]);
                    }
                }
            }
            let o = i * 4;
            row[o] = (col[0] * 255.0 + 0.5) as u8;
            row[o + 1] = (col[1] * 255.0 + 0.5) as u8;
            row[o + 2] = (col[2] * 255.0 + 0.5) as u8;
            row[o + 3] = 255;
        }
    });
    out
}

/// 1-bit page raster. Bit 1 = paper, bit 0 = ink (the PNG / PDF-ImageMask convention).
pub struct Bitmap {
    pub w: usize,
    pub h: usize,
    pub stride: usize,
    pub data: Vec<u8>,
}

/// Render one plate for the whole page at `dpi`, hard-thresholded as film output should be.
/// `progress` counts finished rows. With `mirror`, the page is flipped left-to-right.
pub fn render_plate_bitmap(
    ev: &PlateEval,
    page_w: f32,
    page_h: f32,
    dpi: f32,
    mirror: bool,
    progress: &AtomicUsize,
    cancel: &AtomicBool,
) -> Bitmap {
    let w = ((page_w / 25.4 * dpi).round() as usize).max(1);
    let h = ((page_h / 25.4 * dpi).round() as usize).max(1);
    let stride = (w + 7) / 8;
    let mut data = vec![0xFFu8; stride * h];
    let pxx = page_w / w as f32;
    let pxy = page_h / h as f32;
    let b = ev.content_bounds();
    let (c0, c1) = if mirror { (page_w - b.right(), page_w - b.x) } else { (b.x, b.right()) };
    let i0 = ((c0 / pxx).floor().max(0.0)) as usize;
    let i1 = (((c1 / pxx).ceil().max(0.0)) as usize).min(w);
    par_rows(&mut data, stride, |j, row| {
        progress.fetch_add(1, Ordering::Relaxed);
        if cancel.load(Ordering::Relaxed) {
            return;
        }
        let y = (j as f32 + 0.5) * pxy;
        if y < b.y || y >= b.bottom() {
            return;
        }
        let mut cell = Cell::default();
        for i in i0..i1 {
            let xo = (i as f32 + 0.5) * pxx;
            let x = if mirror { page_w - xo } else { xo };
            if ev.is_on(x, y, &mut cell) {
                row[i >> 3] &= !(0x80u8 >> (i & 7));
            }
        }
    });
    Bitmap { w, h, stride, data }
}
