//! egui front end: page canvas, controls, background preview/export jobs.
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::Duration;

use eframe::egui::{self, pos2, vec2, Color32, Pos2, Stroke, Vec2};

use crate::compress::zlib;
use crate::engine::export::{export_all, ExportSpec, PlateOut};
use crate::engine::layout::{MarkSettings, Prim, Rect as MmRect};
use crate::engine::pages::{Unit, PAGE_SIZES};
use crate::engine::render::{render_composite, PlateEval, PlateParams, PlateSetup, Region, Scene};
use crate::engine::spot::DotShape;
use crate::engine::tonemap::{build_tone_maps, SepMode, SourceImage, ToneMaps, ToneSettings};

const ANGLE_PRESETS: [(&str, [f32; 4]); 2] = [
    ("Offset standard (C15 M75 Y0 K45)", [15.0, 75.0, 0.0, 45.0]),
    ("Alternate (C105 M75 Y90 K45)", [105.0, 75.0, 90.0, 45.0]),
];
const DPI_CHOICES: [f32; 5] = [600.0, 1200.0, 1800.0, 2400.0, 2540.0];
const WORK_RES_CHOICES: [usize; 5] = [1024, 2048, 3072, 4096, 6144];

#[derive(Clone, Copy, PartialEq, Debug)]
enum ViewMode {
    Composite,
    Plate(usize),
    /// Every plate on its own page, laid out side by side.
    AllPlates,
    Original,
}

struct PlateUi {
    name: String,
    ink: [u8; 3],
    angle: f32,
    /// percent values, converted to fractions in `scene()`
    dot_gain: f32,
    min_dot: f32,
    max_dot: f32,
    dropout: f32,
    invert: bool,
    visible: bool,
    lpi_override: Option<f32>,
    shape_override: Option<DotShape>,
}

impl PlateUi {
    fn new(name: &str, ink: [u8; 3], angle: f32) -> Self {
        PlateUi {
            name: name.to_string(),
            ink,
            angle,
            dot_gain: 0.0,
            min_dot: 3.0,
            max_dot: 100.0,
            dropout: 0.5,
            invert: false,
            visible: true,
            lpi_override: None,
            shape_override: None,
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
struct MapsKey {
    source_rev: u64,
    tone: ToneSettings,
    work_res: usize,
}

#[derive(Clone, PartialEq)]
struct PreviewReq {
    scene: Scene,
    maps_rev: u64,
    view: ViewMode,
    /// x, y, w, h in mm
    region: [f32; 4],
    ppmm: f32,
    dots: bool,
    aa: bool,
    plate_colors: bool,
}

impl PreviewReq {
    fn covers(&self, o: &PreviewReq, vis: [f32; 4]) -> bool {
        self.scene == o.scene
            && self.maps_rev == o.maps_rev
            && self.view == o.view
            && self.ppmm == o.ppmm
            && self.dots == o.dots
            && self.aa == o.aa
            && self.plate_colors == o.plate_colors
            && self.region[0] <= vis[0] + 1e-3
            && self.region[1] <= vis[1] + 1e-3
            && self.region[0] + self.region[2] >= vis[2] - 1e-3
            && self.region[1] + self.region[3] >= vis[3] - 1e-3
    }
}

enum Msg {
    Image { name: String, result: Result<SourceImage, String> },
    Maps { key: MapsKey, maps: Arc<ToneMaps> },
    Preview { slot: usize, epoch: u64, req: PreviewReq, ok: bool, w: usize, h: usize, rgba: Vec<u8> },
    Export { result: Result<String, String> },
}

#[derive(Clone, Copy)]
enum DragKind {
    Pan,
    Move { off: (f32, f32) },
    Resize { anchor: (f32, f32), aspect: f32 },
}

struct View {
    zoom: f32,    // points per mm
    origin: Vec2, // page (0,0) relative to the canvas top-left, in points
    fit_pending: bool,
}

struct TexInfo {
    tex: egui::TextureHandle,
    x0: f32,
    y0: f32,
    w_mm: f32,
    h_mm: f32,
}

/// Preview state for one page (the All plates view has one per plate).
#[derive(Default)]
struct Slot {
    tex: Option<TexInfo>,
    have: Option<PreviewReq>,
    inflight: Option<(PreviewReq, Arc<AtomicBool>)>,
    pending: Option<PreviewReq>,
}

struct ExportState {
    progress: Arc<AtomicUsize>,
    total: usize,
    cancel: Arc<AtomicBool>,
}

pub struct HalftoneApp {
    ctx: egui::Context,
    tx: mpsc::Sender<Msg>,
    rx: mpsc::Receiver<Msg>,

    // document
    source: Option<Arc<SourceImage>>,
    source_name: String,
    source_rev: u64,
    src_aspect: f32,
    unit: Unit,
    page_idx: usize,
    custom_w: f32,
    custom_h: f32,
    landscape: bool,
    place: MmRect,
    lock_aspect: bool,
    tone: ToneSettings,
    work_res: usize,
    lpi: f32,
    mesh: f32,
    shape: DotShape,
    elong: f32,
    plates: Vec<PlateUi>,
    marks: MarkSettings,

    // view
    view_mode: ViewMode,
    show_marks: bool,
    always_dots: bool,
    aa: bool,
    view: View,
    last_canvas: egui::Rect,
    drag: Option<DragKind>,
    drag_off: (f32, f32),
    plate_colors: bool,
    prev_view_mode: ViewMode,
    cursor_mm: Option<(f32, f32)>,
    tone_mode: bool,

    // pipeline
    maps: Option<Arc<ToneMaps>>,
    maps_key: Option<MapsKey>,
    maps_building: Option<MapsKey>,
    maps_rev: u64,
    slots: Vec<Slot>,
    slot_epoch: u64,
    slot_mode: ViewMode,
    orig_tex: Option<(u64, egui::TextureHandle)>,

    // export
    exp_dpi: f32,
    exp_mirror: bool,
    exp_pdf: bool,
    exp_png: bool,
    export: Option<ExportState>,

    status: String,
}

fn dv_mm(ui: &mut egui::Ui, v: &mut f32, unit: Unit, min_mm: f32, max_mm: f32) -> bool {
    let mut x = unit.from_mm(*v);
    let speed = if unit == Unit::Mm { 0.5 } else { 0.02 };
    let r = ui.add(
        egui::DragValue::new(&mut x)
            .speed(speed)
            .range(unit.from_mm(min_mm)..=unit.from_mm(max_mm))
            .suffix(unit.suffix())
            .max_decimals(3),
    );
    if r.changed() {
        *v = unit.to_mm(x);
        true
    } else {
        false
    }
}

fn load_image(path: &Path) -> Result<SourceImage, String> {
    use image::ImageDecoder;
    let reader =
        image::ImageReader::open(path).map_err(|e| e.to_string())?.with_guessed_format().map_err(|e| e.to_string())?;
    let mut decoder = reader.into_decoder().map_err(|e| e.to_string())?;
    // Phone photos are usually stored sideways with an EXIF orientation tag.
    let orientation = decoder.orientation().ok();
    let mut img = image::DynamicImage::from_decoder(decoder).map_err(|e| e.to_string())?;
    if let Some(o) = orientation {
        img.apply_orientation(o);
    }
    let (w, h) = (img.width() as usize, img.height() as usize);
    let rgb = if img.color().has_alpha() {
        // Flatten transparency onto white paper.
        let rgba = img.to_rgba8();
        let mut out = Vec::with_capacity(w * h * 3);
        for p in rgba.pixels() {
            let a = p[3] as u32;
            for c in 0..3 {
                out.push(((p[c] as u32 * a + 255 * (255 - a)) / 255) as u8);
            }
        }
        out
    } else {
        img.to_rgb8().into_raw()
    };
    Ok(SourceImage::new(w, h, rgb))
}

fn render_job(req: &PreviewReq, maps: &ToneMaps, stale: &AtomicBool) -> (usize, usize, Vec<u8>) {
    let px = 1.0 / req.ppmm;
    let w = ((req.region[2] * req.ppmm).ceil() as usize).max(1);
    let h = ((req.region[3] * req.ppmm).ceil() as usize).max(1);
    let reg = Region { x0: req.region[0], y0: req.region[1], px, w, h };
    let prep = req.scene.prepare();
    let evals = req.scene.evals(&prep, maps);
    let mut layers: Vec<(PlateEval, [f32; 3])> = Vec::new();
    for (i, (ev, p)) in evals.into_iter().zip(req.scene.plates.iter()).enumerate() {
        match req.view {
            ViewMode::Composite => {
                if p.visible {
                    layers.push((ev, p.ink));
                }
            }
            ViewMode::Plate(k) => {
                if i == k {
                    layers.push((ev, if req.plate_colors { p.ink } else { [0.04; 3] }));
                }
            }
            ViewMode::AllPlates | ViewMode::Original => {}
        }
    }
    let rgba = render_composite(&layers, &reg, req.dots, req.aa, stale);
    (w, h, rgba)
}

impl HalftoneApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let (tx, rx) = mpsc::channel();
        let mut s = HalftoneApp {
            ctx: cc.egui_ctx.clone(),
            tx,
            rx,
            source: None,
            source_name: String::new(),
            source_rev: 0,
            src_aspect: 1.0,
            unit: Unit::Inch,
            page_idx: 0,
            custom_w: 210.0,
            custom_h: 297.0,
            landscape: false,
            place: MmRect::new(20.0, 20.0, 100.0, 100.0),
            lock_aspect: true,
            tone: ToneSettings::default(),
            work_res: 3072,
            lpi: 45.0,
            mesh: 230.0,
            shape: DotShape::Round,
            elong: 1.4,
            plates: Vec::new(),
            marks: MarkSettings::default(),
            view_mode: ViewMode::Composite,
            show_marks: true,
            always_dots: false,
            aa: true,
            view: View { zoom: 0.0, origin: Vec2::ZERO, fit_pending: true },
            last_canvas: egui::Rect::NOTHING,
            drag: None,
            drag_off: (0.0, 0.0),
            plate_colors: false,
            prev_view_mode: ViewMode::Composite,
            cursor_mm: None,
            tone_mode: false,
            maps: None,
            maps_key: None,
            maps_building: None,
            maps_rev: 0,
            slots: Vec::new(),
            slot_epoch: 0,
            slot_mode: ViewMode::Composite,
            orig_tex: None,
            exp_dpi: 1200.0,
            exp_mirror: false,
            exp_pdf: true,
            exp_png: false,
            export: None,
            status: "Open an image (Ctrl+O) or drop one on the window.".to_string(),
        };
        s.reset_plates();
        s
    }

    // ------------------------------------------------------------------ document helpers

    fn page_mm(&self) -> (f32, f32) {
        let (a, b) = if self.page_idx < PAGE_SIZES.len() {
            (PAGE_SIZES[self.page_idx].w_mm, PAGE_SIZES[self.page_idx].h_mm)
        } else {
            (self.custom_w, self.custom_h)
        };
        let (lo, hi) = (a.min(b), a.max(b));
        if self.landscape {
            (hi, lo)
        } else {
            (lo, hi)
        }
    }

    fn reset_plates(&mut self) {
        self.plates = match self.tone.mode {
            SepMode::Mono => vec![PlateUi::new("K", [20, 20, 20], 45.0)],
            SepMode::Cmyk => vec![
                PlateUi::new("C", [0, 174, 239], 15.0),
                PlateUi::new("M", [236, 0, 140], 75.0),
                PlateUi::new("Y", [255, 242, 0], 0.0),
                PlateUi::new("K", [35, 31, 32], 45.0),
            ],
        };
        self.view_mode = ViewMode::Composite;
    }

    fn scene(&self) -> Scene {
        let (pw, ph) = self.page_mm();
        let plates = self
            .plates
            .iter()
            .map(|p| PlateSetup {
                name: p.name.clone(),
                params: PlateParams {
                    angle_deg: p.angle,
                    lpi: p.lpi_override.unwrap_or(self.lpi),
                    shape: p.shape_override.unwrap_or(self.shape),
                    elong: self.elong,
                    dot_gain: p.dot_gain / 100.0,
                    min_dot: p.min_dot / 100.0,
                    max_dot: p.max_dot / 100.0,
                    dropout: p.dropout / 100.0,
                    invert: p.invert,
                },
                ink: [p.ink[0] as f32 / 255.0, p.ink[1] as f32 / 255.0, p.ink[2] as f32 / 255.0],
                visible: p.visible,
            })
            .collect();
        Scene { page_w: pw, page_h: ph, image: self.place, marks: self.marks.clone(), plates }
    }

    fn fit_image(&mut self, fill: bool) {
        let Some(src) = &self.source else { return };
        let (pw, ph) = self.page_mm();
        let m = if fill { [0.0; 4] } else { self.marks.margins(self.plates.len()) };
        let (ax, ay) = (m[0], m[1]);
        let aw = (pw - m[0] - m[2]).max(10.0);
        let ah = (ph - m[1] - m[3]).max(10.0);
        let a = src.w as f32 / src.h as f32;
        let fits_width = aw / a <= ah;
        let (w, h) = if fits_width != fill { (aw, aw / a) } else { (ah * a, ah) };
        self.place = MmRect::new(ax + (aw - w) * 0.5, ay + (ah - h) * 0.5, w, h);
    }

    fn set_source(&mut self, img: SourceImage, name: String) {
        self.src_aspect = img.w as f32 / img.h as f32;
        self.status = format!("{}: {} x {} px", name, img.w, img.h);
        self.source = Some(Arc::new(img));
        self.source_name = name;
        self.source_rev += 1;
        self.orig_tex = None;
        self.reset_slots();
        self.fit_image(false);
    }

    fn rotate(&mut self, cw: bool) {
        let Some(src) = self.source.clone() else { return };
        let r = if cw { src.rotate_cw() } else { src.rotate_ccw() };
        self.src_aspect = r.w as f32 / r.h as f32;
        self.source = Some(Arc::new(r));
        self.source_rev += 1;
        self.orig_tex = None;
        let c = (self.place.x + self.place.w * 0.5, self.place.y + self.place.h * 0.5);
        let (w, h) = (self.place.h, self.place.w);
        self.place = MmRect::new(c.0 - w * 0.5, c.1 - h * 0.5, w, h);
    }

    fn open_dialog(&mut self) {
        if let Some(p) = rfd::FileDialog::new()
            .add_filter("Images", &["png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp", "gif"])
            .pick_file()
        {
            self.load_path(p);
        }
    }

    fn load_path(&mut self, path: PathBuf) {
        let name = path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
        self.status = format!("Loading {}…", name);
        let tx = self.tx.clone();
        let ctx = self.ctx.clone();
        thread::spawn(move || {
            let result = load_image(&path);
            let _ = tx.send(Msg::Image { name, result });
            ctx.request_repaint();
        });
    }

    // ------------------------------------------------------------------ background jobs

    fn poll(&mut self, ctx: &egui::Context) {
        while let Ok(msg) = self.rx.try_recv() {
            match msg {
                Msg::Image { name, result } => match result {
                    Ok(img) => self.set_source(img, name),
                    Err(e) => self.status = format!("Could not open {}: {}", name, e),
                },
                Msg::Maps { key, maps } => {
                    self.maps = Some(maps);
                    self.maps_key = Some(key);
                    self.maps_rev += 1;
                    self.maps_building = None;
                }
                Msg::Preview { slot, epoch, req, ok, w, h, rgba } => {
                    if epoch != self.slot_epoch || slot >= self.slots.len() {
                        continue;
                    }
                    self.slots[slot].inflight = None;
                    if ok {
                        let img = egui::ColorImage::from_rgba_unmultiplied([w, h], &rgba);
                        let tex = ctx.load_texture(format!("preview{}", slot), img, egui::TextureOptions::LINEAR);
                        let px = 1.0 / req.ppmm;
                        self.slots[slot].tex = Some(TexInfo {
                            tex,
                            x0: req.region[0],
                            y0: req.region[1],
                            w_mm: w as f32 * px,
                            h_mm: h as f32 * px,
                        });
                        self.tone_mode = !req.dots;
                        self.slots[slot].have = Some(req);
                    }
                }
                Msg::Export { result } => {
                    self.export = None;
                    self.status = match result {
                        Ok(s) => s,
                        Err(e) => format!("Export failed: {}", e),
                    };
                }
            }
        }
    }

    fn maybe_build_maps(&mut self) {
        let Some(src) = self.source.clone() else { return };
        if self.maps_building.is_some() {
            return;
        }
        let key = MapsKey { source_rev: self.source_rev, tone: self.tone, work_res: self.work_res };
        if self.maps_key == Some(key) {
            return;
        }
        self.maps_building = Some(key);
        let tx = self.tx.clone();
        let ctx = self.ctx.clone();
        thread::spawn(move || {
            let maps = build_tone_maps(&src, &key.tone, key.work_res);
            let _ = tx.send(Msg::Maps { key, maps: Arc::new(maps) });
            ctx.request_repaint();
        });
    }

    fn reset_slots(&mut self) {
        for sl in &self.slots {
            if let Some((_, st)) = &sl.inflight {
                st.store(true, Ordering::Relaxed);
            }
        }
        self.slots.clear();
        self.slot_epoch += 1;
    }

    fn any_inflight(&self) -> bool {
        self.slots.iter().any(|s| s.inflight.is_some())
    }

    fn any_pending(&self) -> bool {
        self.slots.iter().any(|s| s.pending.is_some())
    }

    fn any_tex(&self) -> bool {
        self.slots.iter().any(|s| s.tex.is_some())
    }

    /// Top-left of each page in world millimetres. One page normally, one per plate in All plates.
    fn page_offsets(&self) -> Vec<(f32, f32)> {
        if self.view_mode != ViewMode::AllPlates {
            return vec![(0.0, 0.0)];
        }
        let (pw, ph) = self.page_mm();
        let n = self.plates.len().max(1);
        let cols = if n >= 4 { 2 } else { n };
        let gap = 0.08 * pw.max(ph);
        (0..n).map(|k| ((k % cols) as f32 * (pw + gap), (k / cols) as f32 * (ph + gap))).collect()
    }

    fn world_size(&self) -> (f32, f32) {
        let (pw, ph) = self.page_mm();
        let offs = self.page_offsets();
        let mx = offs.iter().map(|o| o.0).fold(0.0, f32::max);
        let my = offs.iter().map(|o| o.1).fold(0.0, f32::max);
        (mx + pw, my + ph)
    }

    fn slot_view(&self, k: usize) -> ViewMode {
        if self.view_mode == ViewMode::AllPlates {
            ViewMode::Plate(k)
        } else {
            self.view_mode
        }
    }

    fn spawn_preview(&mut self, slot: usize, req: PreviewReq) {
        let Some(maps) = self.maps.clone() else { return };
        let stale = Arc::new(AtomicBool::new(false));
        self.slots[slot].inflight = Some((req.clone(), stale.clone()));
        let epoch = self.slot_epoch;
        let tx = self.tx.clone();
        let ctx = self.ctx.clone();
        thread::spawn(move || {
            let (w, h, rgba) = render_job(&req, &maps, &stale);
            let ok = !stale.load(Ordering::Relaxed);
            let _ = tx.send(Msg::Preview { slot, epoch, req, ok, w, h, rgba });
            ctx.request_repaint();
        });
    }

    /// Make sure each visible page has a preview texture covering the visible part of it.
    /// `vis` is (x0, y0, x1, y1) in world millimetres.
    fn update_preview(&mut self, vis: [f32; 4], ppp: f32) {
        if self.maps_key.map(|k| k.source_rev) != Some(self.source_rev) {
            return;
        }
        let Some(maps) = self.maps.clone() else { return };
        let scene = self.scene();
        if maps.plates.len() != scene.plates.len() || self.view_mode == ViewMode::Original {
            return;
        }
        let offs = self.page_offsets();
        if self.slots.len() != offs.len() || self.slot_mode != self.view_mode {
            self.reset_slots();
            self.slots = (0..offs.len()).map(|_| Slot::default()).collect();
            self.slot_mode = self.view_mode;
        }
        let fast = self.drag.is_some();
        let (pw, ph) = (scene.page_w, scene.page_h);
        let pitch = scene.plates.iter().map(|p| 25.4 / p.params.lpi.max(1.0)).fold(f32::MAX, f32::min);
        let (maps_rev, aa, plate_colors, always_dots, zoom) =
            (self.maps_rev, self.aa, self.plate_colors, self.always_dots, self.view.zoom);

        for (k, &(ox, oy)) in offs.iter().enumerate() {
            let lv = [(vis[0] - ox).max(0.0), (vis[1] - oy).max(0.0), (vis[2] - ox).min(pw), (vis[3] - oy).min(ph)];
            if lv[2] <= lv[0] || lv[3] <= lv[1] {
                continue;
            }
            let (vw, vh) = (lv[2] - lv[0], lv[3] - lv[1]);
            let (ew, eh) = (vw * 1.2, vh * 1.2);
            let mut ppmm = zoom * ppp * if fast { 0.5 } else { 1.0 };
            let s =
                (4096.0 / (ew * ppmm)).min(4096.0 / (eh * ppmm)).min((7.0e6 / (ew * ppmm * eh * ppmm)).sqrt()).min(1.0);
            ppmm = (ppmm * s).max(0.05);
            ppmm = 2f32.powf((ppmm.log2() * 8.0).round() / 8.0);
            let dots = (always_dots || pitch * ppmm >= 3.5) && !fast;

            let x0 = (lv[0] - 0.1 * vw).max(0.0);
            let y0 = (lv[1] - 0.1 * vh).max(0.0);
            let x1 = (lv[2] + 0.1 * vw).min(pw);
            let y1 = (lv[3] + 0.1 * vh).min(ph);
            let wanted = PreviewReq {
                scene: scene.clone(),
                maps_rev,
                view: self.slot_view(k),
                region: [x0, y0, x1 - x0, y1 - y0],
                ppmm,
                dots,
                aa,
                plate_colors,
            };
            let slot = &mut self.slots[k];
            if let Some(h) = &slot.have {
                if h.covers(&wanted, lv) {
                    slot.pending = None;
                    continue;
                }
            }
            if let Some((r, stale)) = &slot.inflight {
                if r.covers(&wanted, lv) {
                    continue;
                }
                stale.store(true, Ordering::Relaxed);
            }
            slot.pending = Some(wanted);
        }
        // One render at a time: each already uses every core.
        if !self.any_inflight() {
            if let Some(k) = self.slots.iter().position(|s| s.pending.is_some()) {
                if let Some(req) = self.slots[k].pending.take() {
                    self.spawn_preview(k, req);
                }
            }
        }
    }

    fn start_export(&mut self) {
        let Some(maps) = self.maps.clone() else {
            self.status = "Open an image first.".to_string();
            return;
        };
        let scene = self.scene();
        if maps.plates.len() != scene.plates.len() || self.maps_key.map(|k| k.source_rev) != Some(self.source_rev) {
            self.status = "Still processing the image, try again in a moment.".to_string();
            return;
        }
        let stem0 = Path::new(&self.source_name)
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "separations".to_string());
        let Some(path) = rfd::FileDialog::new()
            .set_file_name(format!("{}_separations.pdf", stem0))
            .add_filter("PDF", &["pdf"])
            .save_file()
        else {
            return;
        };
        let dir = path.parent().map(|p| p.to_path_buf()).unwrap_or_default();
        let stem = path.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or(stem0);
        let spec = ExportSpec {
            page_w: scene.page_w,
            page_h: scene.page_h,
            dpi: self.exp_dpi,
            mirror: self.exp_mirror,
            pdf: if self.exp_pdf { Some(path) } else { None },
            png: if self.exp_png { Some((dir, stem.clone())) } else { None },
        };
        let rows = (scene.page_h / 25.4 * self.exp_dpi).round() as usize * scene.plates.len();
        let progress = Arc::new(AtomicUsize::new(0));
        let cancel = Arc::new(AtomicBool::new(false));
        self.export = Some(ExportState { progress: progress.clone(), total: rows.max(1), cancel: cancel.clone() });
        self.status = "Exporting…".to_string();
        let tx = self.tx.clone();
        let ctx = self.ctx.clone();
        thread::spawn(move || {
            let result = (|| -> Result<String, String> {
                let prep = scene.prepare();
                let evals = scene.evals(&prep, &maps);
                let n = scene.plates.len();
                let prims: Vec<Vec<Prim>> =
                    (0..n).map(|i| scene.marks.prims(scene.image, &scene.label(i, &stem))).collect();
                let outs: Vec<PlateOut> = (0..n)
                    .map(|i| PlateOut { name: &scene.plates[i].name, eval: &evals[i], prims: &prims[i] })
                    .collect();
                let files = export_all(&spec, &outs, &zlib, &progress, &cancel).map_err(|e| e.to_string())?;
                Ok(format!(
                    "Exported {} file(s), last: {}",
                    files.len(),
                    files.last().map(|p| p.display().to_string()).unwrap_or_default()
                ))
            })();
            let _ = tx.send(Msg::Export { result });
            ctx.request_repaint();
        });
    }

    // ------------------------------------------------------------------ view helpers

    fn zoom_to(&mut self, canvas: egui::Rect, x: f32, y: f32, w: f32, h: f32) {
        if canvas.width() < 10.0 || canvas.height() < 10.0 {
            self.view.fit_pending = true;
            return;
        }
        let z = (canvas.width() * 0.92 / w.max(1.0)).min(canvas.height() * 0.92 / h.max(1.0));
        self.view.zoom = z.clamp(0.05, 200.0);
        self.view.origin = vec2(
            canvas.width() * 0.5 - (x + w * 0.5) * self.view.zoom,
            canvas.height() * 0.5 - (y + h * 0.5) * self.view.zoom,
        );
        self.view.fit_pending = false;
    }

    fn zoom_about(&mut self, canvas: egui::Rect, c: Pos2, f: f32) {
        let z0 = self.view.zoom;
        let z1 = (z0 * f).clamp(0.05, 200.0);
        let p = (c - canvas.min - self.view.origin) / z0;
        self.view.origin = (c - canvas.min) - p * z1;
        self.view.zoom = z1;
    }

    /// Returns what a press at screen point `s` grabs, and the top-left of the page it landed on.
    fn hit_test(&self, s: Pos2, org: Pos2, z: f32) -> (DragKind, (f32, f32)) {
        if self.source.is_none() {
            return (DragKind::Pan, (0.0, 0.0));
        }
        let r = self.place;
        let offs = self.page_offsets();
        let (wx, wy) = ((s.x - org.x) / z, (s.y - org.y) / z);
        for &(ox, oy) in &offs {
            let c = [
                (r.x + ox, r.y + oy),
                (r.right() + ox, r.y + oy),
                (r.x + ox, r.bottom() + oy),
                (r.right() + ox, r.bottom() + oy),
            ];
            for (i, &(cx, cy)) in c.iter().enumerate() {
                if (pos2(org.x + cx * z, org.y + cy * z) - s).length() < 10.0 {
                    let a = c[3 - i];
                    return (DragKind::Resize { anchor: (a.0 - ox, a.1 - oy), aspect: r.w / r.h.max(1e-3) }, (ox, oy));
                }
            }
        }
        for &(ox, oy) in &offs {
            let (px, py) = (wx - ox, wy - oy);
            if r.contains(px, py) {
                return (DragKind::Move { off: (px - r.x, py - r.y) }, (ox, oy));
            }
        }
        (DragKind::Pan, (0.0, 0.0))
    }

    // ------------------------------------------------------------------ UI

    fn controls(&mut self, ui: &mut egui::Ui) {
        let unit = self.unit;
        ui.horizontal(|ui| {
            if ui.button("Open image…").clicked() {
                self.open_dialog();
            }
            if ui.button("Rotate left").clicked() {
                self.rotate(false);
            }
            if ui.button("Rotate right").clicked() {
                self.rotate(true);
            }
        });
        let name_label =
            if self.source_name.is_empty() { "No image loaded".to_string() } else { self.source_name.clone() };
        ui.label(name_label);
        ui.add_space(4.0);

        egui::CollapsingHeader::new("Page").default_open(true).show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label("Units");
                ui.selectable_value(&mut self.unit, Unit::Inch, "in");
                ui.selectable_value(&mut self.unit, Unit::Mm, "mm");
                ui.checkbox(&mut self.landscape, "Landscape");
            });
            let cur = if self.page_idx < PAGE_SIZES.len() { PAGE_SIZES[self.page_idx].name } else { "Custom" };
            egui::ComboBox::from_id_salt("page_size").selected_text(cur).width(220.0).show_ui(ui, |ui| {
                for (i, p) in PAGE_SIZES.iter().enumerate() {
                    ui.selectable_value(&mut self.page_idx, i, p.name);
                }
                ui.selectable_value(&mut self.page_idx, PAGE_SIZES.len(), "Custom");
            });
            if self.page_idx >= PAGE_SIZES.len() {
                ui.horizontal(|ui| {
                    dv_mm(ui, &mut self.custom_w, unit, 20.0, 3000.0);
                    ui.label("x");
                    dv_mm(ui, &mut self.custom_h, unit, 20.0, 3000.0);
                });
            }
            let (pw, ph) = self.page_mm();
            ui.label(format!("{:.2} x {:.2} {}", unit.from_mm(pw), unit.from_mm(ph), unit.suffix().trim()));
        });

        egui::CollapsingHeader::new("Image placement").default_open(true).show(ui, |ui| {
            let Some(src) = self.source.clone() else {
                ui.label("Open an image to place it.");
                return;
            };
            egui::Grid::new("place_grid").num_columns(2).show(ui, |ui| {
                ui.label("X");
                dv_mm(ui, &mut self.place.x, unit, -5000.0, 5000.0);
                ui.end_row();
                ui.label("Y");
                dv_mm(ui, &mut self.place.y, unit, -5000.0, 5000.0);
                ui.end_row();
                ui.label("Width");
                if dv_mm(ui, &mut self.place.w, unit, 5.0, 5000.0) && self.lock_aspect {
                    self.place.h = self.place.w / self.src_aspect;
                }
                ui.end_row();
                ui.label("Height");
                if dv_mm(ui, &mut self.place.h, unit, 5.0, 5000.0) && self.lock_aspect {
                    self.place.w = self.place.h * self.src_aspect;
                }
                ui.end_row();
                ui.label("Resolution");
                let mut ppi = src.w as f32 / (self.place.w / 25.4).max(1e-3);
                if ui.add(egui::DragValue::new(&mut ppi).speed(1.0).range(10.0..=2400.0).suffix(" ppi")).changed() {
                    let ratio = self.place.h / self.place.w.max(1e-3);
                    self.place.w = src.w as f32 / ppi * 25.4;
                    self.place.h = if self.lock_aspect { self.place.w / self.src_aspect } else { self.place.w * ratio };
                }
                ui.end_row();
            });
            ui.horizontal(|ui| {
                ui.checkbox(&mut self.lock_aspect, "Lock aspect");
                if ui.small_button("Reset aspect").clicked() {
                    self.place.h = self.place.w / self.src_aspect;
                }
            });
            ui.horizontal_wrapped(|ui| {
                if ui.button("Fit inside marks").clicked() {
                    self.fit_image(false);
                }
                if ui.button("Fill page").clicked() {
                    self.fit_image(true);
                }
                if ui.button("Center").clicked() {
                    let (pw, ph) = self.page_mm();
                    self.place.x = (pw - self.place.w) * 0.5;
                    self.place.y = (ph - self.place.h) * 0.5;
                }
            });
            ui.label("Drag the image to move it; drag a corner to resize.");
        });

        egui::CollapsingHeader::new("Separation").default_open(true).show(ui, |ui| {
            let mut mode = self.tone.mode;
            egui::ComboBox::from_id_salt("sep_mode")
                .selected_text(if mode == SepMode::Mono { "Single plate (greyscale)" } else { "CMYK (4 plates)" })
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut mode, SepMode::Mono, "Single plate (greyscale)");
                    ui.selectable_value(&mut mode, SepMode::Cmyk, "CMYK (4 plates)");
                });
            if mode != self.tone.mode {
                self.tone.mode = mode;
                self.reset_plates();
            }
            ui.add(egui::Slider::new(&mut self.tone.brightness, -1.0..=1.0).text("Brightness"));
            ui.add(egui::Slider::new(&mut self.tone.contrast, -1.0..=1.0).text("Contrast"));
            ui.add(egui::Slider::new(&mut self.tone.gamma, 0.3..=3.0).logarithmic(true).text("Gamma"));
            if self.tone.mode == SepMode::Cmyk {
                ui.add(egui::Slider::new(&mut self.tone.gcr, 0.0..=1.0).text("Black generation"));
            }
            egui::ComboBox::from_id_salt("work_res")
                .selected_text(format!("Working resolution: {} px", self.work_res))
                .show_ui(ui, |ui| {
                    for r in WORK_RES_CHOICES {
                        ui.selectable_value(&mut self.work_res, r, format!("{} px (long side)", r));
                    }
                });
            ui.label("Plain RGB to CMYK, no ICC profiles.");
        });

        egui::CollapsingHeader::new("Screen (halftone)").default_open(true).show(ui, |ui| {
            ui.add(egui::Slider::new(&mut self.lpi, 10.0..=150.0).text("Lines per inch"));
            ui.horizontal(|ui| {
                ui.label("Mesh");
                ui.add(egui::DragValue::new(&mut self.mesh).speed(1.0).range(40.0..=500.0).suffix(" tpi"));
                ui.label(format!("suggests {:.0}-{:.0} lpi", self.mesh / 5.0, self.mesh / 4.0));
                if ui.small_button("Use").clicked() {
                    self.lpi = (self.mesh / 4.5).round();
                }
            });
            egui::ComboBox::from_id_salt("shape").selected_text(self.shape.name()).show_ui(ui, |ui| {
                for s in DotShape::ALL {
                    ui.selectable_value(&mut self.shape, s, s.name());
                }
            });
            if self.shape == DotShape::Ellipse {
                ui.add(egui::Slider::new(&mut self.elong, 1.0..=2.5).text("Elongation"));
            }
            if self.tone.mode == SepMode::Cmyk && self.plates.len() == 4 {
                let mut chosen = None;
                egui::ComboBox::from_id_salt("angle_preset").selected_text("Angle preset…").show_ui(ui, |ui| {
                    for (name, a) in ANGLE_PRESETS {
                        if ui.selectable_label(false, name).clicked() {
                            chosen = Some(a);
                        }
                    }
                });
                if let Some(a) = chosen {
                    for (p, v) in self.plates.iter_mut().zip(a) {
                        p.angle = v;
                    }
                }
            }
        });

        egui::CollapsingHeader::new("Plates").default_open(true).show(ui, |ui| {
            let (g_lpi, g_shape) = (self.lpi, self.shape);
            for (i, p) in self.plates.iter_mut().enumerate() {
                egui::CollapsingHeader::new(format!("Plate {}", p.name)).id_salt(("plate", i)).default_open(true).show(
                    ui,
                    |ui| {
                        ui.horizontal(|ui| {
                            ui.checkbox(&mut p.visible, "Show in composite");
                            ui.label("Ink");
                            ui.color_edit_button_srgb(&mut p.ink);
                        });
                        ui.add(egui::Slider::new(&mut p.angle, 0.0..=180.0).text("Angle").suffix("°"));
                        ui.add(egui::Slider::new(&mut p.dot_gain, 0.0..=25.0).text("Dot gain comp.").suffix("%"));
                        ui.add(egui::Slider::new(&mut p.min_dot, 0.0..=20.0).text("Min dot").suffix("%"));
                        ui.add(egui::Slider::new(&mut p.max_dot, 80.0..=100.0).text("Max dot").suffix("%"));
                        ui.add(egui::Slider::new(&mut p.dropout, 0.0..=10.0).text("Highlight dropout").suffix("%"));
                        ui.checkbox(&mut p.invert, "Invert (negative)");
                        let mut own_lpi = p.lpi_override.is_some();
                        if ui.checkbox(&mut own_lpi, "Own lines per inch").changed() {
                            p.lpi_override = if own_lpi { Some(g_lpi) } else { None };
                        }
                        if let Some(l) = &mut p.lpi_override {
                            ui.add(egui::Slider::new(l, 10.0..=150.0).text("LPI"));
                        }
                        let mut own_shape = p.shape_override.is_some();
                        if ui.checkbox(&mut own_shape, "Own dot shape").changed() {
                            p.shape_override = if own_shape { Some(g_shape) } else { None };
                        }
                        if let Some(s) = &mut p.shape_override {
                            egui::ComboBox::from_id_salt(("pshape", i)).selected_text(s.name()).show_ui(ui, |ui| {
                                for c in DotShape::ALL {
                                    ui.selectable_value(s, c, c.name());
                                }
                            });
                        }
                    },
                );
            }
        });

        egui::CollapsingHeader::new("Printer's marks").default_open(true).show(ui, |ui| {
            ui.checkbox(&mut self.marks.crop, "Crop marks");
            ui.checkbox(&mut self.marks.reg, "Registration marks");
            ui.checkbox(&mut self.marks.label, "Plate label (angle, lpi, shape)");
            ui.checkbox(&mut self.marks.wedge, "Tone-step wedge per plate");
            ui.horizontal(|ui| {
                ui.label("Offset");
                dv_mm(ui, &mut self.marks.offset, unit, 0.0, 30.0);
                ui.label("Crop length");
                dv_mm(ui, &mut self.marks.crop_len, unit, 1.0, 30.0);
            });
            ui.horizontal(|ui| {
                ui.label("Target");
                dv_mm(ui, &mut self.marks.reg_size, unit, 2.0, 20.0);
                ui.label("Wedge row");
                dv_mm(ui, &mut self.marks.wedge_h, unit, 2.0, 12.0);
            });
            ui.checkbox(&mut self.show_marks, "Show marks in preview");
            let any = self.marks.crop || self.marks.reg || self.marks.label || self.marks.wedge;
            let (pw, ph) = self.page_mm();
            let m = self.marks.margins(self.plates.len());
            let r = self.place;
            if any
                && (r.x - m[0] < -0.01
                    || r.y - m[1] < -0.01
                    || r.right() + m[2] > pw + 0.01
                    || r.bottom() + m[3] > ph + 0.01)
            {
                ui.colored_label(Color32::from_rgb(220, 60, 40), "Marks run off the page. Use Fit inside marks.");
            }
        });

        egui::CollapsingHeader::new("Preview").default_open(true).show(ui, |ui| {
            let names: Vec<String> = self.plates.iter().map(|p| p.name.clone()).collect();
            let label = match self.view_mode {
                ViewMode::Composite => "Composite (overprint)".to_string(),
                ViewMode::Plate(i) => format!("Plate {} (film positive)", names.get(i).cloned().unwrap_or_default()),
                ViewMode::AllPlates => "All plates (separate pages)".to_string(),
                ViewMode::Original => "Original image".to_string(),
            };
            egui::ComboBox::from_id_salt("view_mode").selected_text(label).show_ui(ui, |ui| {
                ui.selectable_value(&mut self.view_mode, ViewMode::Composite, "Composite (overprint)");
                for (i, n) in names.iter().enumerate() {
                    ui.selectable_value(
                        &mut self.view_mode,
                        ViewMode::Plate(i),
                        format!("Plate {} (film positive)", n),
                    );
                }
                ui.selectable_value(&mut self.view_mode, ViewMode::AllPlates, "All plates (separate pages)");
                ui.selectable_value(&mut self.view_mode, ViewMode::Original, "Original image");
            });
            ui.checkbox(&mut self.plate_colors, "Show plates in ink colour (instead of black film)");
            ui.checkbox(&mut self.aa, "Smooth dot edges (2x2 samples)");
            ui.checkbox(&mut self.always_dots, "Always draw dots (even when zoomed out)");
            ui.horizontal(|ui| {
                if ui.button("Fit page").clicked() {
                    self.view.fit_pending = true;
                }
                if ui.button("Zoom to image").clicked() {
                    let (c, r) = (self.last_canvas, self.place);
                    self.zoom_to(c, r.x, r.y, r.w, r.h);
                }
                if ui.button("Actual size").clicked() {
                    let c = self.last_canvas;
                    let center = pos2(
                        (c.width() * 0.5 - self.view.origin.x) / self.view.zoom.max(0.01),
                        (c.height() * 0.5 - self.view.origin.y) / self.view.zoom.max(0.01),
                    );
                    let z = 96.0 / 25.4;
                    self.view.zoom = z;
                    self.view.origin = vec2(c.width() * 0.5 - center.x * z, c.height() * 0.5 - center.y * z);
                }
            });
            ui.label("Mouse wheel zooms, middle/right drag pans.");
        });

        egui::CollapsingHeader::new("Export").default_open(true).show(ui, |ui| {
            egui::ComboBox::from_id_salt("dpi").selected_text(format!("{} dpi", self.exp_dpi)).show_ui(ui, |ui| {
                for d in DPI_CHOICES {
                    ui.selectable_value(&mut self.exp_dpi, d, format!("{} dpi", d));
                }
            });
            ui.checkbox(&mut self.exp_pdf, "PDF (one page per plate, marks as vectors)");
            ui.checkbox(&mut self.exp_png, "1-bit PNG per plate (marks without text)");
            ui.checkbox(&mut self.exp_mirror, "Mirror (wrong-reading, for emulsion-down film)");
            let (pw, ph) = self.page_mm();
            let mb = (pw / 25.4 * self.exp_dpi) * (ph / 25.4 * self.exp_dpi) / 8.0 / 1.0e6;
            ui.label(format!(
                "Raster per plate: {:.0} x {:.0} px, {:.0} MB in memory",
                pw / 25.4 * self.exp_dpi,
                ph / 25.4 * self.exp_dpi,
                mb
            ));
            if let Some(e) = &self.export {
                let f = (e.progress.load(Ordering::Relaxed) as f32 / e.total as f32).clamp(0.0, 1.0);
                ui.add(egui::ProgressBar::new(f).show_percentage());
                if ui.button("Cancel").clicked() {
                    e.cancel.store(true, Ordering::Relaxed);
                }
            }
            let can_export = self.export.is_none() && self.source.is_some() && (self.exp_pdf || self.exp_png);
            if self.export.is_none() && ui.add_enabled(can_export, egui::Button::new("Export separations…")).clicked()
            {
                self.start_export();
            }
        });
    }

    fn status_bar(&self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label(&self.status);
            ui.separator();
            if let Some((x, y)) = self.cursor_mm {
                ui.label(format!("{:.2}, {:.2}{}", self.unit.from_mm(x), self.unit.from_mm(y), self.unit.suffix()));
                ui.separator();
            }
            if let Some(src) = &self.source {
                let ppi = src.w as f32 / (self.place.w / 25.4).max(1e-3);
                ui.label(format!("{:.0} ppi effective", ppi));
                if ppi < 150.0 {
                    ui.colored_label(Color32::from_rgb(220, 140, 0), "low resolution");
                }
                ui.separator();
            }
            if self.tone_mode && self.source.is_some() && self.view_mode != ViewMode::Original {
                ui.label("Zoomed out: showing tone. Zoom in to see the dots.");
            }
            if self.maps_building.is_some() || self.any_inflight() || self.export.is_some() {
                ui.add(egui::Spinner::new());
            }
        });
    }

    fn canvas(&mut self, ui: &mut egui::Ui) {
        let size = ui.available_size();
        let (resp, painter) = ui.allocate_painter(size, egui::Sense::click_and_drag());
        let rect = resp.rect;
        self.last_canvas = rect;
        let ppp = ui.ctx().pixels_per_point();
        let (pw, ph) = self.page_mm();
        if self.prev_view_mode != self.view_mode {
            if self.prev_view_mode == ViewMode::AllPlates || self.view_mode == ViewMode::AllPlates {
                self.view.fit_pending = true;
            }
            self.prev_view_mode = self.view_mode;
        }
        let offs = self.page_offsets();
        if self.view.fit_pending || self.view.zoom <= 0.0 {
            let (ww, wh) = self.world_size();
            self.zoom_to(rect, 0.0, 0.0, ww, wh);
        }

        // zoom
        if resp.hovered() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            let pinch = ui.input(|i| i.zoom_delta());
            let factor = (scroll * 0.0025).exp() * pinch;
            if (factor - 1.0).abs() > 1e-4 {
                if let Some(c) = resp.hover_pos() {
                    self.zoom_about(rect, c, factor);
                }
            }
        }

        // drag
        {
            let z = self.view.zoom;
            let org = rect.min + self.view.origin;
            let to_w = |s: Pos2| ((s.x - org.x) / z, (s.y - org.y) / z);
            if resp.drag_started_by(egui::PointerButton::Primary) {
                if let Some(s) = resp.interact_pointer_pos() {
                    let (kind, off) = self.hit_test(s, org, z);
                    self.drag = Some(kind);
                    self.drag_off = off;
                }
            }
            if resp.dragged_by(egui::PointerButton::Primary) {
                if let (Some(kind), Some(s)) = (self.drag, resp.interact_pointer_pos()) {
                    let (wx, wy) = to_w(s);
                    let (px, py) = (wx - self.drag_off.0, wy - self.drag_off.1);
                    match kind {
                        DragKind::Pan => self.view.origin += resp.drag_delta(),
                        DragKind::Move { off } => {
                            self.place.x = px - off.0;
                            self.place.y = py - off.1;
                        }
                        DragKind::Resize { anchor, aspect } => {
                            let (ax, ay) = anchor;
                            let w = (px - ax).abs().max(5.0);
                            let h = if self.lock_aspect { w / aspect } else { (py - ay).abs().max(5.0) };
                            self.place =
                                MmRect::new(if px < ax { ax - w } else { ax }, if py < ay { ay - h } else { ay }, w, h);
                        }
                    }
                }
            }
            if resp.dragged_by(egui::PointerButton::Middle) || resp.dragged_by(egui::PointerButton::Secondary) {
                self.view.origin += resp.drag_delta();
            }
            if resp.drag_stopped() {
                self.drag = None;
            }
        }

        let z = self.view.zoom;
        let org = rect.min + self.view.origin;
        let to_s = |x: f32, y: f32| pos2(org.x + x * z, org.y + y * z);
        let to_p = |s: Pos2| ((s.x - org.x) / z, (s.y - org.y) / z);
        self.cursor_mm = resp.hover_pos().and_then(|p| {
            let (wx, wy) = to_p(p);
            if offs.len() == 1 {
                return Some((wx, wy));
            }
            offs.iter().find(|o| wx >= o.0 && wx < o.0 + pw && wy >= o.1 && wy < o.1 + ph).map(|o| (wx - o.0, wy - o.1))
        });

        // pages
        painter.rect_filled(rect, 0.0, Color32::from_gray(58));
        let pages: Vec<egui::Rect> =
            offs.iter().map(|&(ox, oy)| egui::Rect::from_min_max(to_s(ox, oy), to_s(ox + pw, oy + ph))).collect();
        for page in &pages {
            painter.rect_filled(page.translate(vec2(4.0, 4.0)), 0.0, Color32::from_black_alpha(70));
            painter.rect_filled(*page, 0.0, Color32::WHITE);
        }

        let place = self.place;
        let img_rect = |ox: f32, oy: f32| {
            egui::Rect::from_min_max(to_s(ox + place.x, oy + place.y), to_s(ox + place.right(), oy + place.bottom()))
        };
        let uv = egui::Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0));

        if self.source.is_none() {
            painter.text(
                pages[0].center(),
                egui::Align2::CENTER_CENTER,
                "Open an image (Ctrl+O) or drop one here",
                egui::FontId::proportional(18.0),
                Color32::from_gray(120),
            );
        } else if self.view_mode == ViewMode::Original {
            let rev = self.source_rev;
            if self.orig_tex.as_ref().map(|t| t.0) != Some(rev) {
                if let Some(src) = &self.source {
                    let small = src.downscaled(1536);
                    let img = egui::ColorImage::from_rgb([small.w, small.h], &small.rgb);
                    let t = ui.ctx().load_texture("original", img, egui::TextureOptions::LINEAR);
                    self.orig_tex = Some((rev, t));
                }
            }
            if let Some((_, t)) = &self.orig_tex {
                painter.image(t.id(), img_rect(0.0, 0.0), uv, Color32::WHITE);
            }
        } else {
            let a = to_p(rect.min);
            let b = to_p(rect.max);
            self.update_preview([a.0, a.1, b.0, b.1], ppp);
            for (k, slot) in self.slots.iter().enumerate() {
                if let (Some(t), Some(&(ox, oy))) = (&slot.tex, offs.get(k)) {
                    let r = egui::Rect::from_min_max(
                        to_s(ox + t.x0, oy + t.y0),
                        to_s(ox + t.x0 + t.w_mm, oy + t.y0 + t.h_mm),
                    );
                    painter.image(t.tex.id(), r, uv, Color32::WHITE);
                }
            }
            if self.maps_building.is_some() && !self.any_tex() {
                painter.text(
                    pages[0].center(),
                    egui::Align2::CENTER_CENTER,
                    "Processing image…",
                    egui::FontId::proportional(18.0),
                    Color32::from_gray(120),
                );
            }
        }

        // captions above each page in the All plates view
        if self.view_mode == ViewMode::AllPlates {
            for (k, page) in pages.iter().enumerate() {
                if let Some(p) = self.plates.get(k) {
                    let col = if self.plate_colors {
                        Color32::from_rgb(p.ink[0], p.ink[1], p.ink[2])
                    } else {
                        Color32::from_gray(220)
                    };
                    painter.text(
                        page.left_top() - vec2(0.0, 8.0),
                        egui::Align2::LEFT_BOTTOM,
                        format!("Plate {}  {:.1} deg", p.name, p.angle),
                        egui::FontId::proportional(16.0),
                        col,
                    );
                }
            }
        }

        // marks
        if self.show_marks && self.source.is_some() {
            let scene = self.scene();
            for (k, &(ox, oy)) in offs.iter().enumerate() {
                let label = match self.view_mode {
                    ViewMode::AllPlates if k < self.plates.len() => scene.label(k, &self.source_name),
                    ViewMode::Plate(i) if i < self.plates.len() => scene.label(i, &self.source_name),
                    _ => {
                        let parts: Vec<String> =
                            scene.plates.iter().map(|p| format!("{} {:.1}deg", p.name, p.params.angle_deg)).collect();
                        format!("{}  {:.0} lpi  {}", parts.join("  "), self.lpi, self.shape.name())
                    }
                };
                for pr in self.marks.prims(self.place, &label) {
                    match pr {
                        Prim::Line { x1, y1, x2, y2, w } => {
                            painter.line_segment(
                                [to_s(ox + x1, oy + y1), to_s(ox + x2, oy + y2)],
                                Stroke::new((w * z).max(1.0), Color32::BLACK),
                            );
                        }
                        Prim::Ring { cx, cy, r, w } => {
                            painter.circle_stroke(
                                to_s(ox + cx, oy + cy),
                                r * z,
                                Stroke::new((w * z).max(1.0), Color32::BLACK),
                            );
                        }
                        Prim::Text { x, y, size_pt, text } => {
                            painter.text(
                                to_s(ox + x, oy + y),
                                egui::Align2::LEFT_BOTTOM,
                                text,
                                egui::FontId::proportional((size_pt * 25.4 / 72.0 * z).clamp(7.0, 40.0)),
                                Color32::BLACK,
                            );
                        }
                    }
                }
            }
        }

        // selection outline + handles on every page (they all edit the same placement)
        if self.source.is_some() {
            let blue = Color32::from_rgb(0, 120, 255);
            for &(ox, oy) in &offs {
                let ir = img_rect(ox, oy);
                painter.rect_stroke(ir, 0.0, Stroke::new(1.0, blue), egui::StrokeKind::Outside);
                for c in [ir.left_top(), ir.right_top(), ir.left_bottom(), ir.right_bottom()] {
                    let h = egui::Rect::from_center_size(c, vec2(9.0, 9.0));
                    painter.rect_filled(h, 0.0, Color32::WHITE);
                    painter.rect_stroke(h, 0.0, Stroke::new(1.0, blue), egui::StrokeKind::Inside);
                }
            }
        }
        for page in &pages {
            painter.rect_stroke(*page, 0.0, Stroke::new(1.0, Color32::from_gray(30)), egui::StrokeKind::Outside);
        }
    }
}

impl eframe::App for HalftoneApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll(ctx);

        let dropped = ctx.input(|i| i.raw.dropped_files.clone());
        for f in dropped {
            if let Some(p) = f.path {
                self.load_path(p);
                break;
            }
        }
        if ctx.input(|i| i.modifiers.command && i.key_pressed(egui::Key::O)) {
            self.open_dialog();
        }
        if let ViewMode::Plate(i) = self.view_mode {
            if i >= self.plates.len() {
                self.view_mode = ViewMode::Composite;
            }
        }

        self.maybe_build_maps();

        egui::TopBottomPanel::bottom("status").show(ctx, |ui| self.status_bar(ui));
        egui::SidePanel::left("controls").default_width(340.0).show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| self.controls(ui));
        });
        egui::CentralPanel::default().show(ctx, |ui| self.canvas(ui));

        if self.maps_building.is_some() || self.any_inflight() || self.export.is_some() || self.any_pending() {
            ctx.request_repaint_after(Duration::from_millis(30));
        }
    }
}
