//! egui front end: page canvas, menus, controls, background preview/export jobs.
#![allow(deprecated)]
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
use crate::engine::spot::{DotShape, SpotTable};
use crate::engine::tonemap::{build_tone_maps, SepMode, SourceImage, ToneMaps, ToneSettings};
use crate::gpu::{self, FrameDesc, GpuInk, GpuResources, LayerDesc, PageCallback};
use crate::icons::{icon_button, Icon};

const ANGLE_PRESETS: [(&str, [f32; 4]); 2] = [
    ("Offset standard (C15 M75 Y0 K45)", [15.0, 75.0, 0.0, 45.0]),
    ("Alternate (C105 M75 Y90 K45)", [105.0, 75.0, 90.0, 45.0]),
];
const DPI_CHOICES: [f32; 5] = [600.0, 1200.0, 1800.0, 2400.0, 2540.0];
const WORK_RES_CHOICES: [usize; 5] = [1024, 2048, 3072, 4096, 6144];
const UI_SCALES: [f32; 6] = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5];
const RULER_SIZE: f32 = 22.0;
const SCROLL_SIZE: f32 = 14.0;
const MIN_ZOOM: f32 = 0.05;
const MAX_ZOOM: f32 = 200.0;
/// Points per mm at "100%" (96 dpi).
const ACTUAL_ZOOM: f32 = 96.0 / 25.4;
/// Screen distance (points) within which a press grabs an image corner.
const HANDLE_GRAB: f32 = 15.0;
const FIT_CHOICES: [(FitKind, &str); 4] = [
    (FitKind::Page, "Fit to page"),
    (FitKind::Width, "Fit width"),
    (FitKind::Height, "Fit height"),
    (FitKind::Fill, "Fill page"),
];

#[derive(Clone, Copy, PartialEq)]
enum Tool {
    Select,
    ZoomBox,
    Hand,
}

#[derive(Clone, Copy, PartialEq)]
enum Tab {
    Layout,
    Separation,
    Plates,
    Export,
}

/// How the pages of the All plates view are arranged.
#[derive(Clone, Copy, PartialEq, Debug)]
enum PageLayout {
    Auto,
    Grid,
    Row,
    Column,
}

impl PageLayout {
    const ALL: [PageLayout; 4] = [PageLayout::Auto, PageLayout::Grid, PageLayout::Row, PageLayout::Column];

    fn name(self) -> &'static str {
        match self {
            PageLayout::Auto => "Automatic (follows zoom)",
            PageLayout::Grid => "2 x 2 grid",
            PageLayout::Row => "One row",
            PageLayout::Column => "One column",
        }
    }

    fn key(self) -> &'static str {
        match self {
            PageLayout::Auto => "auto",
            PageLayout::Grid => "grid",
            PageLayout::Row => "row",
            PageLayout::Column => "column",
        }
    }

    fn from_key(k: &str) -> Option<PageLayout> {
        PageLayout::ALL.into_iter().find(|l| l.key() == k)
    }
}

/// How the image is sized onto the page. Always keeps the image (and, except for Fill, its marks)
/// inside the safe margin.
#[derive(Clone, Copy, PartialEq, Debug)]
enum FitKind {
    Page,
    Width,
    Height,
    Fill,
}

impl FitKind {
    fn help(self) -> &'static str {
        match self {
            FitKind::Page => "Whole image on the page, with its printer's marks inside the safe margin",
            FitKind::Width => "Image as wide as the printable area (marks included)",
            FitKind::Height => "Image as tall as the printable area (marks included)",
            FitKind::Fill => "Cover the whole page edge to edge (marks will not fit)",
        }
    }
}

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
    GpuInk { rev: u64, src_rev: u64, ink: Option<Arc<GpuInk>> },
    Export { result: Result<String, String> },
}

#[derive(Clone, Copy)]
enum DragKind {
    Pan,
    Move { off: (f32, f32) },
    Resize { anchor: (f32, f32), aspect: f32, corner: usize },
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

    fit_kind: Option<FitKind>,
    fit_key_last: [f32; 8],
    safe_margin: f32,
    show_safe: bool,

    // view
    tool: Tool,
    tab: Tab,
    sel_plate: usize,
    layout: PageLayout,
    cols_res: usize,
    zoom_dirty: bool,
    show_rulers: bool,
    view_fit: bool,
    last_canvas_size: Vec2,
    zoom_start: Option<Pos2>,
    zoom_cur: Option<Pos2>,
    ui_scale: f32,
    show_about: bool,
    saved_settings: String,

    // GPU preview
    gpu_avail: bool,
    gpu_enabled: bool,
    gpu_note: String,
    gpu_srgb: bool,
    gpu_ink: Option<Arc<GpuInk>>,
    gpu_ink_rev: u64,
    gpu_ink_src: u64,
    gpu_building: Option<u64>,
    gpu_spots: Arc<Vec<f32>>,
    gpu_spots_ver: u64,
    gpu_spot_key: Vec<(DotShape, u32)>,
    gpu_spot_params: Vec<(f32, f32)>,
    gpu_prev: bool,
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
            fit_kind: Some(FitKind::Page),
            fit_key_last: [0.0; 8],
            safe_margin: 6.35,
            show_safe: true,
            tool: Tool::Select,
            tab: Tab::Layout,
            sel_plate: 0,
            layout: PageLayout::Auto,
            cols_res: 2,
            zoom_dirty: false,
            show_rulers: true,
            view_fit: false,
            last_canvas_size: Vec2::ZERO,
            zoom_start: None,
            zoom_cur: None,
            ui_scale: 1.25,
            show_about: false,
            saved_settings: String::new(),
            gpu_avail: false,
            gpu_enabled: !std::env::args().any(|a| a == "--cpu-preview"),
            gpu_note: String::new(),
            gpu_srgb: false,
            gpu_ink: None,
            gpu_ink_rev: u64::MAX,
            gpu_ink_src: u64::MAX,
            gpu_building: None,
            gpu_spots: Arc::new(Vec::new()),
            gpu_spots_ver: 0,
            gpu_spot_key: Vec::new(),
            gpu_spot_params: Vec::new(),
            gpu_prev: false,
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
        s.load_settings();
        match cc.wgpu_render_state.as_ref() {
            Some(rs) => match GpuResources::new(&rs.device, rs.target_format) {
                Ok(res) => {
                    rs.renderer.write().callback_resources.insert(res);
                    s.gpu_avail = true;
                    s.gpu_srgb = rs.target_format.is_srgb();
                }
                Err(e) => s.gpu_note = format!("GPU preview unavailable: {}", e),
            },
            None => s.gpu_note = "GPU preview unavailable: no wgpu render state".to_string(),
        }
        s.fit_key_last = s.fit_key();
        cc.egui_ctx.options_mut(|o| o.zoom_with_keyboard = false);
        cc.egui_ctx.set_zoom_factor(s.ui_scale);
        let mut style = (*cc.egui_ctx.style()).clone();
        use egui::{FontFamily, FontId, TextStyle};
        style.text_styles.insert(TextStyle::Body, FontId::new(15.0, FontFamily::Proportional));
        style.text_styles.insert(TextStyle::Button, FontId::new(15.0, FontFamily::Proportional));
        style.text_styles.insert(TextStyle::Heading, FontId::new(22.0, FontFamily::Proportional));
        style.text_styles.insert(TextStyle::Small, FontId::new(13.0, FontFamily::Proportional));
        style.text_styles.insert(TextStyle::Monospace, FontId::new(14.0, FontFamily::Monospace));
        style.spacing.item_spacing = vec2(8.0, 6.0);
        style.spacing.interact_size.y = 26.0;
        style.spacing.slider_width = 170.0;
        cc.egui_ctx.set_style(style);
        s
    }

    fn fit_key(&self) -> [f32; 8] {
        let (pw, ph) = self.page_mm();
        let m = self.marks.margins(self.plates.len());
        [pw, ph, m[0], m[1], m[2], m[3], self.safe_margin, self.plates.len() as f32]
    }

    fn settings_text(&self) -> String {
        format!(
            "ui_scale={}\nunit={}\nlayout={}\nsafe_margin={}\nrenderer={}\nshow_safe={}\nrulers={}\nshow_marks={}\nplate_colors={}\naa={}\nalways_dots={}\n",
            self.ui_scale,
            if self.unit == Unit::Mm { "mm" } else { "in" },
            self.layout.key(),
            self.safe_margin,
            if self.gpu_enabled { "gpu" } else { "cpu" },
            self.show_safe,
            self.show_rulers,
            self.show_marks,
            self.plate_colors,
            self.aa,
            self.always_dots
        )
    }

    fn load_settings(&mut self) {
        let Some(text) = settings_path().and_then(|p| std::fs::read_to_string(p).ok()) else { return };
        for line in text.lines() {
            let Some((k, v)) = line.split_once('=') else { continue };
            let b = v == "true";
            match k {
                "ui_scale" => {
                    if let Ok(x) = v.parse::<f32>() {
                        self.ui_scale = x.clamp(0.75, 3.0);
                    }
                }
                "unit" => self.unit = if v == "mm" { Unit::Mm } else { Unit::Inch },
                "layout" => {
                    if let Some(l) = PageLayout::from_key(v) {
                        self.layout = l;
                    }
                }
                "safe_margin" => {
                    if let Ok(x) = v.parse::<f32>() {
                        self.safe_margin = x.clamp(0.0, 50.0);
                    }
                }
                "renderer" => self.gpu_enabled = v != "cpu" && !std::env::args().any(|a| a == "--cpu-preview"),
                "show_safe" => self.show_safe = b,
                "rulers" => self.show_rulers = b,
                "show_marks" => self.show_marks = b,
                "plate_colors" => self.plate_colors = b,
                "aa" => self.aa = b,
                "always_dots" => self.always_dots = b,
                _ => {}
            }
        }
        self.saved_settings = self.settings_text();
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

    fn fit_image(&mut self, kind: FitKind) {
        let Some(src) = &self.source else { return };
        let (pw, ph) = self.page_mm();
        // Marks sit outside the image, so the image area is the page minus the safe margin minus the marks.
        let m = if kind == FitKind::Fill {
            [0.0; 4]
        } else {
            let mm = self.marks.margins(self.plates.len());
            let s = self.safe_margin;
            [mm[0] + s, mm[1] + s, mm[2] + s, mm[3] + s]
        };
        let aw = (pw - m[0] - m[2]).max(10.0);
        let ah = (ph - m[1] - m[3]).max(10.0);
        let a = src.w as f32 / src.h as f32;
        let (w, h) = match kind {
            FitKind::Page => {
                if aw / a <= ah {
                    (aw, aw / a)
                } else {
                    (ah * a, ah)
                }
            }
            FitKind::Fill => {
                if aw / a >= ah {
                    (aw, aw / a)
                } else {
                    (ah * a, ah)
                }
            }
            FitKind::Width => (aw, aw / a),
            FitKind::Height => (ah * a, ah),
        };
        self.place = MmRect::new(m[0] + (aw - w) * 0.5, m[1] + (ah - h) * 0.5, w, h);
        self.fit_kind = Some(kind);
    }

    fn set_source(&mut self, img: SourceImage, name: String) {
        self.src_aspect = img.w as f32 / img.h as f32;
        self.status = format!("{}: {} x {} px", name, img.w, img.h);
        self.source = Some(Arc::new(img));
        self.source_name = name;
        self.source_rev += 1;
        self.orig_tex = None;
        self.reset_slots();
        self.fit_image(FitKind::Page);
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
        if let Some(k) = self.fit_kind {
            self.fit_image(k);
        }
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
                Msg::GpuInk { rev, src_rev, ink } => {
                    self.gpu_building = None;
                    self.gpu_ink_rev = rev;
                    match ink {
                        Some(i) => {
                            self.gpu_ink = Some(i);
                            self.gpu_ink_src = src_rev;
                        }
                        None => self.gpu_note = "Could not prepare the ink maps for the GPU".to_string(),
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

    fn use_gpu(&self) -> bool {
        self.gpu_avail && self.gpu_enabled
    }

    /// Convert the current ink maps to half-float mip chains for the GPU, off the UI thread.
    fn maybe_build_gpu_ink(&mut self) {
        if !self.use_gpu() || self.gpu_building.is_some() {
            return;
        }
        let Some(maps) = self.maps.clone() else { return };
        if self.gpu_ink.is_some() && self.gpu_ink_rev == self.maps_rev {
            return;
        }
        let rev = self.maps_rev;
        let src_rev = self.maps_key.map(|k| k.source_rev).unwrap_or(0);
        self.gpu_building = Some(rev);
        let tx = self.tx.clone();
        let ctx = self.ctx.clone();
        thread::spawn(move || {
            let ink = GpuInk::from_maps(&maps).map(Arc::new);
            let _ = tx.send(Msg::GpuInk { rev, src_rev, ink });
            ctx.request_repaint();
        });
    }

    fn refresh_spots(&mut self, scene: &Scene) {
        let key: Vec<(DotShape, u32)> = scene
            .plates
            .iter()
            .map(|p| (p.params.shape, if p.params.shape == DotShape::Ellipse { p.params.elong.to_bits() } else { 0 }))
            .collect();
        if key == self.gpu_spot_key {
            return;
        }
        let tables: Vec<SpotTable> =
            scene.plates.iter().map(|p| SpotTable::new(p.params.shape, p.params.elong)).collect();
        let (data, params) = gpu::spot_tables(&tables);
        self.gpu_spots = Arc::new(data);
        self.gpu_spot_params = params;
        self.gpu_spot_key = key;
        self.gpu_spots_ver += 1;
    }

    /// Queue one GPU draw per visible page. Returns false while the ink maps are not ready.
    fn paint_gpu(
        &mut self,
        painter: &egui::Painter,
        pages: &[egui::Rect],
        offs: &[(f32, f32)],
        canvas: egui::Rect,
        ppp: f32,
        org: Pos2,
    ) -> bool {
        let n = self.plates.len();
        let Some(ink) = self.gpu_ink.clone() else { return false };
        if ink.plates != n || self.gpu_ink_src != self.source_rev {
            return false;
        }
        let z = self.view.zoom;
        let scene = self.scene();
        self.refresh_spots(&scene);
        let pitch = scene.plates.iter().map(|p| 25.4 / p.params.lpi.max(1.0)).fold(f32::MAX, f32::min);
        let px_per_mm = z * ppp;
        let dots = self.always_dots || pitch * px_per_mm >= 3.5;
        self.tone_mode = !dots;
        for (k, page) in pages.iter().enumerate() {
            let vis = page.intersect(canvas);
            if vis.width() < 1.0 || vis.height() < 1.0 {
                continue;
            }
            let wanted: Vec<usize> = match self.view_mode {
                ViewMode::Composite => (0..n).filter(|&i| scene.plates[i].visible).collect(),
                ViewMode::Plate(i) => vec![i],
                ViewMode::AllPlates => vec![k],
                ViewMode::Original => Vec::new(),
            };
            let single = matches!(self.view_mode, ViewMode::Plate(_) | ViewMode::AllPlates);
            let layers: Vec<LayerDesc> = wanted
                .iter()
                .filter(|&&i| i < n)
                .map(|&i| {
                    // the single-plate views show film (black) or the ink colour
                    let col = if !single || self.plate_colors { scene.plates[i].ink } else { [0.04; 3] };
                    let p = &scene.plates[i];
                    let (smin, sinv) = self.gpu_spot_params.get(i).copied().unwrap_or((0.0, 1.0));
                    LayerDesc {
                        plate: i,
                        ink: col,
                        angle_deg: p.params.angle_deg,
                        lpi: p.params.lpi,
                        dot_gain: p.params.dot_gain,
                        min_dot: p.params.min_dot,
                        max_dot: p.params.max_dot,
                        dropout: p.params.dropout,
                        invert: p.params.invert,
                        shape_id: p.params.shape.id(),
                        elong: p.params.elong,
                        spot_min: smin,
                        spot_inv: sinv,
                        patches: if scene.marks.wedge { scene.marks.wedge_patches(scene.image, i) } else { Vec::new() },
                    }
                })
                .collect();
            let (ox, oy) = offs.get(k).copied().unwrap_or((0.0, 0.0));
            let frame = FrameDesc {
                origin_px: [(org.x + ox * z) * ppp, (org.y + oy * z) * ppp],
                px_per_mm,
                image: scene.image,
                page: (scene.page_w, scene.page_h),
                dots,
                aa: self.aa,
                max_lod: (gpu::mip_count(ink.w, ink.h) - 1) as f32,
                base_w: ink.w as f32,
                srgb_target: self.gpu_srgb,
                layers,
            };
            painter.add(gpu::callback_shape(
                vis,
                PageCallback {
                    slot: k,
                    frame: frame.to_floats(),
                    ink: ink.clone(),
                    ink_rev: self.gpu_ink_rev,
                    spots: self.gpu_spots.clone(),
                    spots_ver: self.gpu_spots_ver,
                },
            ));
        }
        true
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

    /// Columns for a fixed page layout (Auto is decided by the view, see `auto_cols`).
    fn fixed_cols(&self, layout: PageLayout) -> usize {
        let n = self.plates.len().max(1);
        match layout {
            PageLayout::Row => n,
            PageLayout::Column => 1,
            _ => (n as f32).sqrt().ceil() as usize,
        }
    }

    /// Column counts that fill every row evenly (1, 2 and 4 for four plates).
    fn allowed_cols(&self) -> Vec<usize> {
        let n = self.plates.len().max(1);
        (1..=n).filter(|c| n % c == 0).collect()
    }

    fn page_gap(&self) -> f32 {
        let (pw, ph) = self.page_mm();
        0.08 * pw.max(ph)
    }

    /// Top-left of each page in world millimetres. One page normally, one per plate in All plates.
    fn offsets_cols(&self, cols: usize) -> Vec<(f32, f32)> {
        if self.view_mode != ViewMode::AllPlates {
            return vec![(0.0, 0.0)];
        }
        let (pw, ph) = self.page_mm();
        let n = self.plates.len().max(1);
        let cols = cols.clamp(1, n);
        let gap = self.page_gap();
        (0..n).map(|k| ((k % cols) as f32 * (pw + gap), (k / cols) as f32 * (ph + gap))).collect()
    }

    fn page_offsets(&self) -> Vec<(f32, f32)> {
        self.offsets_cols(self.cols_res)
    }

    fn world_size_cols(&self, cols: usize) -> (f32, f32) {
        let (pw, ph) = self.page_mm();
        let offs = self.offsets_cols(cols);
        let mx = offs.iter().map(|o| o.0).fold(0.0, f32::max);
        let my = offs.iter().map(|o| o.1).fold(0.0, f32::max);
        (mx + pw, my + ph)
    }

    fn world_size(&self) -> (f32, f32) {
        self.world_size_cols(self.cols_res)
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

    /// Zoom so the world rectangle fills the canvas (`pad` < 1 leaves a border).
    fn zoom_rect(&mut self, canvas: egui::Rect, x: f32, y: f32, w: f32, h: f32, pad: f32) {
        if canvas.width() < 10.0 || canvas.height() < 10.0 {
            self.view.fit_pending = true;
            return;
        }
        let z = (canvas.width() * pad / w.max(0.5)).min(canvas.height() * pad / h.max(0.5));
        self.view.zoom = z.clamp(MIN_ZOOM, MAX_ZOOM);
        self.view.origin = vec2(
            canvas.width() * 0.5 - (x + w * 0.5) * self.view.zoom,
            canvas.height() * 0.5 - (y + h * 0.5) * self.view.zoom,
        );
        self.view.fit_pending = false;
        self.view_fit = false;
        self.zoom_dirty = true;
    }

    fn zoom_to(&mut self, canvas: egui::Rect, x: f32, y: f32, w: f32, h: f32) {
        self.zoom_rect(canvas, x, y, w, h, 0.92);
    }

    fn zoom_about(&mut self, canvas: egui::Rect, c: Pos2, f: f32) {
        let z0 = self.view.zoom;
        let z1 = (z0 * f).clamp(MIN_ZOOM, MAX_ZOOM);
        let p = (c - canvas.min - self.view.origin) / z0;
        self.view.origin = (c - canvas.min) - p * z1;
        self.view.zoom = z1;
        self.view_fit = false;
        self.zoom_dirty = true;
    }

    fn zoom_step(&mut self, f: f32) {
        let c = self.last_canvas;
        self.zoom_about(c, c.center(), f);
    }

    fn fit_width_view(&mut self) {
        let c = self.last_canvas;
        let (ww, _) = self.world_size();
        if c.width() < 10.0 {
            return;
        }
        let z = (c.width() * 0.96 / ww).clamp(MIN_ZOOM, MAX_ZOOM);
        self.view.zoom = z;
        self.view.origin = vec2((c.width() - ww * z) * 0.5, 18.0);
        self.view.fit_pending = false;
        self.view_fit = false;
        self.zoom_dirty = true;
    }

    fn fit_height_view(&mut self) {
        let c = self.last_canvas;
        let (ww, wh) = self.world_size();
        if c.height() < 10.0 {
            return;
        }
        let z = (c.height() * 0.96 / wh).clamp(MIN_ZOOM, MAX_ZOOM);
        self.view.zoom = z;
        self.view.origin = vec2((c.width() - ww * z) * 0.5, (c.height() - wh * z) * 0.5);
        self.view.fit_pending = false;
        self.view_fit = false;
        self.zoom_dirty = true;
    }

    fn actual_size(&mut self) {
        let c = self.last_canvas;
        let z0 = self.view.zoom.max(0.01);
        let center = ((c.width() * 0.5 - self.view.origin.x) / z0, (c.height() * 0.5 - self.view.origin.y) / z0);
        let z = ACTUAL_ZOOM;
        self.view.zoom = z;
        self.view.origin = vec2(c.width() * 0.5 - center.0 * z, c.height() * 0.5 - center.1 * z);
        self.view_fit = false;
        self.zoom_dirty = true;
    }

    fn zoom_image(&mut self) {
        let (c, r) = (self.last_canvas, self.place);
        self.zoom_to(c, r.x, r.y, r.w, r.h);
    }

    /// Returns what a press at screen point `s` grabs, and the top-left of the page it landed on.
    fn hit_test(&self, s: Pos2, org: Pos2, z: f32) -> (DragKind, (f32, f32)) {
        if self.source.is_none() {
            return (DragKind::Pan, (0.0, 0.0));
        }
        let r = self.place;
        let offs = self.page_offsets();
        let (wx, wy) = ((s.x - org.x) / z, (s.y - org.y) / z);
        // Small images get a smaller grab radius so their corners stay separate.
        let grab = HANDLE_GRAB.min(0.34 * r.w.min(r.h) * z).max(7.0);
        let mut best: Option<(f32, usize, (f32, f32))> = None;
        for &(ox, oy) in &offs {
            let c = [
                (r.x + ox, r.y + oy),
                (r.right() + ox, r.y + oy),
                (r.x + ox, r.bottom() + oy),
                (r.right() + ox, r.bottom() + oy),
            ];
            for (i, &(cx, cy)) in c.iter().enumerate() {
                let d = (pos2(org.x + cx * z, org.y + cy * z) - s).length();
                if d < grab && best.map_or(true, |b| d < b.0) {
                    best = Some((d, i, (ox, oy)));
                }
            }
        }
        if let Some((_, i, off)) = best {
            let opposite = [(r.right(), r.bottom()), (r.x, r.bottom()), (r.right(), r.y), (r.x, r.y)][i];
            return (DragKind::Resize { anchor: opposite, aspect: r.w / r.h.max(1e-3), corner: i }, off);
        }
        for &(ox, oy) in &offs {
            let (px, py) = (wx - ox, wy - oy);
            if r.contains(px, py) {
                return (DragKind::Move { off: (px - r.x, py - r.y) }, (ox, oy));
            }
        }
        (DragKind::Pan, (0.0, 0.0))
    }

    /// Pick the page arrangement for "fit page": the one that gives the biggest pages.
    fn resolve_layout(&mut self, canvas: egui::Rect) {
        if self.layout != PageLayout::Auto {
            self.cols_res = self.fixed_cols(self.layout);
            return;
        }
        let mut best = (1usize, 0.0f32);
        for c in self.allowed_cols() {
            let (w, h) = self.world_size_cols(c);
            let score = (canvas.width() / w).min(canvas.height() / h);
            if score > best.1 * 1.02 {
                best = (c, score);
            }
        }
        self.cols_res = best.0;
    }

    /// Automatic layout follows the zoom: as many pages per row as fit the width (1, 2, then all).
    /// The page in the middle of the view stays where it is while the others reflow around it.
    fn apply_dynamic_layout(&mut self, canvas: egui::Rect) {
        if self.layout != PageLayout::Auto || self.view_mode != ViewMode::AllPlates || self.plates.len() < 2 {
            return;
        }
        let (pw, _) = self.page_mm();
        let (z, gap) = (self.view.zoom.max(1e-4), self.page_gap());
        let fit = (((canvas.width() / z + gap) / (pw + gap)).floor() as usize).max(1);
        let new = self.allowed_cols().into_iter().filter(|&c| c <= fit).max().unwrap_or(1);
        if new == self.cols_res {
            return;
        }
        let old_offs = self.page_offsets();
        let c = canvas.center() - canvas.min;
        let wp = ((c.x - self.view.origin.x) / z, (c.y - self.view.origin.y) / z);
        let (pw, ph) = self.page_mm();
        let k = (0..old_offs.len())
            .min_by(|&a, &b| {
                let d = |k: usize| {
                    let (dx, dy) = (wp.0 - (old_offs[k].0 + pw * 0.5), wp.1 - (old_offs[k].1 + ph * 0.5));
                    dx * dx + dy * dy
                };
                d(a).total_cmp(&d(b))
            })
            .unwrap_or(0);
        let local = (wp.0 - old_offs[k].0, wp.1 - old_offs[k].1);
        self.cols_res = new;
        let no = self.page_offsets()[k];
        self.view.origin = vec2(c.x - (no.0 + local.0) * z, c.y - (no.1 + local.1) * z);
    }

    // ------------------------------------------------------------------ menus and toolbar

    fn menu_bar(&mut self, ui: &mut egui::Ui) {
        egui::menu::bar(ui, |ui| {
            ui.menu_button("File", |ui| {
                if ui.add(egui::Button::new("Open image…").shortcut_text("Ctrl+O")).clicked() {
                    ui.close_menu();
                    self.open_dialog();
                }
                let can_export = self.export.is_none() && self.source.is_some();
                if ui
                    .add_enabled(can_export, egui::Button::new("Export separations…").shortcut_text("Ctrl+E"))
                    .clicked()
                {
                    ui.close_menu();
                    self.start_export();
                }
                ui.separator();
                if ui.button("Quit").clicked() {
                    ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
                }
            });
            ui.menu_button("Edit", |ui| {
                let has = self.source.is_some();
                if ui.add_enabled(has, egui::Button::new("Rotate image left")).clicked() {
                    self.rotate(false);
                    ui.close_menu();
                }
                if ui.add_enabled(has, egui::Button::new("Rotate image right")).clicked() {
                    self.rotate(true);
                    ui.close_menu();
                }
                ui.separator();
                ui.add_enabled_ui(has, |ui| {
                    ui.menu_button("Image size", |ui| {
                        for (k, name) in FIT_CHOICES {
                            if ui.button(name).clicked() {
                                self.fit_image(k);
                                ui.close_menu();
                            }
                        }
                        if ui.button("Center on page").clicked() {
                            self.center_image();
                            ui.close_menu();
                        }
                    });
                });
                ui.separator();
                if ui.button("Reset tone adjustments").clicked() {
                    let m = self.tone.mode;
                    self.tone = ToneSettings::default();
                    self.tone.mode = m;
                    ui.close_menu();
                }
                if ui.button("Reset plate settings").clicked() {
                    self.reset_plates();
                    ui.close_menu();
                }
            });
            ui.menu_button("View", |ui| {
                ui.radio_value(&mut self.view_mode, ViewMode::Composite, "Composite (overprint)");
                for i in 0..self.plates.len() {
                    let n = format!("Plate {} (film positive)", self.plates[i].name);
                    ui.radio_value(&mut self.view_mode, ViewMode::Plate(i), n);
                }
                ui.radio_value(&mut self.view_mode, ViewMode::AllPlates, "All plates (separate pages)");
                ui.radio_value(&mut self.view_mode, ViewMode::Original, "Original image");
                ui.separator();
                ui.menu_button("Page layout (All plates)", |ui| {
                    for l in PageLayout::ALL {
                        if ui.radio_value(&mut self.layout, l, l.name()).clicked() {
                            self.view.fit_pending = true;
                            ui.close_menu();
                        }
                    }
                });
                ui.checkbox(&mut self.plate_colors, "Show plates in ink colour");
                ui.checkbox(&mut self.show_marks, "Show printer's marks");
                ui.checkbox(&mut self.show_safe, "Show safe margin");
                ui.checkbox(&mut self.show_rulers, "Show rulers");
                ui.checkbox(&mut self.aa, "Smooth dot edges");
                ui.checkbox(&mut self.always_dots, "Always draw dots when zoomed out");
                ui.separator();
                if ui.add(egui::Button::new("Zoom in").shortcut_text("Ctrl +")).clicked() {
                    self.zoom_step(1.25);
                }
                if ui.add(egui::Button::new("Zoom out").shortcut_text("Ctrl -")).clicked() {
                    self.zoom_step(0.8);
                }
                if ui.add(egui::Button::new("Fit page").shortcut_text("Ctrl+0")).clicked() {
                    self.view.fit_pending = true;
                    ui.close_menu();
                }
                if ui.add(egui::Button::new("Fit width").shortcut_text("Ctrl+2")).clicked() {
                    self.fit_width_view();
                    ui.close_menu();
                }
                if ui.add(egui::Button::new("Fit height").shortcut_text("Ctrl+3")).clicked() {
                    self.fit_height_view();
                    ui.close_menu();
                }
                if ui.button("Zoom to image").clicked() {
                    self.zoom_image();
                    ui.close_menu();
                }
                if ui.add(egui::Button::new("Actual size").shortcut_text("Ctrl+1")).clicked() {
                    self.actual_size();
                    ui.close_menu();
                }
            });
            ui.menu_button("Options", |ui| {
                ui.menu_button("Interface size", |ui| {
                    for s in UI_SCALES {
                        if ui.radio_value(&mut self.ui_scale, s, format!("{:.0}%", s * 100.0)).clicked() {
                            ui.close_menu();
                        }
                    }
                });
                ui.menu_button("Preview renderer", |ui| {
                    ui.add_enabled_ui(self.gpu_avail, |ui| {
                        ui.radio_value(&mut self.gpu_enabled, true, "GPU (realtime)");
                    });
                    ui.radio_value(&mut self.gpu_enabled, false, "CPU (compatible, slower)");
                    if !self.gpu_note.is_empty() {
                        ui.label(&self.gpu_note);
                    }
                });
                ui.menu_button("Units", |ui| {
                    ui.radio_value(&mut self.unit, Unit::Inch, "Inches");
                    ui.radio_value(&mut self.unit, Unit::Mm, "Millimetres");
                });
            });
            ui.menu_button("Help", |ui| {
                if ui.button("About Rusty Halftones").clicked() {
                    self.show_about = true;
                    ui.close_menu();
                }
            });
        });
    }

    fn toolbar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing = vec2(4.0, 4.0);
            if ui.add_sized([86.0, 36.0], egui::Button::new("Open…")).on_hover_text("Open an image (Ctrl+O)").clicked()
            {
                self.open_dialog();
            }
            ui.separator();
            if icon_button(ui, Icon::Select, self.tool == Tool::Select, "Select tool (V): move and resize the image")
                .clicked()
            {
                self.tool = Tool::Select;
            }
            if icon_button(
                ui,
                Icon::Hand,
                self.tool == Tool::Hand,
                "Hand tool (H): drag to scroll. The middle button always scrolls",
            )
            .clicked()
            {
                self.tool = Tool::Hand;
            }
            if icon_button(
                ui,
                Icon::ZoomBox,
                self.tool == Tool::ZoomBox,
                "Zoom tool (Z): drag a box to zoom to it, click to zoom in, Shift+click to zoom out",
            )
            .clicked()
            {
                self.tool = Tool::ZoomBox;
            }
            ui.separator();
            if icon_button(ui, Icon::ZoomIn, false, "Zoom in (Ctrl +)").clicked() {
                self.zoom_step(1.25);
            }
            if icon_button(ui, Icon::ZoomOut, false, "Zoom out (Ctrl -)").clicked() {
                self.zoom_step(0.8);
            }
            if icon_button(ui, Icon::FitPage, false, "Fit page (Ctrl+0)").clicked() {
                self.view.fit_pending = true;
            }
            if icon_button(ui, Icon::FitWidth, false, "Fit width (Ctrl+2)").clicked() {
                self.fit_width_view();
            }
            if icon_button(ui, Icon::FitHeight, false, "Fit height (Ctrl+3)").clicked() {
                self.fit_height_view();
            }
            if icon_button(ui, Icon::ZoomImage, false, "Zoom to the image").clicked() {
                self.zoom_image();
            }
            let old = self.view.zoom.max(0.001) / ACTUAL_ZOOM * 100.0;
            let mut pct = old;
            if ui
                .add(egui::DragValue::new(&mut pct).range(2.0..=8000.0).speed(2.0).fixed_decimals(0).suffix("%"))
                .on_hover_text("Zoom (100% is actual size at 96 dpi)")
                .changed()
            {
                self.zoom_step(pct / old);
            }
            ui.separator();
            let names: Vec<String> = self.plates.iter().map(|p| p.name.clone()).collect();
            let label = match self.view_mode {
                ViewMode::Composite => "Composite".to_string(),
                ViewMode::Plate(i) => format!("Plate {}", names.get(i).cloned().unwrap_or_default()),
                ViewMode::AllPlates => "All plates".to_string(),
                ViewMode::Original => "Original".to_string(),
            };
            egui::ComboBox::from_id_salt("tb_view").selected_text(format!("View: {}", label)).show_ui(ui, |ui| {
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
            ui.toggle_value(&mut self.plate_colors, "Ink colours").on_hover_text(
                "Show plates in their ink colour instead of black film (preview only, export is unchanged)",
            );
            if self.view_mode == ViewMode::AllPlates {
                egui::ComboBox::from_id_salt("tb_layout")
                    .selected_text(format!("Layout: {}", self.layout.name()))
                    .show_ui(ui, |ui| {
                        for l in PageLayout::ALL {
                            if ui.selectable_value(&mut self.layout, l, l.name()).clicked() {
                                self.view.fit_pending = true;
                            }
                        }
                    });
            }
        });
    }

    fn center_image(&mut self) {
        let (pw, ph) = self.page_mm();
        self.place.x = (pw - self.place.w) * 0.5;
        self.place.y = (ph - self.place.h) * 0.5;
        self.fit_kind = None;
    }

    // ------------------------------------------------------------------ settings panel

    fn controls(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.spacing_mut().button_padding = vec2(10.0, 6.0);
            for (t, n) in [
                (Tab::Layout, "Layout"),
                (Tab::Separation, "Separation"),
                (Tab::Plates, "Plates"),
                (Tab::Export, "Export"),
            ] {
                ui.selectable_value(&mut self.tab, t, n);
            }
        });
        ui.separator();
        egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| match self.tab {
            Tab::Layout => self.tab_layout(ui),
            Tab::Separation => self.tab_separation(ui),
            Tab::Plates => self.tab_plates(ui),
            Tab::Export => self.tab_export(ui),
        });
    }

    fn tab_layout(&mut self, ui: &mut egui::Ui) {
        let unit = self.unit;
        section(ui, "Page");
        ui.horizontal(|ui| {
            ui.label("Units");
            ui.selectable_value(&mut self.unit, Unit::Inch, "in");
            ui.selectable_value(&mut self.unit, Unit::Mm, "mm");
            ui.checkbox(&mut self.landscape, "Landscape");
        });
        let cur = if self.page_idx < PAGE_SIZES.len() { PAGE_SIZES[self.page_idx].name } else { "Custom" };
        egui::ComboBox::from_id_salt("page_size").selected_text(cur).width(240.0).show_ui(ui, |ui| {
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
        ui.horizontal(|ui| {
            ui.label("Safe margin");
            dv_mm(ui, &mut self.safe_margin, unit, 0.0, 50.0);
        })
        .response
        .on_hover_text("Keep everything this far from the paper edge so it is sure to print");

        section(ui, "Image");
        ui.horizontal(|ui| {
            if ui.button("Open…").clicked() {
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
        let Some(src) = self.source.clone() else {
            return self.tab_layout_marks(ui);
        };
        let mut edited = false;
        egui::Grid::new("place_grid").num_columns(2).spacing([12.0, 6.0]).show(ui, |ui| {
            ui.label("X");
            edited |= dv_mm(ui, &mut self.place.x, unit, -5000.0, 5000.0);
            ui.end_row();
            ui.label("Y");
            edited |= dv_mm(ui, &mut self.place.y, unit, -5000.0, 5000.0);
            ui.end_row();
            ui.label("Width");
            if dv_mm(ui, &mut self.place.w, unit, 5.0, 5000.0) {
                edited = true;
                if self.lock_aspect {
                    self.place.h = self.place.w / self.src_aspect;
                }
            }
            ui.end_row();
            ui.label("Height");
            if dv_mm(ui, &mut self.place.h, unit, 5.0, 5000.0) {
                edited = true;
                if self.lock_aspect {
                    self.place.w = self.place.h * self.src_aspect;
                }
            }
            ui.end_row();
            ui.label("Resolution");
            let mut ppi = src.w as f32 / (self.place.w / 25.4).max(1e-3);
            if ui.add(egui::DragValue::new(&mut ppi).speed(1.0).range(10.0..=2400.0).suffix(" ppi")).changed() {
                edited = true;
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
                edited = true;
            }
        });
        if edited {
            self.fit_kind = None;
        }

        section(ui, "Fit image to page");
        ui.horizontal_wrapped(|ui| {
            for (k, name) in FIT_CHOICES {
                let on = self.fit_kind == Some(k);
                if ui.selectable_label(on, name).on_hover_text(k.help()).clicked() {
                    self.fit_image(k);
                }
            }
            if ui.button("Center").clicked() {
                self.center_image();
            }
        });
        ui.label(match self.fit_kind {
            Some(_) => "Stays fitted when the page, marks or safe margin change.",
            None => "Placed by hand. Drag the image to move it, drag a corner to resize.",
        });
        self.tab_layout_marks(ui);
    }

    fn tab_layout_marks(&mut self, ui: &mut egui::Ui) {
        let unit = self.unit;
        section(ui, "Printer's marks");
        ui.checkbox(&mut self.marks.crop, "Crop marks");
        ui.checkbox(&mut self.marks.reg, "Registration marks");
        ui.checkbox(&mut self.marks.label, "Plate label (angle, lpi, shape)");
        ui.checkbox(&mut self.marks.wedge, "Tone-step wedge per plate");
        egui::Grid::new("marks_grid").num_columns(4).spacing([12.0, 6.0]).show(ui, |ui| {
            ui.label("Offset");
            dv_mm(ui, &mut self.marks.offset, unit, 0.0, 30.0);
            ui.label("Crop length");
            dv_mm(ui, &mut self.marks.crop_len, unit, 1.0, 30.0);
            ui.end_row();
            ui.label("Target");
            dv_mm(ui, &mut self.marks.reg_size, unit, 2.0, 20.0);
            ui.label("Wedge row");
            dv_mm(ui, &mut self.marks.wedge_h, unit, 2.0, 12.0);
            ui.end_row();
        });
        ui.checkbox(&mut self.show_marks, "Show marks in preview");
        ui.checkbox(&mut self.show_safe, "Show safe margin in preview");
        if self.source.is_some() {
            let (pw, ph) = self.page_mm();
            let m = self.marks.margins(self.plates.len());
            let (r, s) = (self.place, self.safe_margin);
            let (l, t, rt, b) = (r.x - m[0], r.y - m[1], r.right() + m[2], r.bottom() + m[3]);
            if l < -0.01 || t < -0.01 || rt > pw + 0.01 || b > ph + 0.01 {
                ui.colored_label(
                    Color32::from_rgb(235, 80, 60),
                    "The image or its marks run off the page and will be cut off. Use Fit to page.",
                );
            } else if l < s - 0.01 || t < s - 0.01 || rt > pw - s + 0.01 || b > ph - s + 0.01 {
                ui.colored_label(
                    Color32::from_rgb(235, 160, 30),
                    "The marks reach into the safe margin and might not print. Use Fit to page.",
                );
            } else {
                ui.colored_label(Color32::from_rgb(90, 190, 110), "Image and marks are inside the safe margin.");
            }
        }
    }

    fn tab_separation(&mut self, ui: &mut egui::Ui) {
        section(ui, "Separation");
        let mut mode = self.tone.mode;
        egui::ComboBox::from_id_salt("sep_mode")
            .selected_text(if mode == SepMode::Mono { "Single plate (greyscale)" } else { "CMYK (4 plates)" })
            .width(240.0)
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
            .width(240.0)
            .show_ui(ui, |ui| {
                for r in WORK_RES_CHOICES {
                    ui.selectable_value(&mut self.work_res, r, format!("{} px (long side)", r));
                }
            });
        ui.label("Plain RGB to CMYK, no ICC profiles.");

        section(ui, "Screen (halftone)");
        ui.add(egui::Slider::new(&mut self.lpi, 10.0..=150.0).text("Lines per inch"));
        ui.horizontal(|ui| {
            ui.label("Mesh");
            ui.add(egui::DragValue::new(&mut self.mesh).speed(1.0).range(40.0..=500.0).suffix(" tpi"));
            if ui.small_button("Use").clicked() {
                self.lpi = (self.mesh / 4.5).round();
            }
        });
        ui.label(format!("A {:.0} mesh suggests {:.0}-{:.0} lpi.", self.mesh, self.mesh / 5.0, self.mesh / 4.0));
        ui.horizontal(|ui| {
            ui.label("Dot shape");
            egui::ComboBox::from_id_salt("shape").selected_text(self.shape.name()).show_ui(ui, |ui| {
                for s in DotShape::ALL {
                    ui.selectable_value(&mut self.shape, s, s.name());
                }
            });
        });
        if self.shape == DotShape::Ellipse {
            ui.add(egui::Slider::new(&mut self.elong, 1.0..=2.5).text("Elongation"));
        }
    }

    fn tab_plates(&mut self, ui: &mut egui::Ui) {
        if self.plates.is_empty() {
            return;
        }
        self.sel_plate = self.sel_plate.min(self.plates.len() - 1);
        section(ui, "All plates");
        egui::Grid::new("plates_grid").num_columns(4).spacing([12.0, 6.0]).show(ui, |ui| {
            ui.label("");
            ui.label("Ink");
            ui.label("Angle");
            ui.label("Show");
            ui.end_row();
            for (i, p) in self.plates.iter_mut().enumerate() {
                if ui.selectable_label(self.sel_plate == i, format!("  {}  ", p.name)).clicked() {
                    self.sel_plate = i;
                }
                ui.color_edit_button_srgb(&mut p.ink);
                ui.add(egui::DragValue::new(&mut p.angle).speed(0.5).range(0.0..=180.0).suffix("°"));
                ui.checkbox(&mut p.visible, "");
                ui.end_row();
            }
        });
        if self.tone.mode == SepMode::Cmyk && self.plates.len() == 4 {
            let mut chosen = None;
            egui::ComboBox::from_id_salt("angle_preset").selected_text("Angle preset…").width(240.0).show_ui(
                ui,
                |ui| {
                    for (name, a) in ANGLE_PRESETS {
                        if ui.selectable_label(false, name).clicked() {
                            chosen = Some(a);
                        }
                    }
                },
            );
            if let Some(a) = chosen {
                for (p, v) in self.plates.iter_mut().zip(a) {
                    p.angle = v;
                }
            }
        }

        let i = self.sel_plate;
        section(ui, &format!("Plate {}", self.plates[i].name));
        let (g_lpi, g_shape) = (self.lpi, self.shape);
        let mut copy_all = false;
        let mut preview = false;
        {
            let p = &mut self.plates[i];
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
            ui.horizontal(|ui| {
                if ui.button("Preview this plate").clicked() {
                    preview = true;
                }
                if ui.button("Copy dot settings to all plates").clicked() {
                    copy_all = true;
                }
            });
        }
        if preview {
            self.view_mode = ViewMode::Plate(i);
        }
        if copy_all {
            let (g, mn, mx, d, inv) = {
                let p = &self.plates[i];
                (p.dot_gain, p.min_dot, p.max_dot, p.dropout, p.invert)
            };
            for q in &mut self.plates {
                q.dot_gain = g;
                q.min_dot = mn;
                q.max_dot = mx;
                q.dropout = d;
                q.invert = inv;
            }
        }
    }

    fn tab_export(&mut self, ui: &mut egui::Ui) {
        section(ui, "Export film");
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
        if self.export.is_none() && ui.add_enabled(can_export, egui::Button::new("Export separations…")).clicked() {
            self.start_export();
        }
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
            if self.maps_building.is_some()
                || self.gpu_building.is_some()
                || self.any_inflight()
                || self.export.is_some()
            {
                ui.add(egui::Spinner::new());
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let r = ui.label(if self.use_gpu() { "GPU preview" } else { "CPU preview" });
                if !self.gpu_note.is_empty() {
                    r.on_hover_text(&self.gpu_note);
                }
            });
        });
    }

    // ------------------------------------------------------------------ canvas

    /// Rulers in the current unit. With several pages each column and row restarts at zero.
    fn draw_rulers(
        &self,
        p: &egui::Painter,
        full: egui::Rect,
        rect: egui::Rect,
        org: Pos2,
        offs: &[(f32, f32)],
        hover: Option<Pos2>,
    ) {
        let (pw, ph) = self.page_mm();
        let z = self.view.zoom;
        let mm_per_unit = if self.unit == Unit::Inch { 25.4 } else { 1.0 };
        let px_per_unit = z * mm_per_unit;
        let steps = [0.01f32, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0];
        let major = steps.iter().copied().find(|s| s * px_per_unit >= 70.0).unwrap_or(1000.0);
        let minor = major / 5.0;
        let show_minor = minor * px_per_unit >= 5.0;
        let bg = Color32::from_gray(38);
        let fg = Color32::from_gray(190);
        let tick = Stroke::new(1.0, Color32::from_gray(150));
        let font = egui::FontId::proportional(10.5);
        let top = egui::Rect::from_min_max(pos2(rect.left(), full.top()), pos2(rect.right(), rect.top()));
        let left = egui::Rect::from_min_max(pos2(full.left(), rect.top()), pos2(rect.left(), rect.bottom()));
        let corner = egui::Rect::from_min_max(full.min, rect.min);
        p.rect_filled(top, 0.0, bg);
        p.rect_filled(left, 0.0, bg);
        p.rect_filled(corner, 0.0, Color32::from_gray(30));
        let label = |v: f32| format!("{}", (v * 1000.0).round() / 1000.0);

        let mut xs: Vec<f32> = offs.iter().map(|o| o.0).collect();
        xs.sort_by(|a, b| a.total_cmp(b));
        xs.dedup_by(|a, b| (*a - *b).abs() < 1e-3);
        let mut ys: Vec<f32> = offs.iter().map(|o| o.1).collect();
        ys.sort_by(|a, b| a.total_cmp(b));
        ys.dedup_by(|a, b| (*a - *b).abs() < 1e-3);

        let pt = p.with_clip_rect(top);
        for &ox in &xs {
            let (lo, hi) = (((rect.left() - org.x) / z - ox).max(0.0), ((rect.right() - org.x) / z - ox).min(pw));
            if hi <= lo {
                continue;
            }
            let step = if show_minor { minor } else { major };
            let k0 = (lo / mm_per_unit / step).ceil() as i64;
            let k1 = (hi / mm_per_unit / step).floor() as i64;
            for k in k0..=k1.min(k0 + 4000) {
                let v = k as f32 * step;
                let sx = org.x + (ox + v * mm_per_unit) * z;
                let is_major = (v / major - (v / major).round()).abs() < 1e-3;
                let len = if is_major { RULER_SIZE - 3.0 } else { 6.0 };
                pt.line_segment([pos2(sx, top.bottom() - len), pos2(sx, top.bottom())], tick);
                if is_major {
                    pt.text(pos2(sx + 2.0, top.top() + 1.0), egui::Align2::LEFT_TOP, label(v), font.clone(), fg);
                }
            }
        }
        let pl = p.with_clip_rect(left);
        for &oy in &ys {
            let (lo, hi) = (((rect.top() - org.y) / z - oy).max(0.0), ((rect.bottom() - org.y) / z - oy).min(ph));
            if hi <= lo {
                continue;
            }
            let step = if show_minor { minor } else { major };
            let k0 = (lo / mm_per_unit / step).ceil() as i64;
            let k1 = (hi / mm_per_unit / step).floor() as i64;
            for k in k0..=k1.min(k0 + 4000) {
                let v = k as f32 * step;
                let sy = org.y + (oy + v * mm_per_unit) * z;
                let is_major = (v / major - (v / major).round()).abs() < 1e-3;
                let len = if is_major { RULER_SIZE - 3.0 } else { 6.0 };
                pl.line_segment([pos2(left.right() - len, sy), pos2(left.right(), sy)], tick);
                if is_major {
                    let g = pl.layout_no_wrap(label(v), font.clone(), fg);
                    let at = pos2(left.left() + 2.0, sy + 2.0 + g.size().x);
                    pl.add(egui::epaint::TextShape::new(at, g, fg).with_angle(-std::f32::consts::FRAC_PI_2));
                }
            }
        }
        if let Some(h) = hover {
            let mark = Stroke::new(1.5, Color32::from_rgb(255, 170, 40));
            if rect.contains(h) {
                pt.line_segment([pos2(h.x, top.top()), pos2(h.x, top.bottom())], mark);
                pl.line_segment([pos2(left.left(), h.y), pos2(left.right(), h.y)], mark);
            }
        }
    }

    /// Scroll bars over the world bounds, with half a window of slack on every side.
    fn scrollbars(&mut self, ui: &mut egui::Ui, p: &egui::Painter, full: egui::Rect, rect: egui::Rect) {
        let (ww, wh) = self.world_size();
        let z = self.view.zoom;
        let track_col = Color32::from_gray(46);
        let thumb_col = |r: &egui::Response| {
            if r.dragged() {
                Color32::from_gray(170)
            } else if r.hovered() {
                Color32::from_gray(145)
            } else {
                Color32::from_gray(110)
            }
        };
        p.rect_filled(egui::Rect::from_min_max(pos2(full.left(), rect.bottom()), full.max), 0.0, track_col);
        p.rect_filled(egui::Rect::from_min_max(pos2(rect.right(), full.top()), full.max), 0.0, track_col);

        // horizontal
        {
            let track = egui::Rect::from_min_max(pos2(rect.left(), rect.bottom()), pos2(rect.right(), full.bottom()));
            let vw = rect.width();
            let total = ww * z + vw;
            let offset = -self.view.origin.x + vw * 0.5;
            let len = (vw / total * track.width()).clamp(28.0, track.width());
            let x = track.left() + (offset / total * track.width()).clamp(0.0, track.width() - len);
            let thumb = egui::Rect::from_min_size(pos2(x, track.top() + 2.0), vec2(len, track.height() - 4.0));
            let r = ui.interact(thumb, ui.id().with("hbar"), egui::Sense::drag());
            if r.dragged() {
                self.view.origin.x -= r.drag_delta().x * total / track.width().max(1.0);
                self.view_fit = false;
            }
            p.rect_filled(thumb, 4.0, thumb_col(&r));
        }
        // vertical
        {
            let track = egui::Rect::from_min_max(pos2(rect.right(), rect.top()), pos2(full.right(), rect.bottom()));
            let vh = rect.height();
            let total = wh * z + vh;
            let offset = -self.view.origin.y + vh * 0.5;
            let len = (vh / total * track.height()).clamp(28.0, track.height());
            let y = track.top() + (offset / total * track.height()).clamp(0.0, track.height() - len);
            let thumb = egui::Rect::from_min_size(pos2(track.left() + 2.0, y), vec2(track.width() - 4.0, len));
            let r = ui.interact(thumb, ui.id().with("vbar"), egui::Sense::drag());
            if r.dragged() {
                self.view.origin.y -= r.drag_delta().y * total / track.height().max(1.0);
                self.view_fit = false;
            }
            p.rect_filled(thumb, 4.0, thumb_col(&r));
        }
    }

    fn canvas(&mut self, ui: &mut egui::Ui) {
        let size = ui.available_size();
        let (resp, full_painter) = ui.allocate_painter(size, egui::Sense::click_and_drag());
        let full = resp.rect;
        // The page area leaves room for the rulers (top, left) and the scroll bars (bottom, right).
        let ruler = if self.show_rulers { RULER_SIZE } else { 0.0 };
        let rect = egui::Rect::from_min_max(full.min + vec2(ruler, ruler), full.max - vec2(SCROLL_SIZE, SCROLL_SIZE));
        let painter = full_painter.with_clip_rect(rect);
        self.last_canvas = rect;
        let ppp = ui.ctx().pixels_per_point();
        let (pw, ph) = self.page_mm();
        if self.prev_view_mode != self.view_mode {
            if self.prev_view_mode == ViewMode::AllPlates || self.view_mode == ViewMode::AllPlates {
                self.view.fit_pending = true;
            }
            self.prev_view_mode = self.view_mode;
        }
        // When the view is "fit page" it follows the window, and Auto layout picks the best arrangement.
        if self.view_fit && (rect.size() - self.last_canvas_size).abs().max_elem() > 2.0 {
            self.view.fit_pending = true;
        }
        self.last_canvas_size = rect.size();
        if self.layout != PageLayout::Auto {
            self.cols_res = self.fixed_cols(self.layout);
        }
        if self.use_gpu() != self.gpu_prev {
            self.gpu_prev = self.use_gpu();
            self.reset_slots();
        }
        if self.view.fit_pending || self.view.zoom <= 0.0 {
            self.resolve_layout(rect);
            let (ww, wh) = self.world_size();
            self.zoom_to(rect, 0.0, 0.0, ww, wh);
            self.view_fit = true;
            self.zoom_dirty = false;
        }

        // wheel scrolls, Ctrl+wheel (or a pinch) zooms
        let in_page_area = resp.hover_pos().map_or(false, |p| rect.contains(p));
        if in_page_area {
            let (scroll, pinch) = ui.input(|i| (i.smooth_scroll_delta, i.zoom_delta()));
            if (pinch - 1.0).abs() > 1e-4 {
                if let Some(c) = resp.hover_pos() {
                    self.zoom_about(rect, c, pinch);
                }
            } else if scroll != Vec2::ZERO {
                self.view.origin += scroll;
                self.view_fit = false;
            }
        }

        // pointer: handled on press (not after the drag threshold) so handles are easy to grab
        let (pressed, down, released, latest, delta, shift) = ui.input(|i| {
            (
                i.pointer.primary_pressed(),
                i.pointer.primary_down(),
                i.pointer.primary_released(),
                i.pointer.latest_pos(),
                i.pointer.delta(),
                i.modifiers.shift || i.modifiers.alt,
            )
        });
        {
            let z = self.view.zoom;
            let org = rect.min + self.view.origin;
            let to_w = |s: Pos2| ((s.x - org.x) / z, (s.y - org.y) / z);
            if pressed && resp.hovered() && latest.map_or(false, |s| rect.contains(s)) {
                if let Some(s) = latest {
                    match self.tool {
                        Tool::Select => {
                            let (kind, off) = self.hit_test(s, org, z);
                            self.drag = Some(kind);
                            self.drag_off = off;
                        }
                        Tool::Hand => self.drag = Some(DragKind::Pan),
                        Tool::ZoomBox => {
                            self.zoom_start = Some(s);
                            self.zoom_cur = Some(s);
                        }
                    }
                }
            }
            if down {
                if let (Some(kind), Some(s)) = (self.drag, latest) {
                    let (wx, wy) = to_w(s);
                    let (px, py) = (wx - self.drag_off.0, wy - self.drag_off.1);
                    match kind {
                        DragKind::Pan => {
                            self.view.origin += delta;
                            if delta != Vec2::ZERO {
                                self.view_fit = false;
                            }
                        }
                        DragKind::Move { off } => {
                            let (nx, ny) = (px - off.0, py - off.1);
                            if (nx - self.place.x).abs() + (ny - self.place.y).abs() > 1e-4 {
                                self.fit_kind = None;
                            }
                            self.place.x = nx;
                            self.place.y = ny;
                        }
                        DragKind::Resize { anchor, aspect, .. } => {
                            let (ax, ay) = anchor;
                            let w = (px - ax).abs().max(5.0);
                            let h = if self.lock_aspect { w / aspect } else { (py - ay).abs().max(5.0) };
                            let nr =
                                MmRect::new(if px < ax { ax - w } else { ax }, if py < ay { ay - h } else { ay }, w, h);
                            if (nr.w - self.place.w).abs() > 1e-4 || (nr.h - self.place.h).abs() > 1e-4 {
                                self.fit_kind = None;
                            }
                            self.place = nr;
                        }
                    }
                }
                if self.zoom_start.is_some() {
                    self.zoom_cur = latest;
                }
            }
            if released || (!down && (self.drag.is_some() || self.zoom_start.is_some())) {
                self.drag = None;
                if let (Some(a), Some(b)) = (self.zoom_start.take(), self.zoom_cur.take()) {
                    let r = egui::Rect::from_two_pos(a, b);
                    if r.width() < 6.0 && r.height() < 6.0 {
                        self.zoom_about(rect, a, if shift { 0.5 } else { 2.0 });
                    } else {
                        let (p0, p1) = (to_w(r.min), to_w(r.max));
                        self.zoom_rect(rect, p0.0, p0.1, p1.0 - p0.0, p1.1 - p0.1, 1.0);
                    }
                }
            }
            if resp.dragged_by(egui::PointerButton::Middle) || resp.dragged_by(egui::PointerButton::Secondary) {
                self.view.origin += resp.drag_delta();
                self.view_fit = false;
            }
        }
        if self.zoom_dirty {
            self.zoom_dirty = false;
            self.apply_dynamic_layout(rect);
        }
        let offs = self.page_offsets();

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

        // mouse cursor + handle under the pointer
        let mut hot_corner: Option<usize> = None;
        if in_page_area || self.drag.is_some() {
            let kind = match self.drag {
                Some(k) => Some(k),
                None if self.tool == Tool::Select => resp.hover_pos().map(|p| self.hit_test(p, org, z).0),
                None => None,
            };
            let icon = match (self.tool, kind) {
                (Tool::ZoomBox, _) => {
                    if shift {
                        egui::CursorIcon::ZoomOut
                    } else {
                        egui::CursorIcon::ZoomIn
                    }
                }
                (Tool::Hand, _) => {
                    if self.drag.is_some() {
                        egui::CursorIcon::Grabbing
                    } else {
                        egui::CursorIcon::Grab
                    }
                }
                (_, Some(DragKind::Resize { corner, .. })) => {
                    hot_corner = Some(corner);
                    if corner == 0 || corner == 3 {
                        egui::CursorIcon::ResizeNwSe
                    } else {
                        egui::CursorIcon::ResizeNeSw
                    }
                }
                (_, Some(DragKind::Move { .. })) => egui::CursorIcon::Move,
                (_, Some(DragKind::Pan)) if self.drag.is_some() => egui::CursorIcon::Grabbing,
                _ => egui::CursorIcon::Default,
            };
            ui.ctx().set_cursor_icon(icon);
        }

        // pages
        full_painter.rect_filled(full, 0.0, Color32::from_gray(58));
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
        } else if self.use_gpu() {
            let ready = self.paint_gpu(&painter, &pages, &offs, rect, ppp, org);
            if !ready {
                painter.text(
                    pages[0].center(),
                    egui::Align2::CENTER_CENTER,
                    "Processing image…",
                    egui::FontId::proportional(18.0),
                    Color32::from_gray(120),
                );
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

        // safe margin outline
        if self.show_safe && self.safe_margin > 0.0 {
            let s = self.safe_margin;
            let col = Color32::from_rgba_unmultiplied(70, 140, 255, 190);
            for &(ox, oy) in &offs {
                let pts = [
                    to_s(ox + s, oy + s),
                    to_s(ox + pw - s, oy + s),
                    to_s(ox + pw - s, oy + ph - s),
                    to_s(ox + s, oy + ph - s),
                    to_s(ox + s, oy + s),
                ];
                painter.extend(egui::Shape::dashed_line(&pts, Stroke::new(1.0, col), 6.0, 4.0));
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
                        Prim::Text { x, y, text, .. } => {
                            // Always the same readable size on screen, whatever the zoom.
                            let g = painter.layout_no_wrap(text, egui::FontId::proportional(13.0), Color32::BLACK);
                            if g.size().x <= pages[k].width() {
                                let at = to_s(ox + x, oy + y) - vec2(0.0, g.size().y);
                                let bg = egui::Rect::from_min_size(at, g.size()).expand2(vec2(3.0, 1.0));
                                painter.rect_filled(bg, 2.0, Color32::from_white_alpha(225));
                                painter.galley(at, g, Color32::BLACK);
                            }
                        }
                    }
                }
            }
        }

        // plate captions: constant size, on a chip that contrasts with both the ink colour and the background
        if self.source.is_some() && self.view_mode != ViewMode::Original {
            let scene = self.scene();
            for (k, page) in pages.iter().enumerate() {
                if page.right() < rect.left()
                    || page.left() > rect.right()
                    || page.bottom() < rect.top()
                    || page.top() > rect.bottom()
                {
                    continue;
                }
                let plate = match self.view_mode {
                    ViewMode::AllPlates => Some(k),
                    ViewMode::Plate(i) => Some(i),
                    _ => None,
                };
                let (text, bg, fg) = match plate.and_then(|i| self.plates.get(i)) {
                    Some(p) => {
                        let pp = &scene.plates[plate.unwrap_or(0)].params;
                        let text =
                            format!("Plate {}   {:.1}°   {:.0} lpi   {}", p.name, p.angle, pp.lpi, pp.shape.name());
                        if self.plate_colors {
                            let ink = Color32::from_rgb(p.ink[0], p.ink[1], p.ink[2]);
                            let lum = 0.2126 * p.ink[0] as f32 + 0.7152 * p.ink[1] as f32 + 0.0722 * p.ink[2] as f32;
                            (text, ink, if lum > 140.0 { Color32::BLACK } else { Color32::WHITE })
                        } else {
                            (text, Color32::from_gray(232), Color32::from_gray(20))
                        }
                    }
                    None => ("Composite (overprint)".to_string(), Color32::from_gray(232), Color32::from_gray(20)),
                };
                let g = painter.layout_no_wrap(text, egui::FontId::proportional(15.0), fg);
                let pad = vec2(9.0, 4.0);
                let size = g.size() + pad * 2.0;
                let x = page.left().max(rect.left() + 6.0).min((rect.right() - size.x - 6.0).max(rect.left() + 6.0));
                let y = (page.top() - 6.0 - size.y).max(rect.top() + 6.0);
                let chip = egui::Rect::from_min_size(pos2(x, y), size);
                painter.rect_filled(chip, 5.0, bg);
                painter.rect_stroke(
                    chip,
                    5.0,
                    Stroke::new(1.0, Color32::from_white_alpha(110)),
                    egui::StrokeKind::Outside,
                );
                painter.galley(chip.min + pad, g, fg);
            }
        }

        // selection outline + handles on every page (they all edit the same placement)
        if self.source.is_some() {
            let blue = Color32::from_rgb(0, 120, 255);
            for &(ox, oy) in &offs {
                let ir = img_rect(ox, oy);
                painter.rect_stroke(ir, 0.0, Stroke::new(1.5, blue), egui::StrokeKind::Outside);
                if self.tool == Tool::Select {
                    for (i, c) in
                        [ir.left_top(), ir.right_top(), ir.left_bottom(), ir.right_bottom()].into_iter().enumerate()
                    {
                        let hot = hot_corner == Some(i);
                        let h = egui::Rect::from_center_size(
                            c,
                            vec2(if hot { 16.0 } else { 13.0 }, if hot { 16.0 } else { 13.0 }),
                        );
                        painter.rect_filled(h, 2.0, if hot { blue } else { Color32::WHITE });
                        painter.rect_stroke(h, 2.0, Stroke::new(2.0, blue), egui::StrokeKind::Inside);
                    }
                }
            }
        }
        for page in &pages {
            painter.rect_stroke(*page, 0.0, Stroke::new(1.0, Color32::from_gray(30)), egui::StrokeKind::Outside);
        }

        // rubber band of the zoom tool
        if let (Some(a), Some(b)) = (self.zoom_start, self.zoom_cur) {
            let r = egui::Rect::from_two_pos(a, b);
            let blue = Color32::from_rgb(0, 120, 255);
            painter.rect_filled(r, 0.0, Color32::from_rgba_unmultiplied(0, 120, 255, 40));
            painter.rect_stroke(r, 0.0, Stroke::new(1.5, blue), egui::StrokeKind::Inside);
        }

        if self.show_rulers {
            self.draw_rulers(&full_painter, full, rect, org, &offs, resp.hover_pos());
        }
        self.scrollbars(ui, &full_painter, full, rect);
    }
}

fn section(ui: &mut egui::Ui, title: &str) {
    ui.add_space(10.0);
    ui.label(egui::RichText::new(title).strong().size(16.0));
    ui.separator();
}

fn settings_path() -> Option<PathBuf> {
    let base = std::env::var_os("APPDATA")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))?;
    Some(base.join("RustyHalftones").join("settings.txt"))
}

impl eframe::App for HalftoneApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if (ctx.zoom_factor() - self.ui_scale).abs() > 1e-3 {
            ctx.set_zoom_factor(self.ui_scale);
        }
        self.poll(ctx);

        let dropped = ctx.input(|i| i.raw.dropped_files.clone());
        for f in dropped {
            if let Some(p) = f.path {
                self.load_path(p);
                break;
            }
        }
        let cmd = |k: egui::Key| ctx.input(|i| i.modifiers.command && i.key_pressed(k));
        if cmd(egui::Key::O) {
            self.open_dialog();
        }
        if cmd(egui::Key::E) && self.source.is_some() && self.export.is_none() {
            self.start_export();
        }
        if cmd(egui::Key::Num0) {
            self.view.fit_pending = true;
        }
        if cmd(egui::Key::Num1) {
            self.actual_size();
        }
        if cmd(egui::Key::Num2) {
            self.fit_width_view();
        }
        if cmd(egui::Key::Num3) {
            self.fit_height_view();
        }
        if cmd(egui::Key::Equals) || cmd(egui::Key::Plus) {
            self.zoom_step(1.25);
        }
        if cmd(egui::Key::Minus) {
            self.zoom_step(0.8);
        }
        if !ctx.wants_keyboard_input() {
            if ctx.input(|i| i.key_pressed(egui::Key::V)) {
                self.tool = Tool::Select;
            }
            if ctx.input(|i| i.key_pressed(egui::Key::Z)) {
                self.tool = Tool::ZoomBox;
            }
            if ctx.input(|i| i.key_pressed(egui::Key::H)) {
                self.tool = Tool::Hand;
            }
        }
        if let ViewMode::Plate(i) = self.view_mode {
            if i >= self.plates.len() {
                self.view_mode = ViewMode::Composite;
            }
        }

        // keep a fitted image fitted when the page, marks or safe margin change
        let key = self.fit_key();
        let last = self.fit_key_last;
        if key != last {
            self.fit_key_last = key;
            if let Some(k) = self.fit_kind {
                self.fit_image(k);
            }
            if key[0] != last[0] || key[1] != last[1] {
                self.view.fit_pending = true;
            }
        }

        self.maybe_build_maps();
        self.maybe_build_gpu_ink();

        egui::TopBottomPanel::top("menu").show(ctx, |ui| self.menu_bar(ui));
        egui::TopBottomPanel::top("toolbar").show(ctx, |ui| {
            ui.add_space(2.0);
            self.toolbar(ui);
            ui.add_space(2.0);
        });
        egui::TopBottomPanel::bottom("status").show(ctx, |ui| self.status_bar(ui));
        egui::SidePanel::left("controls").default_width(380.0).min_width(300.0).show(ctx, |ui| self.controls(ui));
        egui::CentralPanel::default().frame(egui::Frame::NONE).show(ctx, |ui| self.canvas(ui));

        if self.show_about {
            egui::Window::new("About Rusty Halftones").open(&mut self.show_about).collapsible(false).show(ctx, |ui| {
                ui.label("Halftone separations for screen printing.");
                ui.label("AM screens: round, ellipse, euclidean, diamond, square, rounded square, line.");
                ui.label("Export: 1-bit PDF and PNG film with printer's marks.");
            });
        }

        let st = self.settings_text();
        if st != self.saved_settings {
            if let Some(p) = settings_path() {
                if let Some(d) = p.parent() {
                    let _ = std::fs::create_dir_all(d);
                }
                let _ = std::fs::write(&p, &st);
            }
            self.saved_settings = st;
        }

        if self.maps_building.is_some()
            || self.gpu_building.is_some()
            || self.any_inflight()
            || self.export.is_some()
            || self.any_pending()
        {
            ctx.request_repaint_after(Duration::from_millis(30));
        }
    }
}
