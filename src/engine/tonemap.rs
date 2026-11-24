//! Source image -> per-plate ink maps (0 = no ink, 1 = full ink) with a mip pyramid
//! so each halftone cell can be sampled with an area average of its own size.
use super::dither::{dither_palette, dither_plane, DitherSettings};
use super::par::par_rows;
use super::render::dot_gain_compensate;

#[derive(Clone)]
pub struct SourceImage {
    pub w: usize,
    pub h: usize,
    pub rgb: Vec<u8>,
}

impl SourceImage {
    pub fn new(w: usize, h: usize, rgb: Vec<u8>) -> Self {
        assert_eq!(rgb.len(), w * h * 3);
        SourceImage { w, h, rgb }
    }

    pub fn rotate_cw(&self) -> SourceImage {
        let (nw, nh) = (self.h, self.w);
        let mut out = vec![0u8; nw * nh * 3];
        for y in 0..self.h {
            for x in 0..self.w {
                let (xd, yd) = (self.h - 1 - y, x);
                let s = (y * self.w + x) * 3;
                let d = (yd * nw + xd) * 3;
                out[d..d + 3].copy_from_slice(&self.rgb[s..s + 3]);
            }
        }
        SourceImage { w: nw, h: nh, rgb: out }
    }

    pub fn rotate_ccw(&self) -> SourceImage {
        let (nw, nh) = (self.h, self.w);
        let mut out = vec![0u8; nw * nh * 3];
        for y in 0..self.h {
            for x in 0..self.w {
                let (xd, yd) = (y, self.w - 1 - x);
                let s = (y * self.w + x) * 3;
                let d = (yd * nw + xd) * 3;
                out[d..d + 3].copy_from_slice(&self.rgb[s..s + 3]);
            }
        }
        SourceImage { w: nw, h: nh, rgb: out }
    }

    fn halved(&self) -> SourceImage {
        let nw = (self.w + 1) / 2;
        let nh = (self.h + 1) / 2;
        let mut out = vec![0u8; nw * nh * 3];
        for y in 0..nh {
            let y0 = 2 * y;
            let y1 = (2 * y + 1).min(self.h - 1);
            for x in 0..nw {
                let x0 = 2 * x;
                let x1 = (2 * x + 1).min(self.w - 1);
                for c in 0..3 {
                    let s = self.rgb[(y0 * self.w + x0) * 3 + c] as u32
                        + self.rgb[(y0 * self.w + x1) * 3 + c] as u32
                        + self.rgb[(y1 * self.w + x0) * 3 + c] as u32
                        + self.rgb[(y1 * self.w + x1) * 3 + c] as u32;
                    out[(y * nw + x) * 3 + c] = ((s + 2) / 4) as u8;
                }
            }
        }
        SourceImage { w: nw, h: nh, rgb: out }
    }

    /// Halve (box filter) until the long side is <= `max_side`. Never upsamples.
    pub fn downscaled(&self, max_side: usize) -> SourceImage {
        let mut cur = self.clone();
        while cur.w.max(cur.h) > max_side && cur.w > 1 && cur.h > 1 {
            cur = cur.halved();
        }
        cur
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SepMode {
    /// One plate from luminance.
    Mono,
    /// Four plates: C, M, Y, K.
    Cmyk,
    /// A handful of spot inks; every grain of the image gets one ink (or bare paper).
    Palette,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct ToneSettings {
    pub mode: SepMode,
    /// -1..1
    pub brightness: f32,
    /// -1..1
    pub contrast: f32,
    /// > 1 lightens midtones
    pub gamma: f32,
    /// 0..1, how much of the grey component moves to the K plate (CMYK only).
    pub gcr: f32,
}

impl Default for ToneSettings {
    fn default() -> Self {
        ToneSettings { mode: SepMode::Cmyk, brightness: 0.0, contrast: 0.0, gamma: 1.0, gcr: 0.8 }
    }
}

fn adjust_lut(ts: &ToneSettings) -> [f32; 256] {
    let k = if ts.contrast >= 0.0 { 1.0 / (1.0 - ts.contrast * 0.95) } else { 1.0 + ts.contrast };
    let inv_g = 1.0 / ts.gamma.max(0.05);
    let mut lut = [0.0f32; 256];
    for (i, o) in lut.iter_mut().enumerate() {
        let v = (i as f32 / 255.0 - 0.5) * k + 0.5 + ts.brightness;
        *o = v.clamp(0.0, 1.0).powf(inv_g);
    }
    lut
}

pub struct Pyramid {
    levels: Vec<(usize, usize, Vec<f32>)>,
}

impl Pyramid {
    pub fn new(w: usize, h: usize, base: Vec<f32>) -> Self {
        let mut levels = vec![(w, h, base)];
        loop {
            let (pw, ph, prev) = levels.last().unwrap();
            let (pw, ph) = (*pw, *ph);
            if pw <= 1 && ph <= 1 {
                break;
            }
            let nw = (pw + 1) / 2;
            let nh = (ph + 1) / 2;
            let mut next = vec![0.0f32; nw * nh];
            for y in 0..nh {
                let y0 = 2 * y;
                let y1 = (2 * y + 1).min(ph - 1);
                for x in 0..nw {
                    let x0 = 2 * x;
                    let x1 = (2 * x + 1).min(pw - 1);
                    next[y * nw + x] =
                        0.25 * (prev[y0 * pw + x0] + prev[y0 * pw + x1] + prev[y1 * pw + x0] + prev[y1 * pw + x1]);
                }
            }
            levels.push((nw, nh, next));
        }
        Pyramid { levels }
    }

    /// The full-resolution ink plane, row-major, 0..1.
    pub fn base_data(&self) -> &[f32] {
        &self.levels[0].2
    }

    /// Value of the base-level pixel containing normalised position (u, v), no filtering.
    #[inline]
    pub fn nearest(&self, u: f32, v: f32) -> f32 {
        let (w, h, d) = (self.levels[0].0, self.levels[0].1, &self.levels[0].2);
        let x = ((u * w as f32) as usize).min(w - 1);
        let y = ((v * h as f32) as usize).min(h - 1);
        d[y * w + x]
    }

    pub fn base_size(&self) -> (usize, usize) {
        (self.levels[0].0, self.levels[0].1)
    }

    /// Smooth ink value at normalised image position (u, v), averaged over roughly
    /// `cell_px` base-level pixels.
    pub fn sample(&self, u: f32, v: f32, cell_px: f32) -> f32 {
        let n = self.levels.len();
        let lf = (cell_px.max(1.0).log2() - 1.0).clamp(0.0, (n - 1) as f32);
        let l0 = lf.floor() as usize;
        let l1 = (l0 + 1).min(n - 1);
        let t = lf - l0 as f32;
        let a = bilinear(&self.levels[l0], u, v);
        if t < 1e-3 || l0 == l1 {
            return a;
        }
        a + (bilinear(&self.levels[l1], u, v) - a) * t
    }
}

#[inline]
fn bilinear(level: &(usize, usize, Vec<f32>), u: f32, v: f32) -> f32 {
    let (w, h, d) = (level.0, level.1, &level.2);
    let x = u * w as f32 - 0.5;
    let y = v * h as f32 - 0.5;
    let x0f = x.floor();
    let y0f = y.floor();
    let fx = x - x0f;
    let fy = y - y0f;
    let cl = |i: f32, m: usize| (i.max(0.0) as usize).min(m - 1);
    let (x0, x1) = (cl(x0f, w), cl(x0f + 1.0, w));
    let (y0, y1) = (cl(y0f, h), cl(y0f + 1.0, h));
    let top = d[y0 * w + x0] * (1.0 - fx) + d[y0 * w + x1] * fx;
    let bot = d[y1 * w + x0] * (1.0 - fx) + d[y1 * w + x1] * fx;
    top * (1.0 - fy) + bot * fy
}

pub struct ToneMaps {
    pub mode: SepMode,
    pub plates: Vec<Pyramid>,
}

/// Colour separation of an already-sized image into ink planes (0 = no ink, 1 = full ink).
fn separate_planes(work: &SourceImage, ts: &ToneSettings) -> Vec<Vec<f32>> {
    let lut = adjust_lut(ts);
    let (w, h) = (work.w, work.h);
    let mut px = vec![[0.0f32; 4]; w * h];
    let mode = ts.mode;
    let gcr = ts.gcr.clamp(0.0, 1.0);
    par_rows(&mut px, w, |y, row| {
        for (x, o) in row.iter_mut().enumerate() {
            let i = (y * w + x) * 3;
            let r = lut[work.rgb[i] as usize];
            let g = lut[work.rgb[i + 1] as usize];
            let b = lut[work.rgb[i + 2] as usize];
            *o = match mode {
                SepMode::Mono | SepMode::Palette => {
                    let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
                    [1.0 - luma, 0.0, 0.0, 0.0]
                }
                SepMode::Cmyk => {
                    let (c, m, yy) = (1.0 - r, 1.0 - g, 1.0 - b);
                    let k = c.min(m).min(yy) * gcr;
                    if k >= 0.9999 {
                        [0.0, 0.0, 0.0, 1.0]
                    } else {
                        let d = 1.0 - k;
                        [(c - k) / d, (m - k) / d, (yy - k) / d, k]
                    }
                }
            };
        }
    });
    let n = if mode == SepMode::Cmyk { 4 } else { 1 };
    (0..n).map(|c| px.iter().map(|p| p[c]).collect()).collect()
}

/// Separate `src` into ink maps. `max_side` caps the working resolution (long side).
pub fn build_tone_maps(src: &SourceImage, ts: &ToneSettings, max_side: usize) -> ToneMaps {
    let work = src.downscaled(max_side);
    let (w, h) = (work.w, work.h);
    let plates = separate_planes(&work, ts).into_iter().map(|plane| Pyramid::new(w, h, plane)).collect();
    ToneMaps { mode: ts.mode, plates }
}

/// Scale `src` to exactly `w` x `h` pixels, averaging when shrinking.
pub fn resize_area(src: &SourceImage, w: usize, h: usize) -> SourceImage {
    let (w, h) = (w.max(1), h.max(1));
    let mut cur = src.clone();
    while cur.w / 2 >= w && cur.h / 2 >= h {
        cur = cur.halved();
    }
    if cur.w == w && cur.h == h {
        return cur;
    }
    let mut out = vec![0u8; w * h * 3];
    let (sx, sy) = (cur.w as f32 / w as f32, cur.h as f32 / h as f32);
    for y in 0..h {
        let fy = ((y as f32 + 0.5) * sy - 0.5).clamp(0.0, (cur.h - 1) as f32);
        let (y0, ty) = (fy.floor() as usize, fy - fy.floor());
        let y1 = (y0 + 1).min(cur.h - 1);
        for x in 0..w {
            let fx = ((x as f32 + 0.5) * sx - 0.5).clamp(0.0, (cur.w - 1) as f32);
            let (x0, tx) = (fx.floor() as usize, fx - fx.floor());
            let x1 = (x0 + 1).min(cur.w - 1);
            for c in 0..3 {
                let p = |xx: usize, yy: usize| cur.rgb[(yy * cur.w + xx) * 3 + c] as f32;
                let top = p(x0, y0) * (1.0 - tx) + p(x1, y0) * tx;
                let bot = p(x0, y1) * (1.0 - tx) + p(x1, y1) * tx;
                out[(y * w + x) * 3 + c] = (top * (1.0 - ty) + bot * ty + 0.5) as u8;
            }
        }
    }
    SourceImage { w, h, rgb: out }
}

/// Everything the dithered separation needs besides the image and tone settings.
#[derive(Clone, Debug)]
pub struct DitherJob {
    pub ds: DitherSettings,
    /// Spot inks (sRGB 0..1), used in `SepMode::Palette`.
    pub inks: Vec<[f32; 3]>,
    pub paper: [f32; 3],
    /// Size the image will be printed at, in mm, which with `grain_dpi` fixes the grain count.
    pub w_mm: f32,
    pub h_mm: f32,
    /// Dot gain to compensate for in the single-ink modes (fraction at 50%).
    pub dot_gain: f32,
}

/// Largest dithered raster, in grains. Beyond this the grain is enlarged to fit.
pub const MAX_GRAINS: f64 = 24.0e6;
/// Longest side of a dithered raster, so it always fits a GPU texture.
pub const MAX_GRAIN_SIDE: f64 = 6144.0;

/// Size of the dithered raster (in grains) for an image printed `w_mm` x `h_mm` at `dpi`.
pub fn grain_size(w_mm: f32, h_mm: f32, dpi: f32) -> (usize, usize) {
    let k = dpi.max(10.0) as f64 / 25.4;
    let (mut gw, mut gh) = ((w_mm as f64 * k).max(8.0), (h_mm as f64 * k).max(8.0));
    let over = gw * gh / MAX_GRAINS;
    if over > 1.0 {
        let f = over.sqrt();
        gw /= f;
        gh /= f;
    }
    let long = gw.max(gh);
    if long > MAX_GRAIN_SIDE {
        let f = long / MAX_GRAIN_SIDE;
        gw /= f;
        gh /= f;
    }
    (gw.round() as usize, gh.round() as usize)
}

/// Separate and dither: the maps hold only 0 and 1, one grain per pixel.
pub fn build_dither_maps(src: &SourceImage, ts: &ToneSettings, job: &DitherJob) -> ToneMaps {
    let (gw, gh) = grain_size(job.w_mm, job.h_mm, job.ds.grain_dpi);
    let work = resize_area(src, gw, gh);
    let planes: Vec<Vec<f32>> = if ts.mode == SepMode::Palette {
        let lut = adjust_lut(ts);
        dither_palette(&work, &job.inks, job.paper, &job.ds, Some(&lut))
    } else {
        let mut planes = separate_planes(&work, ts);
        for (i, plane) in planes.iter_mut().enumerate() {
            if job.dot_gain.abs() > 1e-4 {
                for v in plane.iter_mut() {
                    *v = dot_gain_compensate(*v, job.dot_gain);
                }
            }
            *plane = dither_plane(gw, gh, plane, &job.ds, i);
        }
        planes
    };
    ToneMaps { mode: ts.mode, plates: planes.into_iter().map(|p| Pyramid::new(gw, gh, p)).collect() }
}
