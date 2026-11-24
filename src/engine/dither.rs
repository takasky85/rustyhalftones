//! Dithering: turn continuous ink planes (or a full-colour image and a short list of inks)
//! into 1-bit rasters whose grains are one printable dot each.
//!
//! Error diffusion pushes the quantisation error of every grain onto its neighbours, ordered
//! dithering compares against a fixed threshold pattern. Both keep the average tone, they only
//! differ in how the grains are arranged.
use super::spot::{blue_noise, MASK_N};
use super::tonemap::SourceImage;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DitherAlgo {
    FloydSteinberg,
    JarvisJudiceNinke,
    Stucki,
    Burkes,
    Sierra,
    SierraLite,
    Atkinson,
    Bayer8,
    BlueNoise,
    Threshold,
}

impl DitherAlgo {
    pub const ALL: [DitherAlgo; 10] = [
        DitherAlgo::FloydSteinberg,
        DitherAlgo::JarvisJudiceNinke,
        DitherAlgo::Stucki,
        DitherAlgo::Burkes,
        DitherAlgo::Sierra,
        DitherAlgo::SierraLite,
        DitherAlgo::Atkinson,
        DitherAlgo::Bayer8,
        DitherAlgo::BlueNoise,
        DitherAlgo::Threshold,
    ];

    pub fn name(self) -> &'static str {
        match self {
            DitherAlgo::FloydSteinberg => "Floyd-Steinberg",
            DitherAlgo::JarvisJudiceNinke => "Jarvis-Judice-Ninke",
            DitherAlgo::Stucki => "Stucki",
            DitherAlgo::Burkes => "Burkes",
            DitherAlgo::Sierra => "Sierra",
            DitherAlgo::SierraLite => "Sierra Lite",
            DitherAlgo::Atkinson => "Atkinson",
            DitherAlgo::Bayer8 => "Ordered (Bayer 8x8)",
            DitherAlgo::BlueNoise => "Ordered (blue noise)",
            DitherAlgo::Threshold => "Hard threshold",
        }
    }

    pub fn id(self) -> u8 {
        Self::ALL.iter().position(|&a| a == self).unwrap() as u8
    }

    pub fn from_id(id: u8) -> DitherAlgo {
        Self::ALL.get(id as usize).copied().unwrap_or(DitherAlgo::FloydSteinberg)
    }

    pub fn is_diffusion(self) -> bool {
        self.kernel().is_some()
    }

    /// (dx, dy, weight) taps and the divisor, for the error-diffusion algorithms.
    fn kernel(self) -> Option<(&'static [(i32, i32, f32)], f32)> {
        const FS: &[(i32, i32, f32)] = &[(1, 0, 7.0), (-1, 1, 3.0), (0, 1, 5.0), (1, 1, 1.0)];
        const JJN: &[(i32, i32, f32)] = &[
            (1, 0, 7.0),
            (2, 0, 5.0),
            (-2, 1, 3.0),
            (-1, 1, 5.0),
            (0, 1, 7.0),
            (1, 1, 5.0),
            (2, 1, 3.0),
            (-2, 2, 1.0),
            (-1, 2, 3.0),
            (0, 2, 5.0),
            (1, 2, 3.0),
            (2, 2, 1.0),
        ];
        const STUCKI: &[(i32, i32, f32)] = &[
            (1, 0, 8.0),
            (2, 0, 4.0),
            (-2, 1, 2.0),
            (-1, 1, 4.0),
            (0, 1, 8.0),
            (1, 1, 4.0),
            (2, 1, 2.0),
            (-2, 2, 1.0),
            (-1, 2, 2.0),
            (0, 2, 4.0),
            (1, 2, 2.0),
            (2, 2, 1.0),
        ];
        const BURKES: &[(i32, i32, f32)] =
            &[(1, 0, 8.0), (2, 0, 4.0), (-2, 1, 2.0), (-1, 1, 4.0), (0, 1, 8.0), (1, 1, 4.0), (2, 1, 2.0)];
        const SIERRA: &[(i32, i32, f32)] = &[
            (1, 0, 5.0),
            (2, 0, 3.0),
            (-2, 1, 2.0),
            (-1, 1, 4.0),
            (0, 1, 5.0),
            (1, 1, 4.0),
            (2, 1, 2.0),
            (-1, 2, 2.0),
            (0, 2, 3.0),
            (1, 2, 2.0),
        ];
        const LITE: &[(i32, i32, f32)] = &[(1, 0, 2.0), (-1, 1, 1.0), (0, 1, 1.0)];
        const ATKINSON: &[(i32, i32, f32)] =
            &[(1, 0, 1.0), (2, 0, 1.0), (-1, 1, 1.0), (0, 1, 1.0), (1, 1, 1.0), (0, 2, 1.0)];
        match self {
            DitherAlgo::FloydSteinberg => Some((FS, 16.0)),
            DitherAlgo::JarvisJudiceNinke => Some((JJN, 48.0)),
            DitherAlgo::Stucki => Some((STUCKI, 42.0)),
            DitherAlgo::Burkes => Some((BURKES, 32.0)),
            DitherAlgo::Sierra => Some((SIERRA, 32.0)),
            DitherAlgo::SierraLite => Some((LITE, 4.0)),
            DitherAlgo::Atkinson => Some((ATKINSON, 8.0)),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct DitherSettings {
    pub algo: DitherAlgo,
    /// Scan alternate rows in opposite directions (removes diffusion "worms").
    pub serpentine: bool,
    /// Printed grain size as dots per inch.
    pub grain_dpi: f32,
}

impl Default for DitherSettings {
    fn default() -> Self {
        DitherSettings { algo: DitherAlgo::FloydSteinberg, serpentine: true, grain_dpi: 150.0 }
    }
}

fn bayer8(x: usize, y: usize) -> f32 {
    // classic Bayer matrix: bit-reversed interleave of (x ^ y, y)
    let (xx, yy) = (x & 7, y & 7);
    let mut v = 0usize;
    for bit in 0..3 {
        let (bx, by) = ((xx >> bit) & 1, (yy >> bit) & 1);
        v = v * 4 + ((bx ^ by) * 2 + by);
    }
    (v as f32 + 0.5) / 64.0
}

/// Threshold at (x, y) for the ordered algorithms. `shift` decorrelates plates from each other.
fn ordered_threshold(algo: DitherAlgo, x: usize, y: usize, shift: (usize, usize)) -> f32 {
    match algo {
        DitherAlgo::Bayer8 => bayer8(x + shift.0, y + shift.1),
        DitherAlgo::BlueNoise => blue_noise()[((y + shift.1) % MASK_N) * MASK_N + (x + shift.0) % MASK_N],
        _ => 0.5,
    }
}

/// Dither one ink plane (values 0..1, row-major) to 0.0 / 1.0. `plate` only offsets the
/// ordered patterns so several plates do not stack their grains on top of each other.
pub fn dither_plane(w: usize, h: usize, plane: &[f32], ds: &DitherSettings, plate: usize) -> Vec<f32> {
    assert_eq!(plane.len(), w * h);
    let mut out = vec![0.0f32; w * h];
    match ds.algo.kernel() {
        None => {
            let shift = (plate * 19 + plate * plate * 7, plate * 37 + plate * plate * 5);
            for y in 0..h {
                for x in 0..w {
                    let t = ordered_threshold(ds.algo, x, y, shift);
                    out[y * w + x] = if plane[y * w + x] > t || plane[y * w + x] >= 0.9999 { 1.0 } else { 0.0 };
                }
            }
        }
        Some((taps, div)) => {
            let mut buf = plane.to_vec();
            for y in 0..h {
                let flip = ds.serpentine && y % 2 == 1;
                for i in 0..w {
                    let x = if flip { w - 1 - i } else { i };
                    let v = buf[y * w + x];
                    let q = if v >= 0.5 { 1.0 } else { 0.0 };
                    out[y * w + x] = q;
                    let err = v - q;
                    if err == 0.0 {
                        continue;
                    }
                    for &(dx, dy, wt) in taps {
                        let dx = if flip { -dx } else { dx };
                        let (nx, ny) = (x as i32 + dx, y + dy as usize);
                        if nx >= 0 && (nx as usize) < w && ny < h {
                            buf[ny * w + nx as usize] += err * wt / div;
                        }
                    }
                }
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------------------------
// Palette (spot colour) dithering

pub fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

pub fn linear_to_srgb(c: f32) -> f32 {
    let c = c.clamp(0.0, 1.0);
    if c <= 0.0031308 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

fn lin3(c: [f32; 3]) -> [f32; 3] {
    [srgb_to_linear(c[0]), srgb_to_linear(c[1]), srgb_to_linear(c[2])]
}

fn dist2(a: [f32; 3], b: [f32; 3]) -> f32 {
    // weights follow luminance sensitivity, so green mismatches count most
    let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    0.30 * d[0] * d[0] + 0.55 * d[1] * d[1] + 0.15 * d[2] * d[2]
}

fn nearest(pal: &[[f32; 3]], c: [f32; 3]) -> usize {
    let mut best = 0;
    let mut bd = f32::MAX;
    for (i, p) in pal.iter().enumerate() {
        let d = dist2(*p, c);
        if d < bd {
            bd = d;
            best = i;
        }
    }
    best
}

/// Dither a colour image to `inks` (sRGB 0..1) printed on `paper`. Returns one 0/1 plane per ink;
/// a grain gets at most one ink, paper shows where no plate prints.
pub fn dither_palette(
    img: &SourceImage,
    inks: &[[f32; 3]],
    paper: [f32; 3],
    ds: &DitherSettings,
    brightness_lut: Option<&[f32; 256]>,
) -> Vec<Vec<f32>> {
    let (w, h) = (img.w, img.h);
    let mut pal: Vec<[f32; 3]> = vec![lin3(paper)];
    pal.extend(inks.iter().map(|c| lin3(*c)));
    let mut buf: Vec<[f32; 3]> = (0..w * h)
        .map(|i| {
            let f = |c: usize| {
                let b = img.rgb[i * 3 + c] as usize;
                match brightness_lut {
                    Some(l) => l[b],
                    None => b as f32 / 255.0,
                }
            };
            lin3([f(0), f(1), f(2)])
        })
        .collect();
    let mut choice = vec![0u8; w * h];
    match ds.algo.kernel() {
        None => {
            for y in 0..h {
                for x in 0..w {
                    let t = ordered_threshold(ds.algo, x, y, (0, 0)) - 0.5;
                    let c = buf[y * w + x];
                    let spread = if ds.algo == DitherAlgo::Threshold { 0.0 } else { 0.55 };
                    let q = [c[0] + t * spread, c[1] + t * spread, c[2] + t * spread];
                    choice[y * w + x] = nearest(&pal, q) as u8;
                }
            }
        }
        Some((taps, div)) => {
            for y in 0..h {
                let flip = ds.serpentine && y % 2 == 1;
                for i in 0..w {
                    let x = if flip { w - 1 - i } else { i };
                    let c = buf[y * w + x];
                    let cc = [c[0].clamp(0.0, 1.0), c[1].clamp(0.0, 1.0), c[2].clamp(0.0, 1.0)];
                    let k = nearest(&pal, cc);
                    choice[y * w + x] = k as u8;
                    let err = [cc[0] - pal[k][0], cc[1] - pal[k][1], cc[2] - pal[k][2]];
                    for &(dx, dy, wt) in taps {
                        let dx = if flip { -dx } else { dx };
                        let (nx, ny) = (x as i32 + dx, y + dy as usize);
                        if nx >= 0 && (nx as usize) < w && ny < h {
                            let f = wt / div;
                            let t = &mut buf[ny * w + nx as usize];
                            t[0] += err[0] * f;
                            t[1] += err[1] * f;
                            t[2] += err[2] * f;
                        }
                    }
                }
            }
        }
    }
    (0..inks.len()).map(|k| choice.iter().map(|&c| if c as usize == k + 1 { 1.0 } else { 0.0 }).collect()).collect()
}

/// Pick `n` inks that represent the image well when printed on `paper` (sRGB 0..1 out).
/// Median cut seeds the colours, a few k-means rounds refine them, with paper held as a fixed centre.
pub fn auto_palette(img: &SourceImage, n: usize, paper: [f32; 3]) -> Vec<[f32; 3]> {
    if n == 0 {
        return Vec::new();
    }
    let small = img.downscaled(160);
    let pts: Vec<[f32; 3]> = (0..small.w * small.h)
        .map(|i| {
            lin3([
                small.rgb[i * 3] as f32 / 255.0,
                small.rgb[i * 3 + 1] as f32 / 255.0,
                small.rgb[i * 3 + 2] as f32 / 255.0,
            ])
        })
        .collect();
    let paper_l = lin3(paper);
    // farthest-first seeding (paper is already "chosen"), then k-means with paper held fixed
    let mut cents: Vec<[f32; 3]> = Vec::new();
    let mut mind: Vec<f32> = pts.iter().map(|p| dist2(*p, paper_l)).collect();
    for _ in 0..n {
        let far = (0..pts.len()).max_by(|&a, &b| mind[a].total_cmp(&mind[b]).then(b.cmp(&a))).unwrap();
        cents.push(pts[far]);
        for (i, p) in pts.iter().enumerate() {
            mind[i] = mind[i].min(dist2(*p, pts[far]));
        }
    }
    for _ in 0..16 {
        let mut sum = vec![[0.0f32; 3]; cents.len()];
        let mut cnt = vec![0usize; cents.len()];
        for p in &pts {
            let mut best = usize::MAX;
            let mut bd = dist2(*p, paper_l);
            for (i, c) in cents.iter().enumerate() {
                let d = dist2(*p, *c);
                if d < bd {
                    bd = d;
                    best = i;
                }
            }
            if best != usize::MAX {
                for c in 0..3 {
                    sum[best][c] += p[c];
                }
                cnt[best] += 1;
            }
        }
        for i in 0..cents.len() {
            if cnt[i] > 0 {
                let f = cnt[i] as f32;
                cents[i] = [sum[i][0] / f, sum[i][1] / f, sum[i][2] / f];
            }
        }
    }
    // darkest first reads naturally as a print order
    cents.sort_by(|a, b| (a[0] + a[1] + a[2]).total_cmp(&(b[0] + b[1] + b[2])));
    cents.iter().map(|c| [linear_to_srgb(c[0]), linear_to_srgb(c[1]), linear_to_srgb(c[2])]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mean(v: &[f32]) -> f32 {
        v.iter().sum::<f32>() / v.len() as f32
    }

    #[test]
    fn every_algorithm_keeps_the_tone() {
        let (w, h) = (160, 120);
        for algo in DitherAlgo::ALL {
            if algo == DitherAlgo::Threshold {
                continue;
            }
            for &tone in &[0.1f32, 0.37, 0.5, 0.82] {
                for serp in [false, true] {
                    let ds = DitherSettings { algo, serpentine: serp, grain_dpi: 150.0 };
                    let out = dither_plane(w, h, &vec![tone; w * h], &ds, 0);
                    let m = mean(&out);
                    // Atkinson deliberately drops a quarter of the error, which clips the extremes
                    if !(algo == DitherAlgo::Atkinson && !(0.2..=0.8).contains(&tone)) {
                        let tol = if algo == DitherAlgo::Atkinson { 0.06 } else { 0.03 };
                        assert!((m - tone).abs() < tol, "{:?} serp {} tone {} gave {}", algo, serp, tone, m);
                    }
                    assert!(out.iter().all(|&v| v == 0.0 || v == 1.0));
                }
            }
        }
    }

    #[test]
    fn gradient_ramp_is_monotonic_in_ink() {
        let (w, h) = (256, 64);
        let plane: Vec<f32> = (0..w * h).map(|i| (i % w) as f32 / (w - 1) as f32).collect();
        for algo in DitherAlgo::ALL {
            let out = dither_plane(w, h, &plane, &DitherSettings { algo, serpentine: true, grain_dpi: 150.0 }, 0);
            let q = |a: usize, b: usize| {
                let mut s = 0.0;
                for y in 0..h {
                    for x in a..b {
                        s += out[y * w + x];
                    }
                }
                s / (h * (b - a)) as f32
            };
            let (l, m, r) = (q(0, 64), q(96, 160), q(192, 256));
            assert!(l < m && m < r, "{:?}: {} {} {}", algo, l, m, r);
        }
    }

    #[test]
    fn palette_plates_never_overlap_and_match_the_average_colour() {
        let (w, h) = (96, 96);
        // left half: a mid orange that is not in the palette, right half: pure palette red
        let mut rgb = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let c = if x < w / 2 { [230, 150, 40] } else { [200, 30, 30] };
                rgb[(y * w + x) * 3..(y * w + x) * 3 + 3].copy_from_slice(&c);
            }
        }
        let img = SourceImage::new(w, h, rgb);
        let inks = [[0.8f32, 0.1, 0.1], [0.95, 0.85, 0.1], [0.1, 0.1, 0.1]];
        for algo in [DitherAlgo::FloydSteinberg, DitherAlgo::BlueNoise, DitherAlgo::Bayer8, DitherAlgo::Stucki] {
            let ds = DitherSettings { algo, serpentine: true, grain_dpi: 150.0 };
            let planes = dither_palette(&img, &inks, [1.0, 1.0, 1.0], &ds, None);
            assert_eq!(planes.len(), 3);
            for i in 0..w * h {
                let s: f32 = planes.iter().map(|p| p[i]).sum();
                assert!(s <= 1.0, "grain printed by two inks");
            }
            // average light on the orange half should be close to the source
            let mut lin = [0.0f32; 3];
            let mut n = 0.0;
            for y in 0..h {
                for x in 0..w / 2 {
                    let i = y * w + x;
                    let mut c = [1.0f32; 3];
                    for (k, p) in planes.iter().enumerate() {
                        if p[i] > 0.5 {
                            c = lin3(inks[k]);
                        }
                    }
                    if c == [1.0; 3] {
                        // paper
                    }
                    for j in 0..3 {
                        lin[j] += c[j];
                    }
                    n += 1.0;
                }
            }
            let got = [lin[0] / n, lin[1] / n, lin[2] / n];
            let want = lin3([230.0 / 255.0, 150.0 / 255.0, 40.0 / 255.0]);
            let err = ((got[0] - want[0]).abs() + (got[1] - want[1]).abs() + (got[2] - want[2]).abs()) / 3.0;
            println!("{:?}: mean colour error {:.3}", algo, err);
            assert!(err < 0.12, "{:?} colour error {}", algo, err);
        }
    }

    #[test]
    fn auto_palette_finds_the_dominant_inks() {
        let (w, h) = (64, 64);
        let mut rgb = vec![255u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let c: [u8; 3] = match (x < 24, y < 32) {
                    (true, true) => [200, 20, 20],
                    (true, false) => [20, 30, 160],
                    (false, true) => [255, 255, 255],
                    _ => [30, 140, 40],
                };
                rgb[(y * w + x) * 3..(y * w + x) * 3 + 3].copy_from_slice(&c);
            }
        }
        let pal = auto_palette(&SourceImage::new(w, h, rgb), 3, [1.0, 1.0, 1.0]);
        assert_eq!(pal.len(), 3);
        let has =
            |t: [f32; 3]| pal.iter().any(|p| (p[0] - t[0]).abs() + (p[1] - t[1]).abs() + (p[2] - t[2]).abs() < 0.15);
        assert!(has([200.0 / 255.0, 20.0 / 255.0, 20.0 / 255.0]), "{:?}", pal);
        assert!(has([20.0 / 255.0, 30.0 / 255.0, 160.0 / 255.0]), "{:?}", pal);
        assert!(has([30.0 / 255.0, 140.0 / 255.0, 40.0 / 255.0]), "{:?}", pal);
    }
}
