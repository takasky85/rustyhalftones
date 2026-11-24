//! Dot shapes ("spot functions") and the rank table that makes any shape tone-accurate.
use std::f32::consts::PI;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DotShape {
    Round,
    Ellipse,
    Euclidean,
    Diamond,
    Square,
    RoundedSquare,
    Line,
    /// Stochastic screen: grains placed by a blue-noise threshold mask instead of a spot function.
    Fm,
    /// Not a screen at all: the plate is a ready-made 1-bit raster (error diffusion, palette dither).
    Bitmap,
}

impl DotShape {
    pub const ALL: [DotShape; 8] = [
        DotShape::Round,
        DotShape::Ellipse,
        DotShape::Euclidean,
        DotShape::Diamond,
        DotShape::Square,
        DotShape::RoundedSquare,
        DotShape::Line,
        DotShape::Fm,
    ];

    /// True for the shapes that use the blue-noise mask.
    pub fn uses_mask(self) -> bool {
        matches!(self, DotShape::Fm | DotShape::Bitmap)
    }

    /// Index used by the GPU shader (same order as `ALL`).
    pub fn id(self) -> u32 {
        match self {
            DotShape::Round => 0,
            DotShape::Ellipse => 1,
            DotShape::Euclidean => 2,
            DotShape::Diamond => 3,
            DotShape::Square => 4,
            DotShape::RoundedSquare => 5,
            DotShape::Line => 6,
            DotShape::Fm => 7,
            DotShape::Bitmap => 8,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            DotShape::Round => "Round",
            DotShape::Ellipse => "Ellipse",
            DotShape::Euclidean => "Euclidean",
            DotShape::Diamond => "Diamond",
            DotShape::Square => "Square",
            DotShape::RoundedSquare => "Rounded square",
            DotShape::Line => "Line",
            DotShape::Fm => "FM (blue noise)",
            DotShape::Bitmap => "Bitmap",
        }
    }

    /// Raw spot value over a cell, `fx, fy` in [-0.5, 0.5). Lower = inked earlier as tone rises.
    #[inline]
    fn raw(self, fx: f32, fy: f32, elong: f32) -> f32 {
        match self {
            DotShape::Round => fx * fx + fy * fy,
            DotShape::Ellipse => {
                let k = elong.max(1.0);
                let a = fx * k;
                let b = fy / k;
                a * a + b * b
            }
            DotShape::Euclidean => -((2.0 * PI * fx).cos() + (2.0 * PI * fy).cos()),
            DotShape::Diamond => fx.abs() + fy.abs(),
            DotShape::Square => fx.abs().max(fy.abs()),
            DotShape::RoundedSquare => {
                let a = fx * fx;
                let b = fy * fy;
                a * a + b * b
            }
            DotShape::Line => fy.abs(),
            DotShape::Fm | DotShape::Bitmap => 0.0,
        }
    }
}

const BINS: usize = 1024;
const GRID: usize = 256;

/// Maps a cell position to its *rank*: the fraction of the cell's area whose spot value is
/// lower. A pixel is inked when `rank <= tone`, so the inked area equals the tone for
/// every shape, however irregular.
pub struct SpotTable {
    shape: DotShape,
    elong: f32,
    min: f32,
    inv: f32,
    cum: Vec<f32>,
}

impl SpotTable {
    pub fn new(shape: DotShape, elong: f32) -> Self {
        let mut min = f32::MAX;
        let mut max = f32::MIN;
        let mut vals = Vec::with_capacity(GRID * GRID);
        for j in 0..GRID {
            let fy = (j as f32 + 0.5) / GRID as f32 - 0.5;
            for i in 0..GRID {
                let fx = (i as f32 + 0.5) / GRID as f32 - 0.5;
                let v = shape.raw(fx, fy, elong);
                min = min.min(v);
                max = max.max(v);
                vals.push(v);
            }
        }
        let inv = 1.0 / (max - min).max(1e-9);
        let mut hist = vec![0u32; BINS];
        for v in &vals {
            let b = (((v - min) * inv * BINS as f32) as usize).min(BINS - 1);
            hist[b] += 1;
        }
        let total = vals.len() as f32;
        let mut cum = vec![0.0f32; BINS + 1];
        for b in 0..BINS {
            cum[b + 1] = cum[b] + hist[b] as f32 / total;
        }
        SpotTable { shape, elong, min, inv, cum }
    }

    /// `(min, inv, cum)` for the GPU: the shader recomputes the spot value and looks `cum` up.
    pub fn gpu_data(&self) -> (f32, f32, &[f32]) {
        (self.min, self.inv, &self.cum)
    }

    pub const GPU_BINS: usize = BINS;

    #[inline]
    pub fn rank(&self, fx: f32, fy: f32) -> f32 {
        let v = self.shape.raw(fx, fy, self.elong);
        let x = ((v - self.min) * self.inv * BINS as f32).clamp(0.0, BINS as f32 - 1e-3);
        let b = x as usize;
        let f = x - b as f32;
        self.cum[b] + f * (self.cum[b + 1] - self.cum[b])
    }
}

/// Side of the blue-noise tile in cells.
pub const MASK_N: usize = 64;

/// A 64 x 64 blue-noise threshold tile (void-and-cluster), values in (0, 1), each used exactly once.
/// Inking every cell whose value is at or below the tone gives that tone with evenly spread grains.
pub fn blue_noise() -> &'static [f32] {
    static MASK: std::sync::OnceLock<Vec<f32>> = std::sync::OnceLock::new();
    MASK.get_or_init(generate_blue_noise)
}

fn generate_blue_noise() -> Vec<f32> {
    const N: usize = MASK_N;
    const NN: usize = N * N;
    const R: i32 = 6;
    const SIGMA: f32 = 1.5;
    let mut kernel = Vec::new();
    for dy in -R..=R {
        for dx in -R..=R {
            kernel.push((dx, dy, (-((dx * dx + dy * dy) as f32) / (2.0 * SIGMA * SIGMA)).exp()));
        }
    }
    let apply = |energy: &mut Vec<f32>, idx: usize, sign: f32| {
        let (x, y) = ((idx % N) as i32, (idx / N) as i32);
        for &(dx, dy, w) in &kernel {
            let i = ((y + dy).rem_euclid(N as i32) as usize) * N + (x + dx).rem_euclid(N as i32) as usize;
            energy[i] += sign * w;
        }
    };
    // deterministic start: about 10% of the cells set at random
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut pat = vec![false; NN];
    let mut count = 0;
    while count < NN / 10 {
        let i = (next() % NN as u64) as usize;
        if !pat[i] {
            pat[i] = true;
            count += 1;
        }
    }
    let mut energy = vec![0.0f32; NN];
    for i in 0..NN {
        if pat[i] {
            apply(&mut energy, i, 1.0);
        }
    }
    let argmax_one = |pat: &[bool], e: &[f32]| {
        (0..NN).filter(|&i| pat[i]).max_by(|&a, &b| e[a].total_cmp(&e[b]).then(b.cmp(&a))).unwrap()
    };
    let argmin_zero = |pat: &[bool], e: &[f32]| {
        (0..NN).filter(|&i| !pat[i]).min_by(|&a, &b| e[a].total_cmp(&e[b]).then(a.cmp(&b))).unwrap()
    };
    // relax the initial pattern: move the tightest cluster into the largest void until it stops
    for _ in 0..NN * 20 {
        let c = argmax_one(&pat, &energy);
        pat[c] = false;
        apply(&mut energy, c, -1.0);
        let v = argmin_zero(&pat, &energy);
        pat[v] = true;
        apply(&mut energy, v, 1.0);
        if v == c {
            break;
        }
    }
    let mut rank = vec![0usize; NN];
    // phase 1: peel the ones off, tightest cluster first, highest rank first
    let ones = pat.iter().filter(|&&b| b).count();
    let mut work = pat.clone();
    let mut e = energy.clone();
    for r in (0..ones).rev() {
        let c = argmax_one(&work, &e);
        rank[c] = r;
        work[c] = false;
        apply(&mut e, c, -1.0);
    }
    // phases 2 and 3: fill the largest void each time
    let mut work = pat;
    let mut e = energy;
    for r in ones..NN {
        let v = argmin_zero(&work, &e);
        rank[v] = r;
        work[v] = true;
        apply(&mut e, v, 1.0);
    }
    rank.iter().map(|&r| (r as f32 + 0.5) / NN as f32).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blue_noise_is_a_permutation_with_even_spacing() {
        let m = blue_noise();
        assert_eq!(m.len(), MASK_N * MASK_N);
        let mut seen = vec![false; m.len()];
        for &v in m {
            let r = (v * m.len() as f32 - 0.5).round() as usize;
            assert!(!seen[r], "rank used twice");
            seen[r] = true;
        }
        // grains at 10% tone: nearest neighbour distances stay close to the ideal spacing
        let pts: Vec<(i32, i32)> =
            (0..m.len()).filter(|&i| m[i] < 0.10).map(|i| ((i % MASK_N) as i32, (i / MASK_N) as i32)).collect();
        let n = MASK_N as i32;
        let (mut min_d, mut sum) = (f32::MAX, 0.0f32);
        for (i, a) in pts.iter().enumerate() {
            let mut best = f32::MAX;
            for (j, b) in pts.iter().enumerate() {
                if i == j {
                    continue;
                }
                let dx = (a.0 - b.0).abs().min(n - (a.0 - b.0).abs()) as f32;
                let dy = (a.1 - b.1).abs().min(n - (a.1 - b.1).abs()) as f32;
                best = best.min((dx * dx + dy * dy).sqrt());
            }
            min_d = min_d.min(best);
            sum += best;
        }
        let ideal = (m.len() as f32 / pts.len() as f32).sqrt();
        let mean = sum / pts.len() as f32;
        println!("blue noise: {} points, ideal {:.2}, min {:.2}, mean {:.2}", pts.len(), ideal, min_d, mean);
        assert!(min_d >= 0.55 * ideal, "clumped: min {} vs ideal {}", min_d, ideal);
        assert!(mean >= 0.8 * ideal, "mean {} vs ideal {}", mean, ideal);
    }
}
