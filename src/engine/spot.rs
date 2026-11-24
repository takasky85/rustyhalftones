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
}

impl DotShape {
    pub const ALL: [DotShape; 7] = [
        DotShape::Round,
        DotShape::Ellipse,
        DotShape::Euclidean,
        DotShape::Diamond,
        DotShape::Square,
        DotShape::RoundedSquare,
        DotShape::Line,
    ];

    pub fn name(self) -> &'static str {
        match self {
            DotShape::Round => "Round",
            DotShape::Ellipse => "Ellipse",
            DotShape::Euclidean => "Euclidean",
            DotShape::Diamond => "Diamond",
            DotShape::Square => "Square",
            DotShape::RoundedSquare => "Rounded square",
            DotShape::Line => "Line",
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

    #[inline]
    pub fn rank(&self, fx: f32, fy: f32) -> f32 {
        let v = self.shape.raw(fx, fy, self.elong);
        let x = ((v - self.min) * self.inv * BINS as f32).clamp(0.0, BINS as f32 - 1e-3);
        let b = x as usize;
        let f = x - b as f32;
        self.cum[b] + f * (self.cum[b + 1] - self.cum[b])
    }
}
