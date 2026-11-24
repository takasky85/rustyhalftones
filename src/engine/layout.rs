//! Page geometry and printer's marks. All coordinates are millimetres, origin top-left, y down.

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl Rect {
    pub fn new(x: f32, y: f32, w: f32, h: f32) -> Self {
        Rect { x, y, w, h }
    }
    pub fn right(&self) -> f32 {
        self.x + self.w
    }
    pub fn bottom(&self) -> f32 {
        self.y + self.h
    }
    #[inline]
    pub fn contains(&self, x: f32, y: f32) -> bool {
        x >= self.x && x < self.x + self.w && y >= self.y && y < self.y + self.h
    }
    pub fn union(&self, o: &Rect) -> Rect {
        let x = self.x.min(o.x);
        let y = self.y.min(o.y);
        let r = self.right().max(o.right());
        let b = self.bottom().max(o.bottom());
        Rect { x, y, w: r - x, h: b - y }
    }
}

/// Vector marks. Text is only drawn by the PDF export and the on-screen preview.
#[derive(Clone, Debug, PartialEq)]
pub enum Prim {
    Line { x1: f32, y1: f32, x2: f32, y2: f32, w: f32 },
    Ring { cx: f32, cy: f32, r: f32, w: f32 },
    Text { x: f32, y: f32, size_pt: f32, text: String },
}

pub const WEDGE_TONES: [f32; 12] = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.00];

#[derive(Clone, Debug, PartialEq)]
pub struct MarkSettings {
    pub crop: bool,
    pub reg: bool,
    pub label: bool,
    pub wedge: bool,
    /// Gap between the trim edge and the start of the marks.
    pub offset: f32,
    pub crop_len: f32,
    /// Registration target diameter.
    pub reg_size: f32,
    pub line_w: f32,
    /// Height of one plate's tone-step row.
    pub wedge_h: f32,
}

impl Default for MarkSettings {
    fn default() -> Self {
        MarkSettings {
            crop: true,
            reg: true,
            label: true,
            wedge: true,
            offset: 3.0,
            crop_len: 5.0,
            reg_size: 5.0,
            line_w: 0.15,
            wedge_h: 4.0,
        }
    }
}

const LABEL_GAP: f32 = 1.2;
const LABEL_H: f32 = 2.2;
const WEDGE_GAP: f32 = 2.0;

impl MarkSettings {
    /// Depth of the crop/registration band outside the trim edge.
    pub fn band(&self) -> f32 {
        let mut b: f32 = 0.0;
        if self.crop {
            b = b.max(self.offset + self.crop_len);
        }
        if self.reg {
            b = b.max(self.offset + 1.3 * self.reg_size);
        }
        b
    }

    /// Space the marks need outside the trim box: [left, top, right, bottom].
    pub fn margins(&self, n_plates: usize) -> [f32; 4] {
        let b = self.band();
        let top = b + if self.label { LABEL_GAP + LABEL_H } else { 0.0 };
        let bottom = b + if self.wedge { WEDGE_GAP + n_plates as f32 * self.wedge_h } else { 0.0 };
        [b, top, b, bottom]
    }

    pub fn prims(&self, trim: Rect, label: &str) -> Vec<Prim> {
        let mut v = Vec::new();
        let (l, t, r, b) = (trim.x, trim.y, trim.right(), trim.bottom());
        if self.crop {
            for &(cx, sx) in &[(l, -1.0f32), (r, 1.0)] {
                for &(cy, sy) in &[(t, -1.0f32), (b, 1.0)] {
                    v.push(Prim::Line {
                        x1: cx + sx * self.offset,
                        y1: cy,
                        x2: cx + sx * (self.offset + self.crop_len),
                        y2: cy,
                        w: self.line_w,
                    });
                    v.push(Prim::Line {
                        x1: cx,
                        y1: cy + sy * self.offset,
                        x2: cx,
                        y2: cy + sy * (self.offset + self.crop_len),
                        w: self.line_w,
                    });
                }
            }
        }
        if self.reg {
            let rr = self.reg_size * 0.5;
            let d = self.offset + 1.3 * rr;
            let (mx, my) = (trim.x + trim.w * 0.5, trim.y + trim.h * 0.5);
            for &(cx, cy) in &[(mx, t - d), (mx, b + d), (l - d, my), (r + d, my)] {
                v.push(Prim::Ring { cx, cy, r: rr, w: self.line_w });
                v.push(Prim::Ring { cx, cy, r: rr * 0.5, w: self.line_w });
                v.push(Prim::Line { x1: cx - 1.3 * rr, y1: cy, x2: cx + 1.3 * rr, y2: cy, w: self.line_w });
                v.push(Prim::Line { x1: cx, y1: cy - 1.3 * rr, x2: cx, y2: cy + 1.3 * rr, w: self.line_w });
            }
        }
        if self.label && !label.is_empty() {
            v.push(Prim::Text { x: trim.x, y: t - self.band() - LABEL_GAP, size_pt: 6.0, text: label.to_string() });
        }
        v
    }

    /// Tone-step patches for one plate. Each plate gets its own row so a composite shows the full bar.
    pub fn wedge_patches(&self, trim: Rect, plate_idx: usize) -> Vec<(Rect, f32)> {
        let n = WEDGE_TONES.len();
        let gap = 0.8;
        let pw = ((trim.w - (n as f32 - 1.0) * gap) / n as f32).clamp(2.5, 7.0);
        let y0 = trim.bottom() + self.band() + WEDGE_GAP + plate_idx as f32 * self.wedge_h;
        WEDGE_TONES
            .iter()
            .enumerate()
            .map(|(i, &t)| (Rect::new(trim.x + i as f32 * (pw + gap), y0, pw, self.wedge_h - 0.6), t))
            .collect()
    }
}
