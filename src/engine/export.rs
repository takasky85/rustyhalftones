//! Film output: one 1-bit raster per plate, written as a multi-page PDF (raster halftone +
//! vector printer's marks and text) and/or one 1-bit PNG per plate. No external crates:
//! the caller supplies a zlib compressor.
use std::fmt::Write as _;
use std::io;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use super::layout::Prim;
use super::render::{render_plate_bitmap, Bitmap, PlateEval};

pub type Deflate<'a> = &'a (dyn Fn(&[u8]) -> Vec<u8> + Sync);

pub struct PlateOut<'a> {
    pub name: &'a str,
    pub eval: &'a PlateEval<'a>,
    pub prims: &'a [Prim],
}

pub struct ExportSpec {
    pub page_w: f32,
    pub page_h: f32,
    pub dpi: f32,
    /// Wrong-reading output (flip left-right) for emulsion-down film.
    pub mirror: bool,
    pub pdf: Option<PathBuf>,
    /// (directory, file stem). Files are `<stem>_<plate>.png`.
    pub png: Option<(PathBuf, String)>,
}

fn cancelled() -> io::Error {
    io::Error::new(io::ErrorKind::Interrupted, "export cancelled")
}

/// Render every plate once and write the requested formats. Returns the files written.
pub fn export_all(
    spec: &ExportSpec,
    plates: &[PlateOut],
    deflate: Deflate,
    progress: &AtomicUsize,
    cancel: &AtomicBool,
) -> io::Result<Vec<PathBuf>> {
    let mut written = Vec::new();
    let mut images: Vec<(usize, usize, Vec<u8>)> = Vec::new();
    for pl in plates {
        let mut bm = render_plate_bitmap(pl.eval, spec.page_w, spec.page_h, spec.dpi, spec.mirror, progress, cancel);
        if cancel.load(Ordering::Relaxed) {
            return Err(cancelled());
        }
        if spec.pdf.is_some() {
            images.push((bm.w, bm.h, deflate(&bm.data)));
        }
        if let Some((dir, stem)) = &spec.png {
            draw_prims(&mut bm, pl.prims, spec.page_w, spec.page_h, spec.dpi, spec.mirror);
            let path = dir.join(format!("{}_{}.png", stem, sanitize(pl.name)));
            std::fs::write(&path, png_1bit(&bm, spec.dpi, deflate))?;
            written.push(path);
        }
    }
    if let Some(path) = &spec.pdf {
        std::fs::write(path, build_pdf(spec, plates, &images))?;
        written.push(path.clone());
    }
    Ok(written)
}

fn sanitize(s: &str) -> String {
    s.chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '_' }).collect()
}

// ---------------------------------------------------------------------------------------
// Raster marks (PNG output). Text is skipped.

pub fn draw_prims(bm: &mut Bitmap, prims: &[Prim], page_w: f32, _page_h: f32, dpi: f32, mirror: bool) {
    let k = dpi / 25.4;
    let mx = |x: f32| if mirror { page_w - x } else { x };
    for p in prims {
        match p {
            Prim::Line { x1, y1, x2, y2, w } => {
                stroke(
                    bm,
                    |px, py| seg_dist(px, py, mx(*x1) * k, y1 * k, mx(*x2) * k, y2 * k),
                    (mx(*x1) * k).min(mx(*x2) * k),
                    (y1 * k).min(y2 * k),
                    (mx(*x1) * k).max(mx(*x2) * k),
                    (y1 * k).max(y2 * k),
                    (w * k).max(1.0),
                );
            }
            Prim::Ring { cx, cy, r, w } => {
                let (cxp, cyp, rp) = (mx(*cx) * k, cy * k, r * k);
                stroke(
                    bm,
                    |px, py| ((px - cxp).hypot(py - cyp) - rp).abs(),
                    cxp - rp,
                    cyp - rp,
                    cxp + rp,
                    cyp + rp,
                    (w * k).max(1.0),
                );
            }
            Prim::Text { .. } => {}
        }
    }
}

fn seg_dist(px: f32, py: f32, ax: f32, ay: f32, bx: f32, by: f32) -> f32 {
    let (dx, dy) = (bx - ax, by - ay);
    let l2 = dx * dx + dy * dy;
    let t = if l2 > 0.0 { (((px - ax) * dx + (py - ay) * dy) / l2).clamp(0.0, 1.0) } else { 0.0 };
    (px - (ax + t * dx)).hypot(py - (ay + t * dy))
}

fn stroke(bm: &mut Bitmap, dist: impl Fn(f32, f32) -> f32, x0: f32, y0: f32, x1: f32, y1: f32, width: f32) {
    let hw = width * 0.5;
    let xa = ((x0 - hw - 1.0).floor().max(0.0)) as usize;
    let ya = ((y0 - hw - 1.0).floor().max(0.0)) as usize;
    let xb = (((x1 + hw + 1.0).ceil()).max(0.0) as usize).min(bm.w);
    let yb = (((y1 + hw + 1.0).ceil()).max(0.0) as usize).min(bm.h);
    for y in ya..yb {
        for x in xa..xb {
            if dist(x as f32 + 0.5, y as f32 + 0.5) <= hw {
                bm.data[y * bm.stride + (x >> 3)] &= !(0x80u8 >> (x & 7));
            }
        }
    }
}

// ---------------------------------------------------------------------------------------
// PNG

fn crc32_table() -> [u32; 256] {
    let mut t = [0u32; 256];
    for (n, e) in t.iter_mut().enumerate() {
        let mut c = n as u32;
        for _ in 0..8 {
            c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
        }
        *e = c;
    }
    t
}

fn chunk(out: &mut Vec<u8>, table: &[u32; 256], kind: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    let start = out.len();
    out.extend_from_slice(kind);
    out.extend_from_slice(data);
    let mut c = 0xFFFF_FFFFu32;
    for &b in &out[start..] {
        c = table[((c ^ b as u32) & 0xFF) as usize] ^ (c >> 8);
    }
    out.extend_from_slice(&(c ^ 0xFFFF_FFFF).to_be_bytes());
}

/// 1-bit greyscale PNG (0 = black = ink), with the print resolution in a pHYs chunk.
pub fn png_1bit(bm: &Bitmap, dpi: f32, deflate: Deflate) -> Vec<u8> {
    let table = crc32_table();
    let mut out = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&(bm.w as u32).to_be_bytes());
    ihdr.extend_from_slice(&(bm.h as u32).to_be_bytes());
    ihdr.extend_from_slice(&[1, 0, 0, 0, 0]); // depth 1, greyscale, deflate, filter 0, no interlace
    chunk(&mut out, &table, b"IHDR", &ihdr);
    let ppm = (dpi / 0.0254).round() as u32;
    let mut phys = Vec::new();
    phys.extend_from_slice(&ppm.to_be_bytes());
    phys.extend_from_slice(&ppm.to_be_bytes());
    phys.push(1);
    chunk(&mut out, &table, b"pHYs", &phys);
    let mut raw = Vec::with_capacity(bm.h * (bm.stride + 1));
    for y in 0..bm.h {
        raw.push(0);
        raw.extend_from_slice(&bm.data[y * bm.stride..(y + 1) * bm.stride]);
    }
    chunk(&mut out, &table, b"IDAT", &deflate(&raw));
    chunk(&mut out, &table, b"IEND", &[]);
    out
}

/// Valid zlib stream using stored (uncompressed) blocks. For tests and as a fallback.
pub fn zlib_stored(data: &[u8]) -> Vec<u8> {
    let mut out = vec![0x78, 0x01];
    if data.is_empty() {
        out.extend_from_slice(&[1, 0, 0, 0xFF, 0xFF]);
    }
    let mut chunks = data.chunks(65535).peekable();
    while let Some(c) = chunks.next() {
        out.push(if chunks.peek().is_none() { 1 } else { 0 });
        out.extend_from_slice(&(c.len() as u16).to_le_bytes());
        out.extend_from_slice(&(!(c.len() as u16)).to_le_bytes());
        out.extend_from_slice(c);
    }
    let (mut a, mut b) = (1u32, 0u32);
    for &x in data {
        a = (a + x as u32) % 65521;
        b = (b + a) % 65521;
    }
    out.extend_from_slice(&((b << 16) | a).to_be_bytes());
    out
}

// ---------------------------------------------------------------------------------------
// PDF

const PT: f32 = 72.0 / 25.4;

fn pdf_escape(s: &str) -> String {
    let mut o = String::new();
    for c in s.chars() {
        match c {
            '(' | ')' | '\\' => {
                o.push('\\');
                o.push(c);
            }
            c if c.is_ascii() && !c.is_ascii_control() => o.push(c),
            _ => o.push('?'),
        }
    }
    o
}

fn content_stream(spec: &ExportSpec, prims: &[Prim]) -> String {
    let (wp, hp) = (spec.page_w * PT, spec.page_h * PT);
    let mx = |x: f32| if spec.mirror { spec.page_w - x } else { x };
    let px = |x: f32| mx(x) * PT;
    let py = |y: f32| (spec.page_h - y) * PT;
    let mut s = String::new();
    let _ = writeln!(s, "q 0 g 0 G");
    let _ = writeln!(s, "q {:.3} 0 0 {:.3} 0 0 cm /Im0 Do Q", wp, hp);
    for p in prims {
        match p {
            Prim::Line { x1, y1, x2, y2, w } => {
                let _ =
                    writeln!(s, "{:.3} w {:.3} {:.3} m {:.3} {:.3} l S", w * PT, px(*x1), py(*y1), px(*x2), py(*y2));
            }
            Prim::Ring { cx, cy, r, w } => {
                let (x, y, r) = (px(*cx), py(*cy), r * PT);
                let k = 0.552_284_75 * r;
                let _ = writeln!(
                    s,
                    "{:.3} w {:.3} {:.3} m \
                     {:.3} {:.3} {:.3} {:.3} {:.3} {:.3} c \
                     {:.3} {:.3} {:.3} {:.3} {:.3} {:.3} c \
                     {:.3} {:.3} {:.3} {:.3} {:.3} {:.3} c \
                     {:.3} {:.3} {:.3} {:.3} {:.3} {:.3} c h S",
                    w * PT,
                    x + r,
                    y,
                    x + r,
                    y + k,
                    x + k,
                    y + r,
                    x,
                    y + r,
                    x - k,
                    y + r,
                    x - r,
                    y + k,
                    x - r,
                    y,
                    x - r,
                    y - k,
                    x - k,
                    y - r,
                    x,
                    y - r,
                    x + k,
                    y - r,
                    x + r,
                    y - k,
                    x + r,
                    y
                );
            }
            Prim::Text { x, y, size_pt, text } => {
                let sx = if spec.mirror { -1.0 } else { 1.0 };
                let _ = writeln!(
                    s,
                    "BT /F1 {:.2} Tf {} 0 0 1 {:.3} {:.3} Tm ({}) Tj ET",
                    size_pt,
                    sx,
                    px(*x),
                    py(*y),
                    pdf_escape(text)
                );
            }
        }
    }
    s.push_str("Q\n");
    s
}

fn build_pdf(spec: &ExportSpec, plates: &[PlateOut], images: &[(usize, usize, Vec<u8>)]) -> Vec<u8> {
    let (wp, hp) = (spec.page_w * PT, spec.page_h * PT);
    let mut buf: Vec<u8> = b"%PDF-1.4\n%\xE2\xE3\xCF\xD3\n".to_vec();
    let total_objs = 3 + 3 * plates.len();
    let mut offs = vec![0usize; total_objs];

    let mut obj = |buf: &mut Vec<u8>, id: usize, dict: &str, stream: Option<&[u8]>| {
        offs[id - 1] = buf.len();
        buf.extend_from_slice(format!("{} 0 obj\n", id).as_bytes());
        match stream {
            None => buf.extend_from_slice(format!("{}\n", dict).as_bytes()),
            Some(data) => {
                buf.extend_from_slice(format!("<< {} /Length {} >>\nstream\n", dict, data.len()).as_bytes());
                buf.extend_from_slice(data);
                buf.extend_from_slice(b"\nendstream\n");
            }
        }
        buf.extend_from_slice(b"endobj\n");
    };

    obj(&mut buf, 1, "<< /Type /Catalog /Pages 2 0 R >>", None);
    let kids: String = (0..plates.len()).map(|k| format!("{} 0 R ", 4 + 3 * k)).collect();
    obj(&mut buf, 2, &format!("<< /Type /Pages /Count {} /Kids [{}] >>", plates.len(), kids), None);
    obj(&mut buf, 3, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>", None);

    for (k, pl) in plates.iter().enumerate() {
        let (page, content, image) = (4 + 3 * k, 5 + 3 * k, 6 + 3 * k);
        obj(
            &mut buf,
            page,
            &format!(
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {:.3} {:.3}] \
                 /Resources << /XObject << /Im0 {} 0 R >> /Font << /F1 3 0 R >> >> /Contents {} 0 R >>",
                wp, hp, image, content
            ),
            None,
        );
        obj(&mut buf, content, "", Some(content_stream(spec, pl.prims).as_bytes()));
        let (iw, ih, z) = &images[k];
        obj(
            &mut buf,
            image,
            &format!(
                "/Type /XObject /Subtype /Image /Width {} /Height {} /ImageMask true \
                 /BitsPerComponent 1 /Filter /FlateDecode",
                iw, ih
            ),
            Some(z),
        );
    }

    let xref = buf.len();
    buf.extend_from_slice(format!("xref\n0 {}\n0000000000 65535 f \n", total_objs + 1).as_bytes());
    for o in &offs {
        buf.extend_from_slice(format!("{:010} 00000 n \n", o).as_bytes());
    }
    buf.extend_from_slice(
        format!("trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n", total_objs + 1, xref).as_bytes(),
    );
    buf
}
