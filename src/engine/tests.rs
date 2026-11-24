use std::sync::atomic::{AtomicBool, AtomicUsize};

use super::export::*;
use super::layout::*;
use super::render::*;
use super::spot::*;
use super::tonemap::*;

fn uniform_maps(tone: f32, plates: usize) -> ToneMaps {
    ToneMaps {
        mode: if plates == 1 { SepMode::Mono } else { SepMode::Cmyk },
        plates: (0..plates).map(|_| Pyramid::new(8, 8, vec![tone; 64])).collect(),
    }
}

/// Fraction of a uniform-tone image area that ends up inked.
fn coverage(shape: DotShape, angle: f32, tone: f32) -> f32 {
    let maps = uniform_maps(tone, 1);
    let spot = SpotTable::new(shape, 1.4);
    let params = PlateParams { angle_deg: angle, lpi: 45.0, shape, min_dot: 0.0, dropout: 0.0, ..Default::default() };
    let ev = PlateEval::new(params, &spot, &maps.plates[0], Rect::new(0.0, 0.0, 60.0, 60.0), &[]);
    let n = (60.0 / 25.4 * 600.0) as usize;
    let (mut on, mut tot) = (0usize, 0usize);
    let mut cell = Cell::default();
    for j in 0..n {
        for i in 0..n {
            let x = (i as f32 + 0.5) * 60.0 / n as f32;
            let y = (j as f32 + 0.5) * 60.0 / n as f32;
            if !(10.0..=50.0).contains(&x) || !(10.0..=50.0).contains(&y) {
                continue;
            }
            tot += 1;
            if ev.is_on(x, y, &mut cell) {
                on += 1;
            }
        }
    }
    on as f32 / tot as f32
}

#[test]
fn ink_area_matches_tone_for_every_shape_and_angle() {
    for shape in DotShape::ALL {
        for angle in [0.0, 15.0, 45.0, 75.0] {
            for t in [0.05, 0.25, 0.5, 0.75, 0.95] {
                let c = coverage(shape, angle, t);
                assert!((c - t).abs() < 0.02, "{} at {} deg, tone {}: measured {}", shape.name(), angle, t, c);
            }
        }
    }
}

#[test]
fn dot_gain_compensation_is_monotonic_and_pins_ends() {
    let mut prev = -1.0;
    for i in 0..=100 {
        let v = dot_gain_compensate(i as f32 / 100.0, 0.2);
        assert!(v >= prev);
        prev = v;
    }
    assert_eq!(dot_gain_compensate(0.0, 0.2), 0.0);
    assert!((dot_gain_compensate(1.0, 0.2) - 1.0).abs() < 1e-5);
    // compensating then applying 20% gain at the midpoint should land back on 50%
    let x = dot_gain_compensate(0.5, 0.2);
    assert!((x + 4.0 * 0.2 * x * (1.0 - x) - 0.5).abs() < 1e-4);
}

#[test]
fn marks_fit_inside_reported_margins() {
    let m = MarkSettings::default();
    let trim = Rect::new(50.0, 50.0, 100.0, 80.0);
    let [l, t, r, b] = m.margins(4);
    let outer = Rect::new(trim.x - l, trim.y - t, trim.w + l + r, trim.h + t + b);
    for p in m.prims(trim, "label") {
        let pts: Vec<(f32, f32)> = match p {
            Prim::Line { x1, y1, x2, y2, .. } => vec![(x1, y1), (x2, y2)],
            Prim::Ring { cx, cy, r, .. } => vec![(cx - r, cy - r), (cx + r, cy + r)],
            Prim::Text { x, y, .. } => vec![(x, y)],
        };
        for (x, y) in pts {
            assert!(x >= outer.x - 0.01 && x <= outer.right() + 0.01, "x {} outside {:?}", x, outer);
            assert!(y >= outer.y - 0.01 && y <= outer.bottom() + 0.01, "y {} outside {:?}", y, outer);
        }
    }
    for i in 0..4 {
        for (rect, _) in m.wedge_patches(trim, i) {
            assert!(rect.bottom() <= outer.bottom() + 0.01);
        }
    }
}

#[test]
fn separation_and_export_produce_valid_files() {
    let (w, h) = (64usize, 64usize);
    let mut rgb = Vec::new();
    for y in 0..h {
        for x in 0..w {
            rgb.extend_from_slice(&[(x * 4) as u8, (y * 4) as u8, 128]);
        }
    }
    let src = SourceImage::new(w, h, rgb);
    let maps = build_tone_maps(&src, &ToneSettings::default(), 3072);
    assert_eq!(maps.plates.len(), 4);

    let marks = MarkSettings::default();
    let m = marks.margins(4);
    let image = Rect::new(m[0] + 5.0, m[1] + 5.0, 60.0, 60.0);
    let (pw, ph) = (image.right() + m[2] + 5.0, image.bottom() + m[3] + 5.0);
    let plates: Vec<PlateSetup> = ["C", "M", "Y", "K"]
        .iter()
        .map(|n| PlateSetup { name: n.to_string(), params: PlateParams::default(), ink: [0.0; 3], visible: true })
        .collect();
    let scene = Scene { page_w: pw, page_h: ph, image, marks: marks.clone(), plates };
    let prep = scene.prepare();
    let evals = scene.evals(&prep, &maps);
    let prims: Vec<Vec<Prim>> = (0..4).map(|i| marks.prims(image, &scene.label(i, "t"))).collect();
    let outs: Vec<PlateOut> =
        (0..4).map(|i| PlateOut { name: &scene.plates[i].name, eval: &evals[i], prims: &prims[i] }).collect();

    let dir = std::env::temp_dir();
    let spec = ExportSpec {
        page_w: pw,
        page_h: ph,
        dpi: 150.0,
        mirror: false,
        pdf: Some(dir.join("rustyhalftones_test.pdf")),
        png: Some((dir.clone(), "rustyhalftones_test".to_string())),
    };
    let files = export_all(&spec, &outs, &zlib_stored, &AtomicUsize::new(0), &AtomicBool::new(false)).unwrap();
    assert_eq!(files.len(), 5);
    let pdf = std::fs::read(dir.join("rustyhalftones_test.pdf")).unwrap();
    assert!(pdf.starts_with(b"%PDF-1.4"));
    assert!(pdf.ends_with(b"%%EOF\n"));
    let png = std::fs::read(dir.join("rustyhalftones_test_K.png")).unwrap();
    assert_eq!(&png[..8], &[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);
}

#[test]
fn dithered_plates_print_the_requested_ink_area_and_palette_plates_stay_disjoint() {
    use super::dither::*;
    // flat mid grey 60x60 mm image
    let src = SourceImage::new(16, 16, vec![128u8; 16 * 16 * 3]);
    let image = Rect::new(0.0, 0.0, 60.0, 60.0);
    let job = DitherJob {
        ds: DitherSettings { algo: DitherAlgo::FloydSteinberg, serpentine: true, grain_dpi: 100.0 },
        inks: vec![[0.0, 0.0, 0.0]],
        paper: [1.0, 1.0, 1.0],
        w_mm: 60.0,
        h_mm: 60.0,
        dot_gain: 0.0,
    };
    let ts = ToneSettings { mode: SepMode::Mono, ..Default::default() };
    let maps = build_dither_maps(&src, &ts, &job);
    assert_eq!(maps.plates.len(), 1);
    let spot = SpotTable::new(DotShape::Bitmap, 1.0);
    let params = PlateParams { shape: DotShape::Bitmap, lpi: 100.0, ..Default::default() };
    let ev = PlateEval::new(params, &spot, &maps.plates[0], image, &[]);
    let mut cell = Cell::default();
    let n = 400;
    let mut on = 0;
    for j in 0..n {
        for i in 0..n {
            if ev.is_on((i as f32 + 0.5) * 60.0 / n as f32, (j as f32 + 0.5) * 60.0 / n as f32, &mut cell) {
                on += 1;
            }
        }
    }
    let area = on as f32 / (n * n) as f32;
    // grey 128 is about 50% ink in this simple separation
    assert!(
        (area - (1.0 - 0.7152 * 128.0 / 255.0 - 0.2126 * 128.0 / 255.0 - 0.0722 * 128.0 / 255.0)).abs() < 0.03,
        "area {}",
        area
    );

    // palette: two inks, no grain printed twice
    let mut rgb = Vec::new();
    for y in 0..16 {
        for x in 0..16 {
            rgb.extend_from_slice(&[(x * 16) as u8, (y * 16) as u8, 90]);
        }
    }
    let src = SourceImage::new(16, 16, rgb);
    let job = DitherJob { inks: vec![[0.9, 0.1, 0.1], [0.1, 0.1, 0.8]], ..job };
    let ts = ToneSettings { mode: SepMode::Palette, ..Default::default() };
    let maps = build_dither_maps(&src, &ts, &job);
    assert_eq!(maps.plates.len(), 2);
    let (w, h) = maps.plates[0].base_size();
    for i in 0..w * h {
        assert!(maps.plates[0].base_data()[i] + maps.plates[1].base_data()[i] <= 1.0);
    }
}
