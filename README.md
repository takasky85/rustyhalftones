# Rusty Halftones

Halftone separations for screen printing, in Rust with egui on the wgpu renderer.

Load a photo, pick a page size, place and resize the image on the page, and generate AM halftone
separations with printer's marks. Preview the halftone live, tweak it, then export film.

## Features

- **Page sizes**: Letter, Legal, Tabloid, 12x18, 13x19, 18x24, 20x24, 24x36, 5x7, 8x10, 11x14, 16x20, A0-A6, B4, B5, custom. Portrait or landscape, inches or mm.
- **Image placement**: drag to move, drag corners to resize, lock aspect, set size by ppi, fit inside the marks, fill the page, rotate 90 degrees. EXIF orientation is honoured.
- **Separation**: single plate (greyscale) or CMYK with black generation. Brightness, contrast, gamma.
- **Halftone controls**: lines per inch (with a mesh-count helper), dot shape (round, ellipse with elongation, Euclidean, diamond, square, rounded square, line), per-plate screen angle with presets, per-plate dot gain compensation, minimum dot, maximum dot, highlight dropout, invert, and optional per-plate lpi and shape.
- **Printer's marks**: crop marks, registration targets, plate label (name, angle, lpi, shape), and a 12-step tone wedge per plate (nominal dot areas, so you can measure press gain).
- **Preview**: composite overprint (multiply blend of the ink colours), any single plate as a film positive, or the original. Dots are drawn when you are zoomed in far enough to resolve them; zoomed out it shows tone instead of aliased moire.
- **Export**: PDF with one page per plate (1-bit raster halftone plus vector marks and text) and/or one 1-bit PNG per plate, at 600-2540 dpi, optionally mirrored for emulsion-down film.

## Build and run

```
cargo run --release
```

Needs a recent stable Rust. On Linux you need the usual windowing headers (see `.github/workflows/ci.yml`).

## Layout

- `src/engine/` is the halftone engine. It uses only `std`, so `cargo test` covers it without a window.
  - `spot.rs` dot shapes and the rank table that keeps every shape tone-accurate
  - `tonemap.rs` separation to ink maps and the mip pyramid sampled per cell
  - `render.rs` per-pixel halftone evaluation, preview compositor, 1-bit plate renderer
  - `layout.rs` rectangles, printer's marks, tone wedge
  - `export.rs` PDF and PNG writers
- `src/app.rs` is the egui front end. Preview and export run on background threads.

## How the halftone works

Each pixel is rotated into its plate's screen lattice. The ink value is sampled at the cell centre
from a mip level matching the cell size (an area average), adjusted by the plate's curve, and the
pixel is inked when its rank inside the dot shape is at or below that value. Because rank is the
fraction of the cell area closer to the dot centre, the printed area equals the requested tone for
any shape. The tests check this to within 2% for every shape and angle.

## Status and known limits

- Colour conversion is plain RGB to CMYK with no ICC profiles.
- Spot-colour separations, stochastic (FM) screening and project save/load are not done yet.
- PNG export draws the marks but not the text label (the PDF has the text).
- Planned: import from vectorcraft and Adobe Illustrator files. The engine treats the page as
  rectangular regions with a tone source (`PlateEval::tone_at`), which is where a vector source would plug in.
