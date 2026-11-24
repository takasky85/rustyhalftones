# Rusty Halftones

Halftone separations for screen printing, in Rust with egui on the wgpu renderer.

Load a photo, pick a page size, place and resize the image on the page, and generate AM halftone
separations with printer's marks. Preview the halftone live, tweak it, then export film.

## Features

- **Page sizes**: Letter, Legal, Tabloid, 12x18, 13x19, 18x24, 20x24, 24x36, 5x7, 8x10, 11x14, 16x20, A0-A6, B4, B5, custom. Portrait or landscape, inches or mm.
- **Image placement**: drag to move, drag corners to resize, lock aspect, set size by ppi, rotate 90 degrees. Fit to page, fit width, fit height or fill page; fitting keeps the image and its printer's marks inside a safe margin (default 1/4 in) and re-fits when the page, marks or margin change. EXIF orientation is honoured.
- **Separation**: single plate (greyscale), CMYK with black generation, or spot colours from a palette. Brightness, contrast, gamma.
- **Screening**: AM halftone dots, FM (stochastic) blue-noise grains, or dithering: error diffusion (Floyd-Steinberg, Jarvis-Judice-Ninke, Stucki, Burkes, Sierra, Sierra Lite, Atkinson, optional serpentine scan), ordered Bayer 8x8, ordered blue noise, hard threshold. Dither grain size is set in dpi; each grain is one printed dot.
- **Spot-colour palette**: choose up to 8 inks (or pick N from the image automatically), and the picture is dithered to those inks on white paper. Each ink becomes a plate, no grain is printed by two inks, so there is nothing to overprint or trap.
- **Halftone controls**: lines per inch (with a mesh-count helper), dot shape (round, ellipse with elongation, Euclidean, diamond, square, rounded square, line), per-plate screen angle with presets, per-plate dot gain compensation, minimum dot, maximum dot, highlight dropout, invert, and optional per-plate lpi and shape.
- **Printer's marks**: crop marks, registration targets, plate label (name, angle, lpi, shape), and a 12-step tone wedge per plate (nominal dot areas, so you can measure press gain).
- **Preview**: composite overprint (multiply blend of the ink colours), any single plate as a film positive, all plates on separate pages (2x2, one row, one column or automatic), optionally in ink colours, or the original. Dots are drawn when you are zoomed in far enough to resolve them; zoomed out it shows tone instead of aliased moire. The preview runs on the GPU (a WGSL port of the engine), so dot size, lpi, angle, shape and dot gain update in real time. Options > Preview renderer switches to the CPU preview, which is also the automatic fallback; `--cpu-preview` forces it.
- **Interface**: File/Edit/View/Options/Help menus, a toolbar with select, zoom-box and hand tools, zoom in/out, fit page, fit width, fit height and a zoom percentage box, rulers and scroll bars (View menu), mouse wheel scrolls, Ctrl+wheel zooms, middle or right drag pans, page columns follow the zoom (one, two or all plates across), tabbed settings (Layout, Separation, Plates, Export), adjustable interface size.
- **Export**: PDF with one page per plate (1-bit raster halftone plus vector marks and text) and/or one 1-bit PNG per plate, at 600-2540 dpi, optionally mirrored for emulsion-down film.

## Build and run

```
cargo run --release
```

Needs a recent stable Rust. On Linux you need the usual windowing headers (see `.github/workflows/ci.yml`).

## Layout

- `src/engine/` is the halftone engine. It uses only `std`, so `cargo test` covers it without a window.
  - `spot.rs` dot shapes, the rank table that keeps every shape tone-accurate, and the void-and-cluster blue-noise mask
  - `dither.rs` error diffusion, ordered dither, palette dithering and automatic palette picking
  - `tonemap.rs` separation to ink maps and the mip pyramid sampled per cell
  - `render.rs` per-pixel halftone evaluation, preview compositor, 1-bit plate renderer
  - `layout.rs` rectangles, printer's marks, tone wedge
  - `export.rs` PDF and PNG writers
- `src/app.rs` is the egui front end. CPU preview and export run on background threads.
- `src/gpu.rs` and `src/gpu.wgsl` are the GPU preview: ink maps uploaded as half-float mip chains, spot tables and per-plate parameters as uniforms, one fragment-shader draw per page through an egui paint callback.
- `src/icons.rs` toolbar icons drawn with the painter.

CI also runs `cargo test --features gpu-tests gpu`, which renders every dot shape on a (software) adapter and compares the result with the CPU renderer.

## How FM and dithering work

FM screening reuses the screen lattice: every cell is one grain, switched on when a 64 x 64
void-and-cluster blue-noise mask value is at or below the tone, so grains stay evenly spread at
every tone. Error diffusion and palette dithering are done once per image at the chosen grain
resolution, into 0/1 rasters; the preview and export read those rasters directly. CI checks that
the GPU shader gives the same pixels as the CPU renderer for FM and for dithered plates.

## How the halftone works

Each pixel is rotated into its plate's screen lattice. The ink value is sampled at the cell centre
from a mip level matching the cell size (an area average), adjusted by the plate's curve, and the
pixel is inked when its rank inside the dot shape is at or below that value. Because rank is the
fraction of the cell area closer to the dot centre, the printed area equals the requested tone for
any shape. The tests check this to within 2% for every shape and angle.

## Status and known limits

- Colour conversion is plain RGB to CMYK with no ICC profiles.
- Dithered plates are rasters at the grain size you pick. Export at the same dpi or a whole multiple of it, otherwise grains are resampled and look uneven. The composite preview always sits on white paper, so a coloured substrate is not previewed.
- Project save/load is not done yet.
- PNG export draws the marks but not the text label (the PDF has the text).
- Changing tone settings (brightness, contrast, gamma, black generation) still rebuilds the ink maps on the CPU before the GPU preview updates; screen settings are instant.
- Settings (interface size, units, safe margin, renderer) are saved to `%APPDATA%\RustyHalftones\settings.txt`.
- Planned: import from vectorcraft and Adobe Illustrator files. The engine treats the page as
  rectangular regions with a tone source (`PlateEval::tone_at`), which is where a vector source would plug in.
