#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod app;
mod compress;
mod engine;
mod gpu;
mod icons;

/// The release build has no console on Windows, so errors need a visible dialog.
fn show_error(msg: &str) {
    rfd::MessageDialog::new()
        .set_level(rfd::MessageLevel::Error)
        .set_title("Rusty Halftones")
        .set_description(msg)
        .show();
}

fn main() {
    std::panic::set_hook(Box::new(|info| {
        show_error(&format!("The program crashed:\n\n{}", info));
    }));

    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([1440.0, 900.0])
            .with_title("Rusty Halftones"),
        renderer: eframe::Renderer::Wgpu,
        ..Default::default()
    };
    let result = eframe::run_native("Rusty Halftones", options, Box::new(|cc| Ok(Box::new(app::HalftoneApp::new(cc)))));
    if let Err(e) = result {
        show_error(&format!("Could not start the graphics window:\n\n{}", e));
    }
}
