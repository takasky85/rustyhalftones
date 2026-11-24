#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod app;
mod compress;
mod engine;

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([1440.0, 900.0])
            .with_title("Rusty Halftones"),
        renderer: eframe::Renderer::Wgpu,
        ..Default::default()
    };
    eframe::run_native("Rusty Halftones", options, Box::new(|cc| Ok(Box::new(app::HalftoneApp::new(cc)))))
}
