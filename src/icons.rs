//! Toolbar icons drawn with the painter, so they do not depend on any font.
use eframe::egui::{self, vec2, Pos2, Response, Shape, Stroke, Ui};

#[derive(Clone, Copy, PartialEq)]
pub enum Icon {
    Select,
    ZoomBox,
    ZoomIn,
    ZoomOut,
    FitPage,
    FitWidth,
    ZoomImage,
}

pub fn icon_button(ui: &mut Ui, icon: Icon, selected: bool, tip: &str) -> Response {
    let (rect, resp) = ui.allocate_exact_size(vec2(36.0, 36.0), egui::Sense::click());
    let vis = ui.style().interact_selectable(&resp, selected);
    let p = ui.painter();
    if selected || resp.hovered() {
        p.rect(rect.shrink(1.0), 5.0, vis.bg_fill, vis.bg_stroke, egui::StrokeKind::Inside);
    }
    let col = vis.fg_stroke.color;
    let st = Stroke::new(1.8, col);
    let thick = Stroke::new(2.8, col);
    let c = rect.center();
    let at = |dx: f32, dy: f32| -> Pos2 { c + vec2(dx, dy) };
    // magnifier: lens up-left, handle down-right; `sign` adds a + (Some(true)) or - (Some(false))
    let magnifier = |sign: Option<bool>| {
        let lens = at(-3.0, -3.0);
        p.circle_stroke(lens, 8.0, st);
        p.line_segment([at(3.0, 3.0), at(11.0, 11.0)], thick);
        if let Some(plus) = sign {
            p.line_segment([lens + vec2(-4.0, 0.0), lens + vec2(4.0, 0.0)], st);
            if plus {
                p.line_segment([lens + vec2(0.0, -4.0), lens + vec2(0.0, 4.0)], st);
            }
        }
    };
    match icon {
        Icon::Select => {
            let pts = vec![
                at(-6.0, -9.0),
                at(-6.0, 8.0),
                at(-2.0, 4.0),
                at(2.0, 11.0),
                at(5.0, 9.5),
                at(1.5, 3.0),
                at(7.0, 3.0),
            ];
            p.add(Shape::closed_line(pts, st));
        }
        Icon::ZoomBox => magnifier(None),
        Icon::ZoomIn => magnifier(Some(true)),
        Icon::ZoomOut => magnifier(Some(false)),
        Icon::FitPage => {
            p.rect_stroke(egui::Rect::from_center_size(c, vec2(16.0, 21.0)), 1.0, st, egui::StrokeKind::Middle);
            p.rect_stroke(
                egui::Rect::from_center_size(c, vec2(26.0, 29.0)),
                0.0,
                Stroke::new(1.0, col),
                egui::StrokeKind::Middle,
            );
        }
        Icon::FitWidth => {
            p.rect_stroke(
                egui::Rect::from_center_size(c + vec2(0.0, -3.0), vec2(22.0, 16.0)),
                1.0,
                st,
                egui::StrokeKind::Middle,
            );
            p.line_segment([at(-11.0, 9.0), at(11.0, 9.0)], st);
            p.line_segment([at(-11.0, 9.0), at(-7.0, 6.0)], st);
            p.line_segment([at(-11.0, 9.0), at(-7.0, 12.0)], st);
            p.line_segment([at(11.0, 9.0), at(7.0, 6.0)], st);
            p.line_segment([at(11.0, 9.0), at(7.0, 12.0)], st);
        }
        Icon::ZoomImage => {
            p.rect_stroke(egui::Rect::from_center_size(c, vec2(24.0, 24.0)), 1.0, st, egui::StrokeKind::Middle);
            p.rect_filled(egui::Rect::from_center_size(c, vec2(12.0, 12.0)), 1.0, col);
        }
    }
    resp.on_hover_text(tip)
}
