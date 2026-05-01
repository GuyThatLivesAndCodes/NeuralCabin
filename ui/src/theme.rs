//! Modern monochrome theme + tiny animation helpers.
//!
//! The whole app is intentionally restricted to a black-through-white palette —
//! no accent colour. Liveliness comes from subtle animations: pulsing status
//! dots, smooth tab transitions, hover dimming.

use egui::{Color32, FontFamily, FontId, Rounding, Stroke, Visuals};

pub const BG_0: Color32 = Color32::from_rgb(0x0a, 0x0a, 0x0a);
pub const BG_1: Color32 = Color32::from_rgb(0x12, 0x12, 0x12);
pub const BG_2: Color32 = Color32::from_rgb(0x1a, 0x1a, 0x1a);
pub const BG_3: Color32 = Color32::from_rgb(0x24, 0x24, 0x24);
pub const BORDER: Color32 = Color32::from_rgb(0x2e, 0x2e, 0x2e);
pub const BORDER_STRONG: Color32 = Color32::from_rgb(0x3d, 0x3d, 0x3d);
pub const TEXT: Color32 = Color32::from_rgb(0xee, 0xee, 0xee);
pub const TEXT_WEAK: Color32 = Color32::from_rgb(0x88, 0x88, 0x88);
pub const TEXT_FAINT: Color32 = Color32::from_rgb(0x55, 0x55, 0x55);
pub const ACCENT: Color32 = Color32::from_rgb(0xff, 0xff, 0xff);
pub const DANGER: Color32 = Color32::from_rgb(0xc8, 0xc8, 0xc8);

/// Apply the monochrome theme to the egui context. Idempotent.
pub fn apply(ctx: &egui::Context) {
    let mut visuals = Visuals::dark();
    visuals.dark_mode = true;
    visuals.panel_fill = BG_0;
    visuals.window_fill = BG_1;
    visuals.window_stroke = Stroke::new(1.0, BORDER);
    visuals.window_rounding = Rounding::same(10.0);
    visuals.menu_rounding = Rounding::same(8.0);
    visuals.faint_bg_color = BG_1;
    visuals.extreme_bg_color = BG_0;
    visuals.code_bg_color = BG_2;
    visuals.override_text_color = Some(TEXT);
    visuals.hyperlink_color = TEXT;
    visuals.selection.bg_fill = Color32::from_rgb(0x33, 0x33, 0x33);
    visuals.selection.stroke = Stroke::new(1.0, ACCENT);
    visuals.warn_fg_color = TEXT;
    visuals.error_fg_color = DANGER;

    let widgets = &mut visuals.widgets;
    widgets.noninteractive.bg_fill = BG_1;
    widgets.noninteractive.weak_bg_fill = BG_1;
    widgets.noninteractive.bg_stroke = Stroke::new(1.0, BORDER);
    widgets.noninteractive.fg_stroke = Stroke::new(1.0, TEXT_WEAK);
    widgets.noninteractive.rounding = Rounding::same(6.0);

    widgets.inactive.bg_fill = BG_2;
    widgets.inactive.weak_bg_fill = BG_2;
    widgets.inactive.bg_stroke = Stroke::new(1.0, BORDER);
    widgets.inactive.fg_stroke = Stroke::new(1.0, TEXT);
    widgets.inactive.rounding = Rounding::same(6.0);

    widgets.hovered.bg_fill = BG_3;
    widgets.hovered.weak_bg_fill = BG_3;
    widgets.hovered.bg_stroke = Stroke::new(1.0, BORDER_STRONG);
    widgets.hovered.fg_stroke = Stroke::new(1.5, ACCENT);
    widgets.hovered.rounding = Rounding::same(6.0);

    widgets.active.bg_fill = Color32::from_rgb(0x32, 0x32, 0x32);
    widgets.active.weak_bg_fill = Color32::from_rgb(0x32, 0x32, 0x32);
    widgets.active.bg_stroke = Stroke::new(1.0, ACCENT);
    widgets.active.fg_stroke = Stroke::new(1.5, ACCENT);
    widgets.active.rounding = Rounding::same(6.0);

    widgets.open.bg_fill = BG_3;
    widgets.open.bg_stroke = Stroke::new(1.0, BORDER_STRONG);

    ctx.set_visuals(visuals);

    let mut style = (*ctx.style()).clone();
    style.spacing.item_spacing = egui::vec2(8.0, 6.0);
    style.spacing.button_padding = egui::vec2(10.0, 4.0);
    style.spacing.menu_margin = egui::Margin::same(6.0);
    style.spacing.window_margin = egui::Margin::same(10.0);
    style.spacing.indent = 18.0;

    style.text_styles.insert(
        egui::TextStyle::Heading,
        FontId::new(20.0, FontFamily::Proportional),
    );
    style.text_styles.insert(
        egui::TextStyle::Body,
        FontId::new(13.5, FontFamily::Proportional),
    );
    style.text_styles.insert(
        egui::TextStyle::Button,
        FontId::new(13.5, FontFamily::Proportional),
    );
    style.text_styles.insert(
        egui::TextStyle::Small,
        FontId::new(11.5, FontFamily::Proportional),
    );
    style.text_styles.insert(
        egui::TextStyle::Monospace,
        FontId::new(12.5, FontFamily::Monospace),
    );
    style.animation_time = 0.18;

    ctx.set_style(style);
}

/// A pulsing dot — used to indicate "alive" states like active training.
/// Pulse amplitude depends on `intensity` (0..1).
pub fn pulse_dot(ui: &mut egui::Ui, intensity: f32, label: &str) {
    let intensity = intensity.clamp(0.0, 1.0);
    let t = ui.ctx().input(|i| i.time) as f32;
    // 0.5 Hz pulse when "alive", static when intensity == 0.
    let phase = ((t * 1.6).sin() * 0.5 + 0.5) * intensity;
    let radius = 4.0 + 2.0 * phase;
    let alpha = (140.0 + 115.0 * phase) as u8;
    let colour = Color32::from_rgba_premultiplied(alpha, alpha, alpha, alpha);
    let size = egui::Vec2::new(14.0 + radius, 14.0);
    let (rect, _) = ui.allocate_exact_size(size, egui::Sense::hover());
    let painter = ui.painter_at(rect);
    let centre = egui::Pos2::new(rect.left() + 6.0, rect.center().y);
    if intensity > 0.0 {
        painter.circle_filled(centre, radius + 2.0, colour.gamma_multiply(0.35));
    }
    painter.circle_filled(centre, 3.5, if intensity > 0.0 { ACCENT } else { TEXT_FAINT });
    if !label.is_empty() {
        ui.add_space(2.0);
        ui.label(egui::RichText::new(label).color(TEXT_WEAK).size(11.5));
    }
    if intensity > 0.0 {
        // Schedule a repaint so the pulse animation remains smooth.
        ui.ctx().request_repaint_after(std::time::Duration::from_millis(33));
    }
}

/// Render a stylized section header with a faint underline.
pub fn section_heading(ui: &mut egui::Ui, label: &str) {
    ui.add_space(2.0);
    ui.label(
        egui::RichText::new(label)
            .color(TEXT)
            .size(15.0)
            .strong(),
    );
    let rect = ui
        .allocate_exact_size(egui::Vec2::new(ui.available_width(), 1.0), egui::Sense::hover())
        .0;
    ui.painter().line_segment(
        [rect.left_top(), rect.right_top()],
        Stroke::new(1.0, BORDER),
    );
    ui.add_space(4.0);
}

/// Coloured caption line — used for status / info messages.
pub fn caption(ui: &mut egui::Ui, msg: &str) {
    ui.label(egui::RichText::new(msg).color(TEXT_WEAK).size(11.5));
}

/// Faint hairline separator. Cheap way to make panels feel structured.
pub fn hairline(ui: &mut egui::Ui) {
    let rect = ui
        .allocate_exact_size(egui::Vec2::new(ui.available_width(), 1.0), egui::Sense::hover())
        .0;
    ui.painter().line_segment(
        [rect.left_top(), rect.right_top()],
        Stroke::new(1.0, BORDER),
    );
}
