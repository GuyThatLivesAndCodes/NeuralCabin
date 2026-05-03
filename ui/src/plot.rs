//! Tiny line-plot and scatter-plot widgets built on egui's primitive painter —
//! no external plotting crate required.

use egui::{epaint::PathStroke, Color32, Pos2, Rect, Sense, Stroke, Ui, Vec2};

pub struct LinePlot<'a> {
    pub title: &'a str,
    pub series: Vec<(&'a str, &'a [f32], Color32)>,
    pub log_y: bool,
    pub min_height: f32,
}

impl<'a> LinePlot<'a> {
    pub fn show(self, ui: &mut Ui) {
        ui.label(self.title);
        let avail = ui.available_width().max(200.0);
        let (rect, _) = ui.allocate_exact_size(Vec2::new(avail, self.min_height), Sense::hover());
        let painter = ui.painter_at(rect);
        painter.rect_stroke(rect, 4.0, Stroke::new(1.0, ui.style().visuals.widgets.noninteractive.fg_stroke.color));

        let mut max_len = 0usize;
        let mut y_min = f32::INFINITY;
        let mut y_max = f32::NEG_INFINITY;
        for (_, ys, _) in &self.series {
            if ys.is_empty() { continue; }
            max_len = max_len.max(ys.len());
            for &v in *ys {
                let v = if self.log_y { v.max(1e-12).ln() } else { v };
                if v.is_finite() {
                    if v < y_min { y_min = v; }
                    if v > y_max { y_max = v; }
                }
            }
        }
        if max_len == 0 || !y_min.is_finite() || !y_max.is_finite() {
            painter.text(rect.center(), egui::Align2::CENTER_CENTER, "(no data yet)",
                egui::FontId::proportional(12.0), ui.style().visuals.weak_text_color());
            return;
        }
        if (y_max - y_min).abs() < 1e-9 { y_max = y_min + 1.0; }
        let pad_x = 32.0;
        let pad_y = 20.0;
        let inner = Rect::from_min_size(
            rect.min + Vec2::new(pad_x, pad_y),
            rect.size() - Vec2::new(pad_x + 8.0, pad_y + 24.0),
        );

        let axis = ui.style().visuals.widgets.noninteractive.fg_stroke.color.gamma_multiply(0.4);
        painter.line_segment([inner.left_bottom(), inner.right_bottom()], Stroke::new(1.0, axis));
        painter.line_segment([inner.left_top(), inner.left_bottom()], Stroke::new(1.0, axis));

        let mid = 0.5 * (y_min + y_max);
        let to_label = |v: f32| -> String {
            let actual = if self.log_y { v.exp() } else { v };
            if actual.abs() >= 1000.0 || (actual.abs() < 0.001 && actual != 0.0) {
                format!("{actual:.2e}")
            } else {
                format!("{actual:.3}")
            }
        };
        let font = egui::FontId::proportional(10.0);
        let text_color = ui.style().visuals.weak_text_color();
        painter.text(Pos2::new(rect.left() + 4.0, inner.top() - 2.0),
            egui::Align2::LEFT_TOP, to_label(y_max), font.clone(), text_color);
        painter.text(Pos2::new(rect.left() + 4.0, inner.center().y - 6.0),
            egui::Align2::LEFT_TOP, to_label(mid), font.clone(), text_color);
        painter.text(Pos2::new(rect.left() + 4.0, inner.bottom() - 12.0),
            egui::Align2::LEFT_TOP, to_label(y_min), font.clone(), text_color);

        for (name, ys, color) in &self.series {
            if ys.is_empty() { continue; }
            let n = ys.len();
            let mut points = Vec::with_capacity(n);
            for (i, &v) in ys.iter().enumerate() {
                let v = if self.log_y { v.max(1e-12).ln() } else { v };
                let x = inner.left() + inner.width() * (i as f32 / (max_len - 1).max(1) as f32);
                let y = inner.bottom() - inner.height() * ((v - y_min) / (y_max - y_min));
                points.push(Pos2::new(x, y));
            }
            painter.add(egui::Shape::line(points, PathStroke::new(1.5, *color)));
            let _ = name;
        }
        let mut legend_x = inner.right() - 8.0;
        let legend_y = rect.bottom() - 12.0;
        for (name, _ys, color) in self.series.iter().rev() {
            let label = format!("● {name}");
            let glyph_w = label.len() as f32 * 5.5;
            painter.text(Pos2::new(legend_x, legend_y), egui::Align2::RIGHT_CENTER,
                label, egui::FontId::proportional(10.0), *color);
            legend_x -= glyph_w + 12.0;
        }
    }
}

const PALETTE: [Color32; 6] = [
    Color32::from_rgb(231, 76,  60),
    Color32::from_rgb( 46, 204, 113),
    Color32::from_rgb( 52, 152, 219),
    Color32::from_rgb(241, 196,  15),
    Color32::from_rgb(155,  89, 182),
    Color32::from_rgb( 26, 188, 156),
];

/// 2-D classification scatter. Each point is `(x, y, class_index)`.
pub fn scatter_2d(ui: &mut Ui, points: &[(f32, f32, usize)], min_height: f32) {
    scatter_2d_inner(ui, points, None, min_height);
}

/// Same as `scatter_2d` but draws an extra highlighted query point
/// (the current inference inputs) as a large crosshair + circle.
///
/// `query_class` is the argmax of the model output — used to tint the marker
/// with the predicted class colour; pass `None` if no prediction has run yet.
pub fn scatter_2d_with_query(
    ui: &mut Ui,
    points: &[(f32, f32, usize)],
    query: (f32, f32),
    query_class: Option<usize>,
    min_height: f32,
) {
    scatter_2d_inner(ui, points, Some((query, query_class)), min_height);
}

fn scatter_2d_inner(
    ui: &mut Ui,
    points: &[(f32, f32, usize)],
    query: Option<((f32, f32), Option<usize>)>,
    min_height: f32,
) {
    let avail = ui.available_width().max(200.0);
    let (rect, _) = ui.allocate_exact_size(Vec2::new(avail, min_height), Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_stroke(rect, 4.0,
        Stroke::new(1.0, ui.style().visuals.widgets.noninteractive.fg_stroke.color));

    // Compute bounds from corpus points + query point together so the query
    // never falls off the edge of the plot.
    let mut x_min = f32::INFINITY;
    let mut x_max = f32::NEG_INFINITY;
    let mut y_min = f32::INFINITY;
    let mut y_max = f32::NEG_INFINITY;
    let extend = |x: f32, y: f32,
                  xn: &mut f32, xx: &mut f32,
                  yn: &mut f32, yx: &mut f32| {
        if x < *xn { *xn = x; }
        if x > *xx { *xx = x; }
        if y < *yn { *yn = y; }
        if y > *yx { *yx = y; }
    };
    for (x, y, _) in points { extend(*x, *y, &mut x_min, &mut x_max, &mut y_min, &mut y_max); }
    if let Some(((qx, qy), _)) = query {
        extend(qx, qy, &mut x_min, &mut x_max, &mut y_min, &mut y_max);
    }

    if points.is_empty() && query.is_none() {
        painter.text(rect.center(), egui::Align2::CENTER_CENTER, "(no points)",
            egui::FontId::proportional(12.0), ui.style().visuals.weak_text_color());
        return;
    }

    // Give a bit of margin so border points aren't clipped.
    let margin_x = ((x_max - x_min) * 0.08).max(0.1);
    let margin_y = ((y_max - y_min) * 0.08).max(0.1);
    x_min -= margin_x; x_max += margin_x;
    y_min -= margin_y; y_max += margin_y;

    if (x_max - x_min).abs() < 1e-9 { x_max = x_min + 1.0; }
    if (y_max - y_min).abs() < 1e-9 { y_max = y_min + 1.0; }

    let pad = 12.0;
    let inner = Rect::from_min_size(rect.min + Vec2::splat(pad), rect.size() - Vec2::splat(pad * 2.0));

    let to_px = |x: f32, y: f32| -> Pos2 {
        Pos2::new(
            inner.left() + inner.width()  * ((x - x_min) / (x_max - x_min)),
            inner.bottom() - inner.height() * ((y - y_min) / (y_max - y_min)),
        )
    };

    // Corpus points.
    for (x, y, c) in points {
        painter.circle_filled(to_px(*x, *y), 2.5, PALETTE[c % PALETTE.len()]);
    }

    // Query marker: crosshair + filled ring.
    if let Some(((qx, qy), qclass)) = query {
        let p = to_px(qx, qy);
        let arm = 8.0;
        let ring_r = 6.0;
        let col = qclass
            .map(|c| PALETTE[c % PALETTE.len()])
            .unwrap_or(Color32::WHITE);
        // Crosshair.
        painter.line_segment(
            [Pos2::new(p.x - arm, p.y), Pos2::new(p.x + arm, p.y)],
            Stroke::new(1.5, col),
        );
        painter.line_segment(
            [Pos2::new(p.x, p.y - arm), Pos2::new(p.x, p.y + arm)],
            Stroke::new(1.5, col),
        );
        // Outer ring.
        painter.circle_stroke(p, ring_r, Stroke::new(1.5, col));
        // Small filled centre dot.
        painter.circle_filled(p, 2.0, col);
    }
}
