//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
use image::{Rgb, RgbImage};
use imageproc::drawing::draw_filled_rect_mut;

use crate::base::game::RgbBuffer;
use crate::base::render::{draw_text_centered, fit_to_resolution, luminance, rect};

use super::state::{Orientation, RushHourState};

const EMPTY_COLOR: Rgb<u8> = Rgb([0xD0, 0xD0, 0xD0]);
const GRID_LINE_COLOR: Rgb<u8> = Rgb([0xB0, 0xB0, 0xB0]);

const VEHICLE_COLORS: [Rgb<u8>; 12] = [
    Rgb([0xDD, 0x22, 0x22]), // A = red (target car)
    Rgb([0x22, 0x77, 0xDD]), // B = blue
    Rgb([0x22, 0xAA, 0x22]), // C = green
    Rgb([0xFF, 0xA5, 0x00]), // D = orange
    Rgb([0x88, 0x22, 0xCC]), // E = purple
    Rgb([0x00, 0xBB, 0xBB]), // F = teal
    Rgb([0xDD, 0xDD, 0x22]), // G = yellow
    Rgb([0xDD, 0x66, 0xAA]), // H = pink
    Rgb([0x88, 0x66, 0x33]), // I = brown
    Rgb([0x66, 0xBB, 0x66]), // J = light green
    Rgb([0x99, 0x99, 0xDD]), // K = lavender
    Rgb([0xDD, 0x88, 0x44]), // L = light orange
];

fn vehicle_color(id: char) -> Rgb<u8> {
    let idx = (id as u8).wrapping_sub(b'A') as usize;
    VEHICLE_COLORS[idx % VEHICLE_COLORS.len()]
}

pub fn render(state: &RushHourState, resolution: u32) -> RgbBuffer {
    let gs = state.config.grid_size as u32;
    // Shrink grid to leave room for border + exit gap
    let border = (resolution / 40).max(2);
    let inner_res = resolution - 2 * border;
    let (raw_offset_x, raw_offset_y, _, cell_size) = fit_to_resolution(gs, gs, inner_res);
    let offset_x = raw_offset_x + border;
    let offset_y = raw_offset_y + border;

    let mut img = RgbImage::from_pixel(resolution, resolution, EMPTY_COLOR);

    // Draw thick border around the grid with a gap at the exit row
    let border_color = Rgb([0x44, 0x44, 0x44]);
    let exit_row = state.config.exit_row as u32;
    let exit_top = offset_y + exit_row * cell_size;
    let exit_bot = exit_top + cell_size;
    let grid_left = offset_x - border;
    let grid_right = offset_x + gs * cell_size;
    let grid_top = offset_y - border;
    let grid_bot = offset_y + gs * cell_size;
    let full_w = gs * cell_size + 2 * border;

    // Top border
    draw_filled_rect_mut(
        &mut img,
        rect(grid_left, grid_top, full_w, border),
        border_color,
    );
    // Bottom border
    draw_filled_rect_mut(
        &mut img,
        rect(grid_left, grid_bot, full_w, border),
        border_color,
    );
    // Left border
    draw_filled_rect_mut(
        &mut img,
        rect(grid_left, grid_top, border, grid_bot - grid_top + border),
        border_color,
    );
    // Right border — with gap at exit row
    if exit_top > offset_y {
        draw_filled_rect_mut(
            &mut img,
            rect(grid_right, grid_top, border, exit_top - grid_top),
            border_color,
        );
    }
    if exit_bot < grid_bot {
        draw_filled_rect_mut(
            &mut img,
            rect(grid_right, exit_bot, border, grid_bot + border - exit_bot),
            border_color,
        );
    }

    // Draw grid lines
    for i in 0..=gs {
        let x = offset_x + i * cell_size;
        let y = offset_y + i * cell_size;
        if y < resolution {
            draw_filled_rect_mut(
                &mut img,
                rect(offset_x, y, gs * cell_size, 1),
                GRID_LINE_COLOR,
            );
        }
        if x < resolution {
            draw_filled_rect_mut(
                &mut img,
                rect(x, offset_y, 1, gs * cell_size),
                GRID_LINE_COLOR,
            );
        }
    }

    // Draw vehicles as connected rectangles
    let margin = (cell_size / 8).max(1);
    for (idx, def) in state.config.vehicles.iter().enumerate() {
        let pos = state.positions[idx] as u32;
        let color = vehicle_color(def.id);

        let (vx, vy, vw, vh) = match def.orientation {
            Orientation::Horizontal => (
                offset_x + pos * cell_size + margin,
                offset_y + def.fixed_axis as u32 * cell_size + margin,
                def.length as u32 * cell_size - 2 * margin,
                cell_size - 2 * margin,
            ),
            Orientation::Vertical => (
                offset_x + def.fixed_axis as u32 * cell_size + margin,
                offset_y + pos * cell_size + margin,
                cell_size - 2 * margin,
                def.length as u32 * cell_size - 2 * margin,
            ),
        };

        draw_filled_rect_mut(&mut img, rect(vx, vy, vw, vh), color);

        // Draw vehicle ID label centered on the vehicle
        let cx = (vx + vw / 2) as i32;
        let cy = (vy + vh / 2) as i32;
        let font_h = (cell_size as f32 * 0.6).max(8.0);
        let text_color = if luminance(&color) > 128.0 {
            Rgb([0, 0, 0])
        } else {
            Rgb([255, 255, 255])
        };
        draw_text_centered(&mut img, cx, cy, def.id, font_h, text_color);
    }

    RgbBuffer {
        width: resolution,
        height: resolution,
        data: img.into_raw(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::games::rush_hour::state::from_ascii;

    #[test]
    fn render_dimensions() {
        let state = from_ascii("..BCCC\n..B...\nAAB..D\n.EE..D\n......\n......");
        let buf = render(&state, 64);
        assert_eq!(buf.width, 64);
        assert_eq!(buf.height, 64);
        assert_eq!(buf.data.len(), 64 * 64 * 3);
    }

    #[test]
    fn render_nonzero() {
        let state = from_ascii("..BCCC\n..B...\nAAB..D\n.EE..D\n......\n......");
        let buf = render(&state, 64);
        assert!(buf.data.iter().any(|&b| b != 0));
    }
}
