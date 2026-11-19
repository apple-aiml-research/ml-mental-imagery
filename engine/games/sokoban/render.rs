//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
use image::{Rgb, RgbImage};
use imageproc::drawing::{draw_filled_circle_mut, draw_filled_rect_mut};

use crate::base::game::RgbBuffer;
use crate::base::render::{fill_diamond, fit_to_resolution, rect};

use super::state::{Cell, SokobanState};

const WALL_COLOR: Rgb<u8> = Rgb([0x40, 0x40, 0x40]);
const FLOOR_COLOR: Rgb<u8> = Rgb([0xD0, 0xD0, 0xD0]);
const PLAYER_COLOR: Rgb<u8> = Rgb([0x44, 0x44, 0xFF]);
const BOX_COLOR: Rgb<u8> = Rgb([0x8B, 0x45, 0x13]);
const GOAL_COLOR: Rgb<u8> = Rgb([0xFF, 0x44, 0x44]);
const BOX_ON_GOAL_COLOR: Rgb<u8> = Rgb([0x44, 0xAA, 0x44]);
const PLAYER_ON_GOAL_COLOR: Rgb<u8> = Rgb([0x44, 0xAA, 0xAA]);

/// Render a Sokoban state to an RgbBuffer at the given resolution.
pub fn render(state: &SokobanState, resolution: u32) -> RgbBuffer {
    let board_w = state.layout.width as u32;
    let board_h = state.layout.height as u32;
    let (offset_x, offset_y, _, cell_size) = fit_to_resolution(board_w, board_h, resolution);

    let mut img = RgbImage::from_pixel(resolution, resolution, FLOOR_COLOR);

    for row in 0..state.layout.height {
        for col in 0..state.layout.width {
            let cx = offset_x + col as u32 * cell_size;
            let cy = offset_y + row as u32 * cell_size;
            let cell = state.layout.cell_at(col, row);

            let bg = if cell == Cell::Wall {
                WALL_COLOR
            } else {
                FLOOR_COLOR
            };
            draw_filled_rect_mut(&mut img, rect(cx, cy, cell_size, cell_size), bg);

            let is_player = col == state.player_col && row == state.player_row;
            let is_box = state.has_box(col, row);
            let is_goal = state.layout.is_goal(col, row);

            let margin = cell_size / 6;
            let inner_size = cell_size.saturating_sub(2 * margin);
            let center_x = cx as i32 + cell_size as i32 / 2;
            let center_y = cy as i32 + cell_size as i32 / 2;

            if is_player && is_goal {
                let r = inner_size as i32 / 2;
                draw_filled_circle_mut(&mut img, (center_x, center_y), r, PLAYER_ON_GOAL_COLOR);
            } else if is_player {
                let r = inner_size as i32 / 2;
                draw_filled_circle_mut(&mut img, (center_x, center_y), r, PLAYER_COLOR);
            } else if is_box && is_goal {
                draw_filled_rect_mut(
                    &mut img,
                    rect(cx + margin, cy + margin, inner_size, inner_size),
                    BOX_ON_GOAL_COLOR,
                );
            } else if is_box {
                draw_filled_rect_mut(
                    &mut img,
                    rect(cx + margin, cy + margin, inner_size, inner_size),
                    BOX_COLOR,
                );
            } else if is_goal {
                let half = inner_size / 2;
                fill_diamond(
                    &mut img,
                    cx + cell_size / 2,
                    cy + cell_size / 2,
                    half,
                    GOAL_COLOR,
                );
            }
        }
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
    use crate::games::sokoban::state::from_ascii;

    #[test]
    fn render_dimensions() {
        let state = from_ascii("####\n#@X#\n#O.#\n####");
        let buf = render(&state, 64);
        assert_eq!(buf.width, 64);
        assert_eq!(buf.height, 64);
        assert_eq!(buf.data.len(), 64 * 64 * 3);
    }

    #[test]
    fn render_nonzero() {
        let state = from_ascii("####\n#@X#\n#O.#\n####");
        let buf = render(&state, 64);
        assert!(buf.data.iter().any(|&b| b != 0));
    }

    #[test]
    fn test_render_square_multiple_resolutions() {
        let state = from_ascii("####\n#@X#\n#O.#\n####");
        for res in [64, 128, 256] {
            let buf = render(&state, res);
            assert_eq!(buf.width, res, "width mismatch at resolution {res}");
            assert_eq!(buf.height, res, "height mismatch at resolution {res}");
            assert_eq!(
                buf.data.len(),
                (res * res * 3) as usize,
                "data length mismatch at resolution {res}"
            );
        }
    }
}
