//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
use image::{Rgb, RgbImage};
use imageproc::drawing::draw_filled_rect_mut;

use crate::base::game::RgbBuffer;
use crate::base::render::{draw_str_centered, draw_text_centered, luminance, rect};

use super::{PieceDef, TangramState};

const TANGRAM_BG: Rgb<u8> = Rgb([0xF0, 0xF0, 0xF0]);
const TANGRAM_TARGET: Rgb<u8> = Rgb([0xC0, 0xC0, 0xC0]);
const TANGRAM_GRID: Rgb<u8> = Rgb([0x80, 0x80, 0x80]);
const TANGRAM_LABEL: Rgb<u8> = Rgb([0x60, 0x60, 0x60]);
const TANGRAM_PIECE_COLORS: [Rgb<u8>; 12] = [
    Rgb([0x44, 0x44, 0xFF]),
    Rgb([0xFF, 0x44, 0x44]),
    Rgb([0x44, 0xCC, 0x44]),
    Rgb([0xFF, 0xCC, 0x00]),
    Rgb([0x88, 0x00, 0xFF]),
    Rgb([0xFF, 0x88, 0x00]),
    Rgb([0x00, 0xCC, 0xCC]),
    Rgb([0xCC, 0x00, 0x88]),
    Rgb([0x88, 0x88, 0x00]),
    Rgb([0x00, 0x44, 0x88]),
    Rgb([0x88, 0x44, 0x00]),
    Rgb([0x44, 0x88, 0x00]),
];

fn piece_color(piece_id: &str, color_map: &[usize]) -> Rgb<u8> {
    let idx: usize = piece_id
        .strip_prefix('P')
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let palette_idx = if idx < color_map.len() {
        color_map[idx]
    } else {
        idx
    };
    TANGRAM_PIECE_COLORS[palette_idx % TANGRAM_PIECE_COLORS.len()]
}

fn piece_label(piece_id: &str) -> char {
    let idx: usize = piece_id
        .strip_prefix('P')
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    if idx < 10 {
        (b'0' + idx as u8) as char
    } else {
        (b'A' + (idx - 10) as u8) as char
    }
}

/// Compute tray piece positions once at puzzle creation time.
/// Returns `(cell_size, positions)` where positions[i] = (x, y) for piece i.
pub fn compute_tray_layout(
    board_w: u32,
    board_h: u32,
    pieces: &[PieceDef],
    queue_rotations: &[(String, i32)],
    resolution: u32,
) -> (u32, Vec<(u32, u32)>) {
    let total_piece_count = pieces.len() as u32;
    if total_piece_count == 0 {
        return (0, vec![]);
    }

    let margin = (resolution as f32 * 0.02) as u32;
    let label_font_height = (resolution as f32 * 0.04).max(8.0);
    let label_gap = (label_font_height * 1.2) as u32;

    let all_display_cells: Vec<Vec<(i32, i32)>> = pieces
        .iter()
        .map(|p| {
            let rotation = queue_rotations
                .iter()
                .find(|(id, _)| *id == p.id)
                .map(|(_, r)| *r)
                .unwrap_or(0);
            p.rotated_cells(rotation)
        })
        .collect();

    let max_piece_dim = all_display_cells
        .iter()
        .map(|cells| {
            let min_x = cells.iter().map(|c| c.0).min().unwrap_or(0);
            let max_x = cells.iter().map(|c| c.0).max().unwrap_or(0);
            let min_y = cells.iter().map(|c| c.1).min().unwrap_or(0);
            let max_y = cells.iter().map(|c| c.1).max().unwrap_or(0);
            ((max_x - min_x + 1) as u32).max((max_y - min_y + 1) as u32)
        })
        .max()
        .unwrap_or(1);

    let tray_cells = max_piece_dim + 1;
    let avail_w = resolution - 2 * margin;
    let avail_h = resolution - 2 * margin;
    let h_cells = board_w + tray_cells;
    let v_cells = board_h + tray_cells;

    let cs_h = if h_cells > 0 {
        (avail_w.saturating_sub(label_gap)) / h_cells
    } else {
        avail_w
    };
    let cs_v = if v_cells > 0 {
        (avail_h.saturating_sub(label_gap)) / v_cells
    } else {
        avail_h
    };
    let base_cell_size = ((cs_h.min(cs_v) as f32 * 0.9) as u32).max(1);

    let mut cell_size = base_cell_size;
    let mut final_positions: Vec<(u32, u32)> = vec![(0, 0); pieces.len()];

    for _attempt in 0..5 {
        let label_font_h = (cell_size as f32 * 0.75).max(10.0);
        let label_extra = label_font_h as u32 + 4;
        let scan_step = cell_size.max(1);

        let bp_w = board_w * cell_size;
        let bp_h = board_h * cell_size;
        let bx = margin + label_gap;
        let by = margin + label_gap;
        let board_zone_x = bx.saturating_sub(label_gap);
        let board_zone_y = by.saturating_sub(label_gap);
        let board_zone_w = bp_w + label_gap;
        let board_zone_h = bp_h + label_gap;

        let dims: Vec<(u32, u32)> = all_display_cells
            .iter()
            .map(|cells| {
                let min_x = cells.iter().map(|c| c.0).min().unwrap_or(0);
                let max_x = cells.iter().map(|c| c.0).max().unwrap_or(0);
                let min_y = cells.iter().map(|c| c.1).min().unwrap_or(0);
                let max_y = cells.iter().map(|c| c.1).max().unwrap_or(0);
                let pw = (max_x - min_x + 1) as u32 * cell_size + cell_size / 2;
                let ph = (max_y - min_y + 1) as u32 * cell_size + label_extra + cell_size / 2;
                (pw, ph)
            })
            .collect();

        let mut order: Vec<usize> = (0..pieces.len()).collect();
        order.sort_by(|&a, &b| {
            let area_a = dims[a].0 * dims[a].1;
            let area_b = dims[b].0 * dims[b].1;
            area_b.cmp(&area_a)
        });

        let mut occupied: Vec<(u32, u32, u32, u32)> =
            vec![(board_zone_x, board_zone_y, board_zone_w, board_zone_h)];
        let mut positions: Vec<(u32, u32)> = vec![(0, 0); pieces.len()];
        let mut all_placed = true;

        for &idx in &order {
            let (pw, ph) = dims[idx];
            let mut placed = false;

            'scan: for y in
                (margin..resolution.saturating_sub(ph + margin)).step_by(scan_step as usize)
            {
                for x in
                    (margin..resolution.saturating_sub(pw + margin)).step_by(scan_step as usize)
                {
                    let mut overlaps = false;
                    for &(ox, oy, ow, oh) in &occupied {
                        if x < ox + ow && x + pw > ox && y < oy + oh && y + ph > oy {
                            overlaps = true;
                            break;
                        }
                    }
                    if !overlaps {
                        positions[idx] = (x, y);
                        occupied.push((x, y, pw, ph));
                        placed = true;
                        break 'scan;
                    }
                }
            }

            if !placed {
                all_placed = false;
                break;
            }
        }

        final_positions = positions;
        if all_placed {
            break;
        }

        cell_size = ((cell_size as f32 * 0.9) as u32).max(1);
    }

    (cell_size, final_positions)
}

/// Render a tangram state to an RgbBuffer at the given resolution.
pub fn render(state: &TangramState, resolution: u32) -> RgbBuffer {
    let board_w = state.puzzle.canvas_width as u32;
    let board_h = state.puzzle.canvas_height as u32;

    let mut img = RgbImage::from_pixel(resolution, resolution, TANGRAM_BG);

    let margin = (resolution as f32 * 0.02) as u32;
    let label_font_height = (resolution as f32 * 0.04).max(8.0);
    let label_gap = (label_font_height * 1.2) as u32;

    // Compute piece data early for unified cell size calculation
    let all_pieces = &state.puzzle.pieces;
    let total_piece_count = all_pieces.len() as u32;

    let all_display_cells: Vec<Vec<(i32, i32)>> = all_pieces
        .iter()
        .map(|p| {
            let rotation = state.queue_rotation(&p.id);
            p.rotated_cells(rotation)
        })
        .collect();

    // Tray layout of the first render, reused afterwards.
    let (_layout_resolution, cell_size, tray_positions) =
        state.puzzle.tray_layout.get_or_init(|| {
            let (cell_size, positions) = compute_tray_layout(
                board_w,
                board_h,
                all_pieces,
                &state.queue_rotations,
                resolution,
            );
            (resolution, cell_size, positions)
        });
    let cell_size = *cell_size;

    // Pixel dimensions using final cell_size
    let board_pixel_w = board_w * cell_size;
    let board_pixel_h = board_h * cell_size;

    // Position the layout: top-left aligned
    let board_x = margin + label_gap;
    let board_y = margin + label_gap;

    // Draw board background (target silhouette)
    draw_filled_rect_mut(
        &mut img,
        rect(board_x, board_y, board_pixel_w, board_pixel_h),
        TANGRAM_TARGET,
    );

    // Draw placed pieces
    for placement in &state.placed {
        let color = piece_color(&placement.piece_id, &state.puzzle.color_map);
        let label = piece_label(&placement.piece_id);

        if let Some(piece) = state
            .puzzle
            .pieces
            .iter()
            .find(|p| p.id == placement.piece_id)
        {
            let cells = placement.occupied_cells(piece);

            for &(cx, cy) in &cells {
                if cx >= 0 && cy >= 0 {
                    let px = board_x + cx as u32 * cell_size;
                    let py = board_y + cy as u32 * cell_size;
                    draw_filled_rect_mut(&mut img, rect(px, py, cell_size, cell_size), color);
                }
            }

            // Label centered on bounding box
            if !cells.is_empty() {
                let min_x = cells.iter().map(|c| c.0).min().unwrap();
                let max_x = cells.iter().map(|c| c.0).max().unwrap();
                let min_y = cells.iter().map(|c| c.1).min().unwrap();
                let max_y = cells.iter().map(|c| c.1).max().unwrap();
                let center_x = board_x as i32 + (min_x + max_x + 1) * cell_size as i32 / 2;
                let center_y = board_y as i32 + (min_y + max_y + 1) * cell_size as i32 / 2;
                let font_height = (cell_size as f32 * 0.75).max(10.0);
                let text_color = if luminance(&color) > 128.0 {
                    Rgb([0, 0, 0])
                } else {
                    Rgb([255, 255, 255])
                };
                draw_text_centered(&mut img, center_x, center_y, label, font_height, text_color);
            }

            // Anchor point marker (X drawn on the anchor cell)
            let ax = board_x + placement.x as u32 * cell_size;
            let ay = board_y + placement.y as u32 * cell_size;
            let anchor_color = if luminance(&color) > 128.0 {
                Rgb([0, 0, 0])
            } else {
                Rgb([255, 255, 255])
            };
            let cs = cell_size as i32;
            let anchor_margin = cs / 5;
            for d in anchor_margin..cs - anchor_margin {
                let ix = ax as i32 + d;
                let iy = ay as i32 + d;
                if ix >= 0 && iy >= 0 && (ix as u32) < resolution && (iy as u32) < resolution {
                    img.put_pixel(ix as u32, iy as u32, anchor_color);
                }
                let ix2 = ax as i32 + cs - 1 - d;
                if ix2 >= 0 && iy >= 0 && (ix2 as u32) < resolution && (iy as u32) < resolution {
                    img.put_pixel(ix2 as u32, iy as u32, anchor_color);
                }
            }
        }
    }

    // Grid lines over the board
    for i in 0..=board_w {
        let gx = board_x + i * cell_size;
        draw_filled_rect_mut(&mut img, rect(gx, board_y, 1, board_pixel_h), TANGRAM_GRID);
    }
    for i in 0..=board_h {
        let gy = board_y + i * cell_size;
        draw_filled_rect_mut(&mut img, rect(board_x, gy, board_pixel_w, 1), TANGRAM_GRID);
    }

    // Axis labels
    for i in 0..board_w {
        let cx = board_x as i32 + i as i32 * cell_size as i32 + cell_size as i32 / 2;
        let cy = board_y as i32 - label_gap as i32 / 2;
        let label = i.to_string();
        draw_str_centered(&mut img, cx, cy, &label, label_font_height, TANGRAM_LABEL);
    }
    for i in 0..board_h {
        let cx = board_x as i32 - label_gap as i32 / 2;
        let cy = board_y as i32 + i as i32 * cell_size as i32 + cell_size as i32 / 2;
        let label = i.to_string();
        draw_str_centered(&mut img, cx, cy, &label, label_font_height, TANGRAM_LABEL);
    }

    // Render unplaced pieces at their cached tray positions
    if total_piece_count > 0 {
        for (slot_idx, piece) in all_pieces.iter().enumerate() {
            let is_remaining = state.remaining_piece_ids.contains(&piece.id);
            if !is_remaining {
                continue;
            }

            if slot_idx >= tray_positions.len() {
                continue;
            }
            let (piece_px, piece_py) = tray_positions[slot_idx];

            let color = piece_color(&piece.id, &state.puzzle.color_map);
            let label = piece_label(&piece.id);
            let rotated_cells = &all_display_cells[slot_idx];

            let min_x = rotated_cells.iter().map(|c| c.0).min().unwrap_or(0);
            let min_y = rotated_cells.iter().map(|c| c.1).min().unwrap_or(0);
            let norm_cells: Vec<(i32, i32)> = rotated_cells
                .iter()
                .map(|&(x, y)| (x - min_x, y - min_y))
                .collect();

            let piece_w = norm_cells.iter().map(|c| c.0).max().unwrap_or(0) as u32 + 1;
            let piece_h = norm_cells.iter().map(|c| c.1).max().unwrap_or(0) as u32 + 1;

            // Draw piece cells
            for &(cx, cy) in &norm_cells {
                let px = piece_px + cx as u32 * cell_size;
                let py = piece_py + cy as u32 * cell_size;
                draw_filled_rect_mut(&mut img, rect(px, py, cell_size, cell_size), color);
            }

            // Grid overlay on piece
            for gx in 0..=piece_w {
                let x = piece_px + gx * cell_size;
                let y_end = piece_py + piece_h * cell_size;
                if x < resolution {
                    draw_filled_rect_mut(
                        &mut img,
                        rect(
                            x,
                            piece_py,
                            1,
                            (y_end - piece_py).min(resolution - piece_py),
                        ),
                        TANGRAM_GRID,
                    );
                }
            }
            for gy in 0..=piece_h {
                let y = piece_py + gy * cell_size;
                let x_end = piece_px + piece_w * cell_size;
                if y < resolution {
                    draw_filled_rect_mut(
                        &mut img,
                        rect(
                            piece_px,
                            y,
                            (x_end - piece_px).min(resolution - piece_px),
                            1,
                        ),
                        TANGRAM_GRID,
                    );
                }
            }

            // Label below piece
            let center_x = piece_px as i32 + (piece_w * cell_size) as i32 / 2;
            let font_height = (cell_size as f32 * 0.75).max(10.0);
            let below_y =
                piece_py as i32 + (piece_h * cell_size) as i32 + font_height as i32 / 2 + 2;
            draw_text_centered(
                &mut img,
                center_x,
                below_y,
                label,
                font_height,
                TANGRAM_LABEL,
            );

            // Anchor point marker
            let anchor_rot_x = (0 - min_x) as u32;
            let anchor_rot_y = (0 - min_y) as u32;
            let ax = piece_px + anchor_rot_x * cell_size;
            let ay = piece_py + anchor_rot_y * cell_size;
            let anchor_color = if luminance(&color) > 128.0 {
                Rgb([0, 0, 0])
            } else {
                Rgb([255, 255, 255])
            };
            let cs = cell_size as i32;
            let anchor_margin = cs / 5;
            for d in anchor_margin..cs - anchor_margin {
                let ix = ax as i32 + d;
                let iy = ay as i32 + d;
                if ix >= 0 && iy >= 0 && (ix as u32) < resolution && (iy as u32) < resolution {
                    img.put_pixel(ix as u32, iy as u32, anchor_color);
                }
                let ix2 = ax as i32 + cs - 1 - d;
                if ix2 >= 0 && iy >= 0 && (ix2 as u32) < resolution && (iy as u32) < resolution {
                    img.put_pixel(ix2 as u32, iy as u32, anchor_color);
                }
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
    use super::super::*;
    use std::collections::HashSet;
    use std::sync::Arc;

    fn make_test_state() -> TangramState {
        let target_set: HashSet<(i32, i32)> = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)]
            .iter()
            .copied()
            .collect();

        let puzzle = Arc::new(TangramPuzzle {
            canvas_width: 3,
            canvas_height: 2,
            pieces: vec![
                PieceDef {
                    id: "P0".to_string(),
                    cells: vec![(0, 0), (1, 0)],
                },
                PieceDef {
                    id: "P1".to_string(),
                    cells: vec![(0, 0), (1, 0)],
                },
                PieceDef {
                    id: "P2".to_string(),
                    cells: vec![(0, 0), (1, 0)],
                },
            ],
            target_set,
            color_map: vec![0, 1, 2],
            target_placements: vec![],
            solvable: true,
            tray_layout: std::sync::OnceLock::new(),
        });

        TangramState {
            puzzle,
            placed: vec![],
            remaining_piece_ids: vec!["P0".to_string(), "P1".to_string(), "P2".to_string()],
            queue_rotations: vec![
                ("P0".to_string(), 0),
                ("P1".to_string(), 0),
                ("P2".to_string(), 0),
            ],
            occupied_cells: HashSet::new(),
        }
    }

    #[test]
    fn render_dimensions() {
        let state = make_test_state();
        let buf = super::render(&state, 64);
        assert_eq!(buf.width, 64);
        assert_eq!(buf.height, 64);
        assert_eq!(buf.data.len(), 64 * 64 * 3);
    }

    #[test]
    fn render_nonzero() {
        let state = make_test_state();
        let buf = super::render(&state, 64);
        assert!(buf.data.iter().any(|&b| b != 0));
    }
}
