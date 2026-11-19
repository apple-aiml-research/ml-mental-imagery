//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
use image::{Rgb, RgbImage};

use crate::base::game::RgbBuffer;
use crate::base::render::draw_str_centered;

use super::state::{cell_label, EdgeType, JigsawState};

// ---------------------------------------------------------------------------
// Colours
// ---------------------------------------------------------------------------

const BG_COLOR: Rgb<u8> = Rgb([0xF0, 0xF0, 0xF0]);
const EMPTY_CELL: Rgb<u8> = Rgb([0xD0, 0xD0, 0xD0]);
const GRID_LINE: Rgb<u8> = Rgb([0x80, 0x80, 0x80]);
const TRAY_BG: Rgb<u8> = Rgb([0xE0, 0xE0, 0xE0]);

// ---------------------------------------------------------------------------
// Layout — greedy packing (cached via OnceLock on JigsawPuzzle)
// ---------------------------------------------------------------------------

/// Greedily pack N² uniform slots into the area outside the grid.
/// Uses worst-case bounding boxes (tabs on all sides) so the layout is
/// independent of per-piece rotation state.
fn try_greedy_pack(n: usize, cell_size: u32, resolution: u32) -> Option<Vec<(u32, u32)>> {
    let tab_r = (cell_size as f32 * TAB_FRAC) as u32;
    let font_h = (cell_size as f32 * 0.3).clamp(8.0, 22.0);
    let label_h = font_h as u32 + 4;
    let pad = (cell_size / 8).max(2);
    let scan_step = (cell_size / 3).max(1);
    let grid_px = cell_size * n as u32;
    let margin = pad;

    let bw = tab_r + cell_size + tab_r;
    let bh = tab_r + cell_size + tab_r + label_h;

    let grid_zone = (0u32, 0u32, grid_px + pad, grid_px + pad);
    let mut occupied: Vec<(u32, u32, u32, u32)> = vec![grid_zone];
    let mut positions: Vec<(u32, u32)> = Vec::with_capacity(n * n);

    for _ in 0..(n * n) {
        let mut placed = false;

        'scan: for y in (margin..resolution.saturating_sub(bh)).step_by(scan_step as usize) {
            for x in (margin..resolution.saturating_sub(bw)).step_by(scan_step as usize) {
                let mut overlaps = false;
                for &(ox, oy, ow, oh) in &occupied {
                    if x < ox + ow && x + bw > ox && y < oy + oh && y + bh > oy {
                        overlaps = true;
                        break;
                    }
                }
                if !overlaps {
                    occupied.push((x, y, bw, bh));
                    positions.push((x + tab_r, y + tab_r));
                    placed = true;
                    break 'scan;
                }
            }
        }

        if !placed {
            return None;
        }
    }

    Some(positions)
}

/// Compute tray layout with retry. Shrinks cell_size by 10% on each failure.
pub fn compute_tray_layout(n: usize, resolution: u32) -> (u32, Vec<(u32, u32)>) {
    let base_cell_size = (resolution / (n as u32 + 1)).max(1);
    let mut cell_size = base_cell_size;

    for _attempt in 0..8 {
        if let Some(positions) = try_greedy_pack(n, cell_size, resolution) {
            return (cell_size, positions);
        }
        cell_size = ((cell_size as f32 * 0.9) as u32).max(1);
    }

    let positions =
        try_greedy_pack(n, cell_size, resolution).unwrap_or_else(|| vec![(0, 0); n * n]);
    (cell_size, positions)
}

// ---------------------------------------------------------------------------
// Jigsaw mask
// ---------------------------------------------------------------------------

/// Tab radius as a fraction of cell_size.
const TAB_FRAC: f32 = 0.18;

/// Check if pixel (dx, dy) relative to the cell origin is inside the jigsaw
/// shape defined by the four edges. Coordinates can be negative (tab above/left)
/// or >= cell_size (tab below/right).
fn in_jigsaw_shape(
    dx: i32,
    dy: i32,
    cs: i32,
    tab_r: i32,
    edges: &[EdgeType; 4], // [Top, Right, Bottom, Left]
) -> bool {
    let mid = cs / 2;

    // Start with base rectangle.
    let mut inside = dx >= 0 && dx < cs && dy >= 0 && dy < cs;

    // Top edge (y = 0, semicircle centred at (mid, 0))
    let top_dx = dx - mid;
    let top_dist2 = top_dx * top_dx + dy * dy;
    match edges[0] {
        EdgeType::Tab => {
            if dy < 0 && top_dist2 <= tab_r * tab_r && dx >= 0 && dx < cs {
                inside = true;
            }
        }
        EdgeType::Slot => {
            if dy >= 0 && top_dist2 <= tab_r * tab_r {
                inside = false;
            }
        }
        EdgeType::Flat => {}
    }

    // Bottom edge (y = cs, semicircle centred at (mid, cs))
    let bot_dy = dy - cs;
    let bot_dx = dx - mid;
    let bot_dist2 = bot_dx * bot_dx + bot_dy * bot_dy;
    match edges[2] {
        EdgeType::Tab => {
            if dy >= cs && bot_dist2 <= tab_r * tab_r && dx >= 0 && dx < cs {
                inside = true;
            }
        }
        EdgeType::Slot => {
            if dy < cs && bot_dist2 <= tab_r * tab_r {
                inside = false;
            }
        }
        EdgeType::Flat => {}
    }

    // Left edge (x = 0, semicircle centred at (0, mid))
    let left_dy = dy - mid;
    let left_dist2 = dx * dx + left_dy * left_dy;
    match edges[3] {
        EdgeType::Tab => {
            if dx < 0 && left_dist2 <= tab_r * tab_r && dy >= 0 && dy < cs {
                inside = true;
            }
        }
        EdgeType::Slot => {
            if dx >= 0 && left_dist2 <= tab_r * tab_r {
                inside = false;
            }
        }
        EdgeType::Flat => {}
    }

    // Right edge (x = cs, semicircle centred at (cs, mid))
    let right_dx = dx - cs;
    let right_dy = dy - mid;
    let right_dist2 = right_dx * right_dx + right_dy * right_dy;
    match edges[1] {
        EdgeType::Tab => {
            if dx >= cs && right_dist2 <= tab_r * tab_r && dy >= 0 && dy < cs {
                inside = true;
            }
        }
        EdgeType::Slot => {
            if dx < cs && right_dist2 <= tab_r * tab_r {
                inside = false;
            }
        }
        EdgeType::Flat => {}
    }

    inside
}

// ---------------------------------------------------------------------------
// Public render
// ---------------------------------------------------------------------------

pub fn render(state: &JigsawState, resolution: u32) -> RgbBuffer {
    let n = state.puzzle.grid_size;

    // Tray layout of the first render, reused afterwards.
    let (_layout_resolution, cell_size, tray_positions) =
        state.puzzle.tray_layout.get_or_init(|| {
            let (cell_size, positions) = compute_tray_layout(n, resolution);
            (resolution, cell_size, positions)
        });
    let cell_size = *cell_size;

    let tab_r = (cell_size as f32 * TAB_FRAC) as i32;
    let grid_px = cell_size * n as u32;
    let mut img = RgbImage::from_pixel(resolution, resolution, BG_COLOR);

    // Fill tray background (L-shaped area outside the grid)
    fill_rect(&mut img, grid_px, 0, resolution - grid_px, grid_px, TRAY_BG);
    fill_rect(&mut img, 0, grid_px, grid_px, resolution - grid_px, TRAY_BG);
    fill_rect(
        &mut img,
        grid_px,
        grid_px,
        resolution - grid_px,
        resolution - grid_px,
        TRAY_BG,
    );

    // --- Puzzle grid (top-left NxN) ---
    let grid_font_h = (cell_size as f32 * 0.25).clamp(8.0, 20.0);

    // Fill empty cells with plain squares (no jigsaw shape).
    for row in 0..n {
        for col in 0..n {
            let idx = row * n + col;
            if state.grid[idx].is_none() {
                let cx = col as u32 * cell_size;
                let cy = row as u32 * cell_size;
                fill_rect(&mut img, cx, cy, cell_size, cell_size, EMPTY_CELL);
                let label = cell_label(idx);
                let lcx = cx as i32 + cell_size as i32 / 2;
                let lcy = cy as i32 + cell_size as i32 / 2;
                draw_str_centered(
                    &mut img,
                    lcx,
                    lcy,
                    &label,
                    grid_font_h,
                    Rgb([0x60, 0x60, 0x60]),
                );
            }
        }
    }

    // Draw placed pieces (tabs overlay into adjacent cells).
    for row in 0..n {
        for col in 0..n {
            let idx = row * n + col;
            if let Some(piece_id) = state.grid[idx] {
                let cx = col as u32 * cell_size;
                let cy = row as u32 * cell_size;
                let edges = state.puzzle.edge_types(idx);
                let rotation = state.rotations[piece_id];
                blit_jigsaw_piece(
                    &mut img, cx, cy, cell_size, tab_r, &edges, state, piece_id, rotation,
                );
            }
        }
    }

    // Draw plain square grid outlines for all cells.
    for row in 0..n {
        for col in 0..n {
            let cx = col as u32 * cell_size;
            let cy = row as u32 * cell_size;
            draw_rect_outline(&mut img, cx, cy, cell_size, cell_size, GRID_LINE);
        }
    }

    // --- Tray (greedy-packed pieces) ---
    let font_h = (cell_size as f32 * 0.3).clamp(8.0, 22.0);
    let label_gap = font_h as u32 + 4;

    for (slot_idx, &(px, py)) in tray_positions.iter().enumerate() {
        if slot_idx >= state.puzzle.tray_order.len() {
            break;
        }
        let piece_id = state.puzzle.tray_order[slot_idx];
        if state.unplaced.contains(&piece_id) {
            let rotation = state.rotations[piece_id];
            let edges = state.puzzle.rotated_edge_types(piece_id, rotation);
            blit_jigsaw_piece(
                &mut img, px, py, cell_size, tab_r, &edges, state, piece_id, rotation,
            );
            draw_jigsaw_outline(
                &mut img,
                px,
                py,
                cell_size,
                tab_r,
                &edges,
                Rgb([0x40, 0x40, 0x40]),
            );
            let bottom_offset = match edges[2] {
                EdgeType::Tab => tab_r as u32,
                EdgeType::Flat => 0,
                EdgeType::Slot => 0,
            };
            let label = format!("{}", piece_id);
            let lcx = px as i32 + cell_size as i32 / 2;
            let lcy = (py + cell_size + bottom_offset) as i32 + label_gap as i32 / 2;
            draw_str_centered(&mut img, lcx, lcy, &label, font_h, Rgb([0x30, 0x30, 0x30]));
        } else {
            let label = format!("{}", piece_id);
            let lcx = px as i32 + cell_size as i32 / 2;
            let lcy = py as i32 + cell_size as i32 / 2;
            draw_str_centered(&mut img, lcx, lcy, &label, font_h, Rgb([0xC0, 0xC0, 0xC0]));
        }
    }

    RgbBuffer {
        width: resolution,
        height: resolution,
        data: img.into_raw(),
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Blit a piece image with jigsaw-shaped mask, applying rotation to the image.
fn blit_jigsaw_piece(
    img: &mut RgbImage,
    dst_x: u32,
    dst_y: u32,
    cell_size: u32,
    tab_r: i32,
    edges: &[EdgeType; 4],
    state: &JigsawState,
    piece_id: usize,
    rotation: i32,
) {
    let puzzle = &state.puzzle;
    let src_w = puzzle.piece_width as u64;
    let src_h = puzzle.piece_height as u64;
    let src = &puzzle.piece_images[piece_id];
    let (iw, ih) = (img.width(), img.height());
    let cs = cell_size as i32;
    let cs64 = cell_size as u64;
    let tr = tab_r;
    let rot = ((rotation % 360) + 360) % 360;

    for dy_i in -tr..(cs + tr) {
        for dx_i in -tr..(cs + tr) {
            if !in_jigsaw_shape(dx_i, dy_i, cs, tr, edges) {
                continue;
            }
            let px = dst_x as i32 + dx_i;
            let py = dst_y as i32 + dy_i;
            if px < 0 || py < 0 || px >= iw as i32 || py >= ih as i32 {
                continue;
            }
            // Clamp to cell bounds for source sampling.
            let cdx = dx_i.clamp(0, cs - 1) as u64;
            let cdy = dy_i.clamp(0, cs - 1) as u64;

            // Apply rotation to source coordinates.
            // Pieces are square (src_w == src_h).
            let (sx, sy) = match rot {
                0 => (cdx * src_w / cs64, cdy * src_h / cs64),
                90 => {
                    // 90 CW: (dx, dy) -> source (dy, size-1-dx)
                    (cdy * src_w / cs64, (cs64 - 1 - cdx) * src_h / cs64)
                }
                180 => (
                    (cs64 - 1 - cdx) * src_w / cs64,
                    (cs64 - 1 - cdy) * src_h / cs64,
                ),
                270 => {
                    // 270 CW: (dx, dy) -> source (size-1-dy, dx)
                    ((cs64 - 1 - cdy) * src_w / cs64, cdx * src_h / cs64)
                }
                _ => (cdx * src_w / cs64, cdy * src_h / cs64),
            };
            let si = ((sy * src_w + sx) * 3) as usize;
            if si + 2 < src.len() {
                img.put_pixel(
                    px as u32,
                    py as u32,
                    Rgb([src[si], src[si + 1], src[si + 2]]),
                );
            }
        }
    }
}

/// Draw the jigsaw-shaped outline of a piece.
fn draw_jigsaw_outline(
    img: &mut RgbImage,
    dst_x: u32,
    dst_y: u32,
    cell_size: u32,
    tab_r: i32,
    edges: &[EdgeType; 4],
    color: Rgb<u8>,
) {
    let (iw, ih) = (img.width(), img.height());
    let cs = cell_size as i32;

    for dy_i in -tab_r..(cs + tab_r) {
        for dx_i in -tab_r..(cs + tab_r) {
            if !in_jigsaw_shape(dx_i, dy_i, cs, tab_r, edges) {
                continue;
            }
            // Check if this pixel is on the boundary (any neighbour is outside).
            let on_edge = !in_jigsaw_shape(dx_i - 1, dy_i, cs, tab_r, edges)
                || !in_jigsaw_shape(dx_i + 1, dy_i, cs, tab_r, edges)
                || !in_jigsaw_shape(dx_i, dy_i - 1, cs, tab_r, edges)
                || !in_jigsaw_shape(dx_i, dy_i + 1, cs, tab_r, edges);
            if !on_edge {
                continue;
            }
            let px = dst_x as i32 + dx_i;
            let py = dst_y as i32 + dy_i;
            if px >= 0 && py >= 0 && (px as u32) < iw && (py as u32) < ih {
                img.put_pixel(px as u32, py as u32, color);
            }
        }
    }
}

/// Draw a plain rectangle outline (1px border).
fn draw_rect_outline(img: &mut RgbImage, x: u32, y: u32, w: u32, h: u32, color: Rgb<u8>) {
    let (iw, ih) = (img.width(), img.height());
    for dx in 0..w {
        let px = x + dx;
        if px < iw {
            if y < ih {
                img.put_pixel(px, y, color);
            }
            let by = y + h - 1;
            if by < ih {
                img.put_pixel(px, by, color);
            }
        }
    }
    for dy in 0..h {
        let py = y + dy;
        if py < ih {
            if x < iw {
                img.put_pixel(x, py, color);
            }
            let rx = x + w - 1;
            if rx < iw {
                img.put_pixel(rx, py, color);
            }
        }
    }
}

fn fill_rect(img: &mut RgbImage, x: u32, y: u32, w: u32, h: u32, color: Rgb<u8>) {
    let (iw, ih) = (img.width(), img.height());
    for dy in 0..h {
        for dx in 0..w {
            let px = x + dx;
            let py = y + dy;
            if px < iw && py < ih {
                img.put_pixel(px, py, color);
            }
        }
    }
}
