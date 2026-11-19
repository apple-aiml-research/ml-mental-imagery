//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
use std::collections::HashSet;
use std::sync::OnceLock;

use ab_glyph::PxScale;
use image::{Rgb, RgbImage};
use imageproc::drawing::draw_text_mut;

use crate::base::game::RgbBuffer;
use crate::base::render::{downsample, draw_text_centered};

use super::gen;
use super::state::*;

const BG_COLOR: Rgb<u8> = Rgb([255, 255, 255]);
const ASSEMBLY_BG: Rgb<u8> = Rgb([255, 255, 255]);
const PIECE_LABEL_COLOR: Rgb<u8> = Rgb([80, 80, 80]);
const DIVIDER_COLOR: Rgb<u8> = Rgb([200, 200, 200]);
const GRID_LINE_COLOR: Rgb<u8> = Rgb([210, 210, 210]);
const INTERSECTION_COLOR: Rgb<u8> = Rgb([170, 170, 170]);
const PALETTE_BG: Rgb<u8> = Rgb([255, 255, 255]);

// ---------------------------------------------------------------------------
// Character rendering parameters (computed once per puzzle, cached)
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct CharRenderParams {
    src_img: RgbImage,
    min_px: f64,
    min_py: f64,
    scale_factor: f64,
    off_x: f64,
    off_y: f64,
}

impl std::fmt::Debug for CharRenderParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CharRenderParams")
            .field("scale_factor", &self.scale_factor)
            .finish_non_exhaustive()
    }
}

/// Cache for `CharRenderParams`, stored on the puzzle via `OnceLock`.
pub type CharRenderCache = OnceLock<Option<CharRenderParams>>;

fn compute_char_params(font_idx: usize, ch: char, grid_size: usize) -> Option<CharRenderParams> {
    let render_size = 200u32;
    let font = &gen::fonts()[font_idx];
    let mut src_img = RgbImage::from_pixel(render_size, render_size, Rgb([0, 0, 0]));
    let font_height = render_size as f32 * 0.85;
    let scale = PxScale::from(font_height);
    let approx_w = (font_height * 0.6) as i32;
    let x = (render_size as i32 - approx_w) / 2;
    let y = (render_size as i32 - font_height as i32) / 4;
    draw_text_mut(
        &mut src_img,
        Rgb([255, 255, 255]),
        x,
        y,
        scale,
        font,
        &ch.to_string(),
    );

    let mut min_px = render_size;
    let mut min_py = render_size;
    let mut max_px = 0u32;
    let mut max_py = 0u32;
    for py in 0..render_size {
        for px in 0..render_size {
            if src_img.get_pixel(px, py)[0] > 128 {
                min_px = min_px.min(px);
                min_py = min_py.min(py);
                max_px = max_px.max(px);
                max_py = max_py.max(py);
            }
        }
    }
    if max_px < min_px {
        return None;
    }

    let src_w = (max_px - min_px + 1) as f64;
    let src_h = (max_py - min_py + 1) as f64;
    let margin = 1;
    let available = (grid_size - 2 * margin) as f64;
    if available <= 0.0 {
        return None;
    }
    let sx = available / src_w;
    let sy = available / src_h;
    let scale_factor = sx.min(sy);
    let dest_w = (src_w * scale_factor).round() as usize;
    let dest_h = (src_h * scale_factor).round() as usize;
    let off_x = ((grid_size - dest_w) / 2) as f64;
    let off_y = ((grid_size - dest_h) / 2) as f64;

    Some(CharRenderParams {
        src_img,
        min_px: min_px as f64,
        min_py: min_py as f64,
        scale_factor,
        off_x,
        off_y,
    })
}

fn sample_alpha(params: &CharRenderParams, grid_x: f64, grid_y: f64) -> f32 {
    let src_x = (grid_x - params.off_x) / params.scale_factor + params.min_px;
    let src_y = (grid_y - params.off_y) / params.scale_factor + params.min_py;
    bilinear_sample(&params.src_img, src_x, src_y)
}

fn bilinear_sample(img: &RgbImage, x: f64, y: f64) -> f32 {
    let w = img.width() as i32;
    let h = img.height() as i32;
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let fx = (x - x0 as f64) as f32;
    let fy = (y - y0 as f64) as f32;

    let s = |xi: i32, yi: i32| -> f32 {
        if xi >= 0 && yi >= 0 && xi < w && yi < h {
            img.get_pixel(xi as u32, yi as u32)[0] as f32 / 255.0
        } else {
            0.0
        }
    };

    s(x0, y0) * (1.0 - fx) * (1.0 - fy)
        + s(x0 + 1, y0) * fx * (1.0 - fy)
        + s(x0, y0 + 1) * (1.0 - fx) * fy
        + s(x0 + 1, y0 + 1) * fx * fy
}

fn blend(bg: Rgb<u8>, fg: [u8; 3], alpha: f32) -> Rgb<u8> {
    let a = alpha.clamp(0.0, 1.0);
    let inv = 1.0 - a;
    Rgb([
        (bg.0[0] as f32 * inv + fg[0] as f32 * a) as u8,
        (bg.0[1] as f32 * inv + fg[1] as f32 * a) as u8,
        (bg.0[2] as f32 * inv + fg[2] as f32 * a) as u8,
    ])
}

// ---------------------------------------------------------------------------
// Main render
// ---------------------------------------------------------------------------

pub fn render(state: &CharacterRecognitionState, resolution: u32) -> RgbBuffer {
    let aa_factor = 2u32;
    let hi_res = resolution * aa_factor;
    let buf = render_at_resolution(state, hi_res);
    downsample(&buf, aa_factor)
}

fn render_at_resolution(state: &CharacterRecognitionState, resolution: u32) -> RgbBuffer {
    let mut img = RgbImage::from_pixel(resolution, resolution, BG_COLOR);
    let puzzle = &state.puzzle;
    let grid_size = puzzle.grid_size;

    let params = puzzle
        .char_render_cache
        .get_or_init(|| compute_char_params(puzzle.font_idx, puzzle.character_char, grid_size));

    // Layout: 50% assembly, 5% gap, 45% palette
    let assembly_h = resolution / 2;
    let palette_top = resolution * 55 / 100;
    let palette_h = resolution.saturating_sub(palette_top);

    let margin = resolution / 14;
    let avail_w = resolution.saturating_sub(2 * margin);
    let avail_h = assembly_h.saturating_sub(2 * margin);
    let cell_size = (avail_w / grid_size as u32)
        .min(avail_h / grid_size as u32)
        .max(1);

    let grid_pixel_w = cell_size * grid_size as u32;
    let grid_pixel_h = cell_size * grid_size as u32;
    let grid_x0 = (resolution - grid_pixel_w) / 2;
    let grid_y0 = (assembly_h - grid_pixel_h) / 2;

    let label_font_h = (resolution as f32 / 15.0).max(16.0);

    // Assembly background
    fill_rect(
        &mut img,
        grid_x0.saturating_sub(2),
        grid_y0.saturating_sub(2),
        grid_pixel_w + 4,
        grid_pixel_h + 4,
        ASSEMBLY_BG,
    );

    // Grid overlay: N lines in each direction at intersection positions
    let grid_dim = puzzle.slot_grid_dim;
    let grid_line_positions_x: Vec<u32> = (0..grid_dim)
        .map(|i| grid_x0 + ((i + 1) as u32 * grid_pixel_w / (grid_dim + 1) as u32))
        .collect();
    let grid_line_positions_y: Vec<u32> = (0..grid_dim)
        .map(|i| grid_y0 + ((i + 1) as u32 * grid_pixel_h / (grid_dim + 1) as u32))
        .collect();

    for &vx in &grid_line_positions_x {
        for y in grid_y0..grid_y0 + grid_pixel_h {
            if vx < img.width() && y < img.height() {
                img.put_pixel(vx, y, GRID_LINE_COLOR);
            }
        }
    }
    for &hy in &grid_line_positions_y {
        for x in grid_x0..grid_x0 + grid_pixel_w {
            if x < img.width() && hy < img.height() {
                img.put_pixel(x, hy, GRID_LINE_COLOR);
            }
        }
    }

    // Intersection circles
    let circle_r = (cell_size / 3).max(2);
    for &hy in &grid_line_positions_y {
        for &vx in &grid_line_positions_x {
            draw_filled_circle(&mut img, vx as i32, hy as i32, circle_r, INTERSECTION_COLOR);
        }
    }

    // Row labels (left of grid)
    let label_font_small = (label_font_h * 0.7).max(12.0);
    for (i, &hy) in grid_line_positions_y.iter().enumerate() {
        let label_x = grid_x0.saturating_sub(margin / 2) as i32;
        draw_text_centered(
            &mut img,
            label_x,
            hy as i32,
            char::from_digit(i as u32, 10).unwrap_or('?'),
            label_font_small,
            PIECE_LABEL_COLOR,
        );
    }
    // Column labels (above grid)
    for (i, &vx) in grid_line_positions_x.iter().enumerate() {
        let label_y = grid_y0.saturating_sub(margin / 3) as i32;
        draw_text_centered(
            &mut img,
            vx as i32,
            label_y,
            char::from_digit(i as u32, 10).unwrap_or('?'),
            label_font_small,
            PIECE_LABEL_COLOR,
        );
    }

    // Build placed set: which piece indices are placed?
    let mut placed_pieces: Vec<bool> = vec![false; puzzle.num_pieces];
    for pid in 0..puzzle.num_pieces {
        if state.placements[pid].is_some() && state.piece_correct(pid) {
            placed_pieces[pid] = true;
        }
    }

    // Also build grid map for incorrectly-placed pieces (no cut tree for those)
    let mut placed_map: Vec<Option<usize>> = vec![None; grid_size * grid_size];
    for pid in 0..puzzle.num_pieces {
        if state.placements[pid].is_some() && !state.piece_correct(pid) {
            let slot_idx = state.placements[pid].unwrap();
            let slot = &puzzle.slots[slot_idx];
            let piece = &puzzle.pieces[pid];
            let rendered = piece.rendered_cells(state.rotations[pid], slot.centroid);
            for &(gx, gy) in &rendered {
                if gx >= 0 && gy >= 0 && (gx as usize) < grid_size && (gy as usize) < grid_size {
                    placed_map[gy as usize * grid_size + gx as usize] = Some(pid);
                }
            }
        }
    }

    if let Some(ref params) = params {
        for py in 0..grid_pixel_h {
            for px in 0..grid_pixel_w {
                let gx_f = (px as f64 + 0.5) / cell_size as f64;
                let gy_f = (py as f64 + 0.5) / cell_size as f64;

                let font_alpha = sample_alpha(params, gx_f, gy_f);
                if font_alpha < 0.005 {
                    continue;
                }

                // Check if pixel belongs to a correctly-placed piece
                let mut show = false;
                if let Some(ref lookup) = puzzle.piece_lookup {
                    let pid = lookup.piece_at(gx_f, gy_f);
                    if pid < puzzle.num_pieces && placed_pieces[pid] {
                        show = true;
                    }
                }

                // Fallback: check grid map for incorrectly-placed pieces
                if !show {
                    let gx = gx_f.floor() as usize;
                    let gy = gy_f.floor() as usize;
                    if gx < grid_size && gy < grid_size {
                        if placed_map[gy * grid_size + gx].is_some() {
                            show = true;
                        }
                        if !show && puzzle.piece_lookup.is_none() {
                            for pid in 0..puzzle.num_pieces {
                                if placed_pieces[pid] {
                                    if puzzle.pieces[pid].cells.contains(&(gx as i32, gy as i32)) {
                                        show = true;
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }

                if show {
                    let color = blend(ASSEMBLY_BG, puzzle.fill_color, font_alpha);
                    let ix = grid_x0 + px;
                    let iy = grid_y0 + py;
                    if ix < img.width() && iy < img.height() {
                        img.put_pixel(ix, iy, color);
                    }
                }
            }
        }
    }

    // Divider
    let div_y = palette_top.saturating_sub(3);
    for x in (margin / 2)..(resolution - margin / 2) {
        if div_y < resolution {
            img.put_pixel(x, div_y, DIVIDER_COLOR);
        }
    }

    // Palette background
    fill_rect(&mut img, 0, palette_top, resolution, palette_h, PALETTE_BG);

    // Palette: all pieces get a fixed slot; only unplaced ones are drawn
    let num_pieces = puzzle.num_pieces;

    if num_pieces > 0 && palette_h > 4 {
        let section_w = resolution / num_pieces as u32;
        let label_space = (palette_h as f64 * 0.18) as u32;

        for pid in 0..num_pieces {
            if state.placements[pid].is_some() {
                continue;
            }
            let piece = &puzzle.pieces[pid];
            let rotation = state.rotations[pid];

            let center_x = pid as u32 * section_w + section_w / 2;
            let center_y = palette_top + label_space + (palette_h - label_space) / 2;

            render_piece(
                &mut img,
                piece,
                pid,
                rotation,
                center_x,
                center_y,
                cell_size,
                &params,
                &puzzle.piece_lookup,
                puzzle.fill_color,
                PALETTE_BG,
            );

            // Piece label
            let label_x = center_x as i32;
            let label_y = (palette_top + label_space / 2) as i32;
            let font_h = label_font_h;
            draw_text_centered(
                &mut img,
                label_x,
                label_y,
                char::from_digit(pid as u32, 10).unwrap_or('?'),
                font_h,
                PIECE_LABEL_COLOR,
            );
        }
    }

    RgbBuffer::from_image(img)
}

/// Render a single piece at a pixel center using graph lookup for pixel-precise boundaries.
fn render_piece(
    img: &mut RgbImage,
    piece: &PieceDef,
    piece_idx: usize,
    rotation: u8,
    center_px: u32,
    center_py: u32,
    cell_size: u32,
    params: &Option<CharRenderParams>,
    piece_lookup: &Option<PieceLookup>,
    fill_color: [u8; 3],
    bg_color: Rgb<u8>,
) {
    if piece.cells.is_empty() {
        return;
    }
    let min_gx = piece.cells.iter().map(|&(x, _)| x).min().unwrap();
    let max_gx = piece.cells.iter().map(|&(x, _)| x).max().unwrap();
    let min_gy = piece.cells.iter().map(|&(_, y)| y).min().unwrap();
    let max_gy = piece.cells.iter().map(|&(_, y)| y).max().unwrap();
    let piece_grid_w = (max_gx - min_gx + 1) as u32;
    let piece_grid_h = (max_gy - min_gy + 1) as u32;

    let (rot_grid_w, rot_grid_h) = if rotation % 2 == 0 {
        (piece_grid_w, piece_grid_h)
    } else {
        (piece_grid_h, piece_grid_w)
    };

    let display_w = rot_grid_w * cell_size;
    let display_h = rot_grid_h * cell_size;

    let piece_cell_set: HashSet<(i32, i32)> = piece.cells.iter().copied().collect();

    if let Some(ref params) = params {
        for py in 0..display_h {
            for px in 0..display_w {
                let dx = (px as f64 + 0.5) / cell_size as f64 - rot_grid_w as f64 / 2.0;
                let dy = (py as f64 + 0.5) / cell_size as f64 - rot_grid_h as f64 / 2.0;

                let (orig_dx, orig_dy) = match rotation % 4 {
                    0 => (dx, dy),
                    1 => (dy, -dx),
                    2 => (-dx, -dy),
                    3 => (-dy, dx),
                    _ => unreachable!(),
                };

                let gx_f = piece.centroid.0 + orig_dx;
                let gy_f = piece.centroid.1 + orig_dy;

                let belongs = if let Some(ref lookup) = piece_lookup {
                    lookup.piece_at(gx_f, gy_f) == piece_idx
                } else {
                    piece_cell_set.contains(&(gx_f.floor() as i32, gy_f.floor() as i32))
                };

                if belongs {
                    let font_alpha = sample_alpha(params, gx_f, gy_f);
                    if font_alpha > 0.003 {
                        let color = blend(bg_color, fill_color, font_alpha);
                        let ix = center_px as i32 + px as i32 - display_w as i32 / 2;
                        let iy = center_py as i32 + py as i32 - display_h as i32 / 2;
                        if ix >= 0
                            && iy >= 0
                            && (ix as u32) < img.width()
                            && (iy as u32) < img.height()
                        {
                            img.put_pixel(ix as u32, iy as u32, color);
                        }
                    }
                }
            }
        }
    }
}

fn fill_rect(img: &mut RgbImage, x: u32, y: u32, w: u32, h: u32, color: Rgb<u8>) {
    let img_w = img.width();
    let img_h = img.height();
    for dy in 0..h {
        for dx in 0..w {
            let px = x + dx;
            let py = y + dy;
            if px < img_w && py < img_h {
                img.put_pixel(px, py, color);
            }
        }
    }
}

fn draw_filled_circle(img: &mut RgbImage, cx: i32, cy: i32, radius: u32, color: Rgb<u8>) {
    let r = radius as i32;
    let r_sq = r * r;
    for dy in -r..=r {
        for dx in -r..=r {
            if dx * dx + dy * dy <= r_sq {
                let px = cx + dx;
                let py = cy + dy;
                if px >= 0 && py >= 0 && (px as u32) < img.width() && (py as u32) < img.height() {
                    img.put_pixel(px as u32, py as u32, color);
                }
            }
        }
    }
}
