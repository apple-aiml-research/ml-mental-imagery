//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
use std::sync::Arc;

use image::{Rgb, RgbImage};
use rand::Rng;
use rand_chacha::ChaChaRng;

use super::state::{GenCircle, GenRect, JigsawGenParams, JigsawPuzzle, JigsawState};

// ---------------------------------------------------------------------------
// Procedural image generation
// ---------------------------------------------------------------------------

/// Palette of vibrant, distinguishable colours for shapes.
const PALETTE: [[u8; 3]; 12] = [
    [220, 50, 50],   // red
    [50, 130, 220],  // blue
    [50, 180, 80],   // green
    [230, 160, 40],  // orange
    [160, 60, 200],  // purple
    [40, 190, 190],  // teal
    [230, 200, 40],  // yellow
    [200, 80, 140],  // pink
    [100, 60, 30],   // brown
    [80, 180, 130],  // sea-green
    [180, 130, 60],  // tan
    [100, 120, 200], // slate-blue
];

/// Generate a procedural image plus the exact primitives drawn (`JigsawGenParams`),
/// so the image can be reproduced and fingerprinted from those params without
/// touching pixels. Changing the RNG draw order changes every generated puzzle.
fn generate_image(width: u32, height: u32, rng: &mut ChaChaRng) -> (RgbImage, JigsawGenParams) {
    // --- draw params from RNG ---
    let bg_a = random_color(rng);
    let bg_b = random_color(rng);

    let num_rects = rng.gen_range(6..=14);
    let mut rects = Vec::with_capacity(num_rects as usize);
    for _ in 0..num_rects {
        let color_idx = rng.gen_range(0..PALETTE.len());
        let x = rng.gen_range(0..width);
        let y = rng.gen_range(0..height);
        let w = rng.gen_range(width / 8..=width / 3);
        let h = rng.gen_range(height / 8..=height / 3);
        rects.push(GenRect {
            color_idx: color_idx as u8,
            x,
            y,
            w,
            h,
        });
    }

    let num_circles = rng.gen_range(4..=10);
    let mut circles = Vec::with_capacity(num_circles as usize);
    for _ in 0..num_circles {
        let color_idx = rng.gen_range(0..PALETTE.len());
        let cx = rng.gen_range(0..width) as i32;
        let cy = rng.gen_range(0..height) as i32;
        let radius = rng.gen_range(width / 10..=width / 4) as i32;
        circles.push(GenCircle {
            color_idx: color_idx as u8,
            cx,
            cy,
            radius,
        });
    }

    let params = JigsawGenParams {
        bg_a,
        bg_b,
        rects,
        circles,
    };
    let img = paint_image(width, height, &params);
    (img, params)
}

/// Paint the procedural image deterministically from its generating params.
/// Pure (no RNG). `generate_image` and puzzle-fingerprint reconstruction both
/// go through this, so pixels and params can never diverge.
pub(super) fn paint_image(width: u32, height: u32, params: &JigsawGenParams) -> RgbImage {
    let mut img = RgbImage::new(width, height);

    // --- gradient background ---
    for y in 0..height {
        let t = y as f32 / height.max(1) as f32;
        let r = lerp_u8(params.bg_a[0], params.bg_b[0], t);
        let g = lerp_u8(params.bg_a[1], params.bg_b[1], t);
        let b = lerp_u8(params.bg_a[2], params.bg_b[2], t);
        for x in 0..width {
            img.put_pixel(x, y, Rgb([r, g, b]));
        }
    }

    // --- filled rectangles, in draw order ---
    for rect in &params.rects {
        fill_rect(
            &mut img,
            rect.x,
            rect.y,
            rect.w,
            rect.h,
            PALETTE[rect.color_idx as usize],
        );
    }

    // --- filled circles, in draw order (painted after all rectangles) ---
    for c in &params.circles {
        fill_circle(
            &mut img,
            c.cx,
            c.cy,
            c.radius,
            PALETTE[c.color_idx as usize],
        );
    }

    img
}

fn random_color(rng: &mut ChaChaRng) -> [u8; 3] {
    [
        rng.gen_range(60..=240),
        rng.gen_range(60..=240),
        rng.gen_range(60..=240),
    ]
}

fn lerp_u8(a: u8, b: u8, t: f32) -> u8 {
    (a as f32 * (1.0 - t) + b as f32 * t).round() as u8
}

fn fill_rect(img: &mut RgbImage, x: u32, y: u32, w: u32, h: u32, color: [u8; 3]) {
    let (iw, ih) = (img.width(), img.height());
    for dy in 0..h {
        for dx in 0..w {
            let px = x + dx;
            let py = y + dy;
            if px < iw && py < ih {
                img.put_pixel(px, py, Rgb(color));
            }
        }
    }
}

fn fill_circle(img: &mut RgbImage, cx: i32, cy: i32, radius: i32, color: [u8; 3]) {
    let (iw, ih) = (img.width() as i32, img.height() as i32);
    let r2 = radius * radius;
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            if dx * dx + dy * dy <= r2 {
                let px = cx + dx;
                let py = cy + dy;
                if px >= 0 && px < iw && py >= 0 && py < ih {
                    img.put_pixel(px as u32, py as u32, Rgb(color));
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Puzzle slicing
// ---------------------------------------------------------------------------

/// Slice an image into grid_size x grid_size rectangular pieces.
/// Returns (piece_images, piece_width, piece_height).
fn slice_image(img: &RgbImage, grid_size: usize) -> (Vec<Vec<u8>>, u32, u32) {
    let (iw, ih) = (img.width(), img.height());
    let piece_w = iw / grid_size as u32;
    let piece_h = ih / grid_size as u32;
    let mut pieces = Vec::with_capacity(grid_size * grid_size);

    for row in 0..grid_size {
        for col in 0..grid_size {
            let ox = col as u32 * piece_w;
            let oy = row as u32 * piece_h;
            let mut data = Vec::with_capacity((piece_w * piece_h * 3) as usize);
            for y in 0..piece_h {
                for x in 0..piece_w {
                    let px = img.get_pixel(ox + x, oy + y);
                    data.extend_from_slice(&px.0);
                }
            }
            pieces.push(data);
        }
    }

    (pieces, piece_w, piece_h)
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Internal image resolution per piece cell.
const PIECE_RESOLUTION: u32 = 64;

/// Generate a jigsaw puzzle with all pieces unplaced.
pub fn generate(grid_size: usize, rng: &mut ChaChaRng) -> JigsawState {
    let img_size = grid_size as u32 * PIECE_RESOLUTION;
    let (img, gen_params) = generate_image(img_size, img_size, rng);
    let (piece_images, piece_width, piece_height) = slice_image(&img, grid_size);
    let num_pieces = grid_size * grid_size;

    // Shuffle tray order so pieces don't reveal the solution.
    let mut tray_order: Vec<usize> = (0..num_pieces).collect();
    for i in (1..tray_order.len()).rev() {
        let j = rng.gen_range(0..=i);
        tray_order.swap(i, j);
    }

    // Random jigsaw edge assignments.
    let h_tabs: Vec<bool> = (0..((grid_size - 1) * grid_size))
        .map(|_| rng.gen_bool(0.5))
        .collect();
    let v_tabs: Vec<bool> = (0..(grid_size * (grid_size - 1)))
        .map(|_| rng.gen_bool(0.5))
        .collect();

    let puzzle = Arc::new(JigsawPuzzle {
        grid_size,
        piece_width,
        piece_height,
        piece_images,
        tray_order,
        h_tabs,
        v_tabs,
        tray_layout: std::sync::OnceLock::new(),
        gen_params,
    });

    // Random initial rotations for each piece.
    let rotation_choices = [0i32, 90, 180, 270];
    let rotations: Vec<i32> = (0..num_pieces)
        .map(|_| rotation_choices[rng.gen_range(0..4)])
        .collect();

    JigsawState {
        puzzle,
        grid: vec![None; num_pieces],
        unplaced: (0..num_pieces).collect(),
        rotations,
    }
}

/// Generate a trivially solvable state: all pieces placed except piece 0.
pub fn trivial(grid_size: usize) -> JigsawState {
    let img_size = grid_size as u32 * PIECE_RESOLUTION;
    // Deterministic solid-colour image for the trivial fallback.
    let mut img = RgbImage::new(img_size, img_size);
    for (i, pix) in img.pixels_mut().enumerate() {
        let idx = i % PALETTE.len();
        *pix = Rgb(PALETTE[idx]);
    }
    let (piece_images, piece_width, piece_height) = slice_image(&img, grid_size);
    let num_pieces = grid_size * grid_size;

    let puzzle = Arc::new(JigsawPuzzle {
        grid_size,
        piece_width,
        piece_height,
        piece_images,
        tray_order: (0..num_pieces).collect(),
        h_tabs: vec![false; (grid_size - 1) * grid_size],
        v_tabs: vec![false; grid_size * (grid_size - 1)],
        tray_layout: std::sync::OnceLock::new(),
        // Sentinel: no procedural primitives (distinguishes the trivial
        // exhaustion fallback from real generated puzzles).
        gen_params: JigsawGenParams::default(),
    });

    // Place all pieces except piece 0.
    let mut grid: Vec<Option<usize>> = (0..num_pieces).map(|i| Some(i)).collect();
    grid[0] = None;

    JigsawState {
        puzzle,
        grid,
        unplaced: vec![0],
        rotations: vec![0; num_pieces],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn generate_produces_correct_piece_count() {
        let mut rng = ChaChaRng::seed_from_u64(42);
        let state = generate(3, &mut rng);
        assert_eq!(state.puzzle.piece_images.len(), 9);
        assert_eq!(state.unplaced.len(), 9);
        assert!(state.grid.iter().all(|c| c.is_none()));
    }

    #[test]
    fn trivial_has_one_unplaced() {
        let state = trivial(3);
        assert_eq!(state.unplaced, vec![0]);
        assert_eq!(state.grid[0], None);
        for i in 1..9 {
            assert_eq!(state.grid[i], Some(i));
        }
    }

    #[test]
    fn slice_produces_correct_dimensions() {
        let img = RgbImage::new(128, 128);
        let (pieces, pw, ph) = slice_image(&img, 4);
        assert_eq!(pieces.len(), 16);
        assert_eq!(pw, 32);
        assert_eq!(ph, 32);
        assert_eq!(pieces[0].len(), (32 * 32 * 3) as usize);
    }

    #[test]
    fn gen_params_reproduce_piece_images() {
        // The captured generating params must reproduce the exact pixels, so
        // fingerprinting the params is equivalent to fingerprinting the image.
        for grid_size in [2usize, 3] {
            for seed in 0..25u64 {
                let mut rng = ChaChaRng::seed_from_u64(seed);
                let state = generate(grid_size, &mut rng);
                let img_size = grid_size as u32 * PIECE_RESOLUTION;
                let rebuilt = paint_image(img_size, img_size, &state.puzzle.gen_params);
                let (rebuilt_pieces, _, _) = slice_image(&rebuilt, grid_size);
                assert_eq!(
                    rebuilt_pieces, state.puzzle.piece_images,
                    "gen_params did not reproduce piece_images (grid {grid_size}, seed {seed})"
                );
            }
        }
    }

    #[test]
    fn generation_rng_stream_unchanged_by_capture() {
        // The whole generated puzzle, including the draws after the image
        // (tray_order, tabs, rotations), is deterministic per seed.
        let mut rng_a = ChaChaRng::seed_from_u64(123);
        let mut rng_b = ChaChaRng::seed_from_u64(123);
        let a = generate(3, &mut rng_a);
        let b = generate(3, &mut rng_b);
        assert_eq!(a.puzzle.piece_images, b.puzzle.piece_images);
        assert_eq!(a.puzzle.tray_order, b.puzzle.tray_order);
        assert_eq!(a.puzzle.h_tabs, b.puzzle.h_tabs);
        assert_eq!(a.puzzle.v_tabs, b.puzzle.v_tabs);
        assert_eq!(a.rotations, b.rotations);
    }
}
