//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
use std::collections::HashSet;

use crate::base::game::RgbBuffer;

use super::state::MentalRotation2dState;

const BG: [u8; 3] = [0x1A, 0x1A, 0x2E];
const DIVIDER_COLOR: [u8; 3] = [0x44, 0x44, 0x66];

/// HSL-to-RGB conversion (h in [0,360), s and l in [0,1]).
fn hsl_to_rgb(h: f32, s: f32, l: f32) -> [u8; 3] {
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let h2 = h / 60.0;
    let x = c * (1.0 - (h2 % 2.0 - 1.0).abs());
    let (r1, g1, b1) = match h2 as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let m = l - c / 2.0;
    [
        ((r1 + m) * 255.0).round() as u8,
        ((g1 + m) * 255.0).round() as u8,
        ((b1 + m) * 255.0).round() as u8,
    ]
}

/// Derive fill and border colors from a seed.
fn colors_from_seed(seed: u32) -> ([u8; 3], [u8; 3]) {
    // Use seed to pick a hue, keep saturation/lightness in a pleasant range
    let hue = (seed % 360) as f32;
    let fill = hsl_to_rgb(hue, 0.65, 0.55);
    let border = hsl_to_rgb(hue, 0.55, 0.30);
    (fill, border)
}

/// Render the dual-view 2D mental rotation scene using inverse-mapping rasterization.
pub fn render(state: &MentalRotation2dState, resolution: u32) -> RgbBuffer {
    let puzzle = &state.puzzle;
    let (fill_color, border_color) = colors_from_seed(puzzle.color_seed);

    let divider_width = (resolution / 64).max(2) as usize;
    let half_w = ((resolution as usize) - divider_width) / 2;
    let h = resolution as usize;

    // --- Left view: player's current rotation (never flipped) ---
    let left_buf = render_panel(
        &puzzle.polyomino.cells,
        state.rotation,
        false,
        half_w,
        h,
        fill_color,
        border_color,
    );

    // --- Right view: target orientation (may be flipped in certificate mode) ---
    let right_buf = render_panel(
        &puzzle.polyomino.cells,
        puzzle.target_rotation,
        puzzle.target_flipped,
        half_w,
        h,
        fill_color,
        border_color,
    );

    // --- Stitch together: left | divider | right ---
    let total_w = half_w + divider_width + half_w;
    let mut data = Vec::with_capacity(total_w * h * 3);

    for row in 0..h {
        let left_start = row * half_w * 3;
        let left_end = left_start + half_w * 3;
        data.extend_from_slice(&left_buf[left_start..left_end]);

        for _ in 0..divider_width {
            data.push(DIVIDER_COLOR[0]);
            data.push(DIVIDER_COLOR[1]);
            data.push(DIVIDER_COLOR[2]);
        }

        let right_start = row * half_w * 3;
        let right_end = right_start + half_w * 3;
        data.extend_from_slice(&right_buf[right_start..right_end]);
    }

    RgbBuffer {
        width: total_w as u32,
        height: h as u32,
        data,
    }
}

/// Render a single panel via inverse-mapping rasterization.
///
/// For each pixel in the output:
/// 1. Convert to world coordinates
/// 2. Apply inverse transform (undo rotation and flip)
/// 3. Check if the resulting grid cell exists in the polyomino
/// 4. Detect cell borders via edge proximity in grid space
fn render_panel(
    cells: &[(i32, i32)],
    rotation_deg: u16,
    flipped: bool,
    width: usize,
    height: usize,
    fill_color: [u8; 3],
    border_color: [u8; 3],
) -> Vec<u8> {
    let cell_set: HashSet<(i32, i32)> = cells.iter().copied().collect();

    // Compute centroid in grid space
    let n = cells.len() as f32;
    let cx = cells.iter().map(|c| c.0 as f32).sum::<f32>() / n;
    let cy = cells.iter().map(|c| c.1 as f32).sum::<f32>() / n;

    // Compute bounding radius for framing
    let mut max_r = 0.0f32;
    for &(x, y) in cells {
        let dx = (x as f32 - cx).abs() + 0.5;
        let dy = (y as f32 - cy).abs() + 0.5;
        max_r = max_r.max(dx.max(dy));
    }
    // Use uniform scale (pixels per world unit) so squares stay square.
    // Fit the bounding circle into whichever panel axis is tighter.
    let world_extent = max_r * 2.5;
    let aspect = width as f32 / height as f32;
    let (scale_x, scale_y) = if aspect < 1.0 {
        // Panel is taller than wide — x is the tight axis
        (world_extent, world_extent / aspect)
    } else {
        (world_extent * aspect, world_extent)
    };

    // Precompute inverse rotation
    let to_rad = std::f32::consts::PI / 180.0;
    let angle = rotation_deg as f32 * to_rad;
    // Forward transform: R(θ) then optionally negate-x
    // Inverse: optionally negate-x, then R(-θ)
    let cos_neg = (-angle).cos();
    let sin_neg = (-angle).sin();

    let mut data = vec![0u8; width * height * 3];
    let border_threshold = 0.42; // cells are unit squares, border near edges

    for py in 0..height {
        for px in 0..width {
            // Pixel to world coordinates (centered, aspect-correct)
            let wx = (px as f32 + 0.5) / width as f32 * scale_x - scale_x / 2.0 + cx;
            let wy = (py as f32 + 0.5) / height as f32 * scale_y - scale_y / 2.0 + cy;

            // Offset from centroid in world space
            let dwx = wx - cx;
            let dwy = wy - cy;

            // Inverse transform: undo flip (negate x), then undo rotation R(-θ)
            let (ux, uy) = if flipped {
                // Undo flip: negate x
                let fx = -dwx;
                let fy = dwy;
                // Undo rotation: R(-θ)
                (cos_neg * fx - sin_neg * fy, sin_neg * fx + cos_neg * fy)
            } else {
                // Undo rotation: R(-θ)
                (cos_neg * dwx - sin_neg * dwy, sin_neg * dwx + cos_neg * dwy)
            };

            // Convert back to grid space (add centroid back)
            let gx = ux + cx;
            let gy = uy + cy;

            // Round to grid cell
            let cell_x = gx.round() as i32;
            let cell_y = gy.round() as i32;

            let idx = (py * width + px) * 3;
            if cell_set.contains(&(cell_x, cell_y)) {
                // Edge proximity for border
                let frac_x = (gx - cell_x as f32).abs();
                let frac_y = (gy - cell_y as f32).abs();
                if frac_x > border_threshold || frac_y > border_threshold {
                    data[idx] = border_color[0];
                    data[idx + 1] = border_color[1];
                    data[idx + 2] = border_color[2];
                } else {
                    data[idx] = fill_color[0];
                    data[idx + 1] = fill_color[1];
                    data[idx + 2] = fill_color[2];
                }
            } else {
                data[idx] = BG[0];
                data[idx + 1] = BG[1];
                data[idx + 2] = BG[2];
            }
        }
    }

    data
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::games::mental_rotation_2d::gen::generate;
    use crate::games::mental_rotation_2d::state::Rotation2dAction;
    use rand::SeedableRng;
    use rand_chacha::ChaChaRng;

    #[test]
    fn render_correct_dimensions() {
        let mut rng = ChaChaRng::seed_from_u64(42);
        let state =
            generate(&mut rng, 5, 90, 180, 180, 1, 100_000).expect("test seed should generate");
        let buf = render(&state, 128);
        assert_eq!(buf.data.len() as u32, buf.width * buf.height * 3);
    }

    #[test]
    fn render_has_nonbackground_pixels() {
        let mut rng = ChaChaRng::seed_from_u64(42);
        let state =
            generate(&mut rng, 5, 90, 180, 180, 1, 100_000).expect("test seed should generate");
        let buf = render(&state, 128);
        let non_bg = buf.data.chunks(3).filter(|p| *p != BG).count();
        assert!(
            non_bg > 50,
            "Should have visible polyomino pixels, got {}",
            non_bg
        );
    }

    #[test]
    fn render_different_rotations_differ() {
        let mut rng = ChaChaRng::seed_from_u64(42);
        let state =
            generate(&mut rng, 5, 90, 180, 180, 1, 100_000).expect("test seed should generate");
        let buf1 = render(&state, 64);

        let rotated = state.apply_action(Rotation2dAction::Rotate(90));
        let buf2 = render(&rotated, 64);
        assert_ne!(buf1.data, buf2.data);
    }
}
