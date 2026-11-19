//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
use std::hash::{Hash, Hasher};

use glam::Vec3;
use image::Rgb;
use imageproc::drawing::{draw_filled_circle_mut, draw_line_segment_mut};

use crate::base::game::RgbBuffer;
use crate::base::render::draw_text_centered;
use crate::base::render3d::{box_mesh, project_to_screen, render_scene, Camera, Light, Scene};

use super::state::{BloxorzGrid, BloxorzState, Orientation};

/// Max jitter in radians for horizontal and vertical camera angles (~8°).
const ANGLE_JITTER: f32 = 0.14;

/// Isometric camera with small per-puzzle angle perturbation.
/// The jitter is derived from a hash of the grid, so it stays constant
/// throughout an episode but varies across puzzles.
fn jittered_isometric_camera(
    target: Vec3,
    distance: f32,
    ortho_scale: f32,
    grid: &BloxorzGrid,
) -> Camera {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    grid.width.hash(&mut hasher);
    grid.height.hash(&mut hasher);
    grid.cells.hash(&mut hasher);
    grid.goal.hash(&mut hasher);
    grid.fragile_tiles.hash(&mut hasher);
    let hash = hasher.finish();

    // Map hash bytes to [-1, 1] range, then scale by ANGLE_JITTER
    let h_jitter = ((hash & 0xFFFF) as f32 / 0xFFFF as f32 * 2.0 - 1.0) * ANGLE_JITTER;
    let v_jitter = (((hash >> 16) & 0xFFFF) as f32 / 0xFFFF as f32 * 2.0 - 1.0) * ANGLE_JITTER;

    let angle_h = std::f32::consts::FRAC_PI_4 + h_jitter;
    let angle_v = (1.0_f32 / 2.0_f32.sqrt()).atan() + v_jitter;

    let dir = Vec3::new(
        angle_h.cos() * angle_v.cos(),
        angle_v.sin(),
        angle_h.sin() * angle_v.cos(),
    );

    Camera {
        eye: target + dir * distance,
        target,
        up: Vec3::Y,
        ortho_scale,
    }
}

// Tile face colors: [+x, -x, +y(top), -y(bottom), +z, -z]
const TILE_COLORS: [[u8; 3]; 6] = [
    [0x77, 0x88, 0x99],
    [0x66, 0x77, 0x88],
    [0x99, 0xAA, 0xBB],
    [0x55, 0x66, 0x77],
    [0x77, 0x88, 0x99],
    [0x66, 0x77, 0x88],
];
const GOAL_COLORS: [[u8; 3]; 6] = [
    [0xCC, 0x33, 0x33],
    [0xAA, 0x22, 0x22],
    [0xEE, 0x55, 0x55],
    [0x88, 0x11, 0x11],
    [0xCC, 0x33, 0x33],
    [0xAA, 0x22, 0x22],
];
const BLOCK_COLORS: [[u8; 3]; 6] = [
    [0x44, 0x55, 0xBB],
    [0x33, 0x44, 0x99],
    [0x55, 0x77, 0xEE],
    [0x22, 0x33, 0x77],
    [0x44, 0x55, 0xBB],
    [0x33, 0x44, 0x99],
];
// Fragile tile stripe color (amber).
const FRAGILE_COLORS: [[u8; 3]; 6] = [
    [0xCC, 0x99, 0x33],
    [0xAA, 0x77, 0x22],
    [0xEE, 0xBB, 0x44],
    [0x88, 0x66, 0x11],
    [0xCC, 0x99, 0x33],
    [0xAA, 0x77, 0x22],
];

/// Draw a compass overlay showing screen directions for U/D/L/R actions.
fn draw_compass(buf: RgbBuffer, camera: &Camera, resolution: u32) -> RgbBuffer {
    let mut img = buf.into_image();

    let radius = (resolution as f32 * 0.08).max(8.0);
    let margin = radius * 1.8;
    let cx = margin;
    let cy = resolution as f32 - margin;
    let font_height = radius * 0.7;

    // Dark background disc
    draw_filled_circle_mut(
        &mut img,
        (cx as i32, cy as i32),
        radius as i32 + 1,
        Rgb([0x15, 0x15, 0x25]),
    );

    // Project game directions to screen-space vectors
    let origin = camera.target;
    let (ox, oy) = project_to_screen(camera, origin, resolution, resolution);
    let directions: [(&str, Vec3); 4] = [
        ("U", Vec3::new(0.0, 0.0, -1.0)),
        ("D", Vec3::new(0.0, 0.0, 1.0)),
        ("L", Vec3::new(-1.0, 0.0, 0.0)),
        ("R", Vec3::new(1.0, 0.0, 0.0)),
    ];

    for (label, dir) in &directions {
        let (px, py) = project_to_screen(camera, origin + *dir, resolution, resolution);
        let dx = px - ox;
        let dy = py - oy;
        let len = (dx * dx + dy * dy).sqrt();
        if len < 1e-6 {
            continue;
        }
        let nx = dx / len;
        let ny = dy / len;

        // Line from center to tip
        let tip_x = cx + nx * radius * 0.75;
        let tip_y = cy + ny * radius * 0.75;
        draw_line_segment_mut(&mut img, (cx, cy), (tip_x, tip_y), Rgb([0xCC, 0xCC, 0xCC]));

        // Label past the tip
        let label_x = cx + nx * radius * 1.35;
        let label_y = cy + ny * radius * 1.35;
        draw_text_centered(
            &mut img,
            label_x as i32,
            label_y as i32,
            label.chars().next().unwrap(),
            font_height,
            Rgb([0xCC, 0xCC, 0xCC]),
        );
    }

    RgbBuffer::from_image(img)
}

/// Camera of the rendered scene: isometric, framing the solid cells tightly,
/// with the per-puzzle angle jitter. Shared with `object_boxes`.
pub fn scene_camera(state: &BloxorzState) -> Camera {
    let w = state.grid.width;
    let h = state.grid.height;
    let mut min_col = w;
    let mut max_col = 0;
    let mut min_row = h;
    let mut max_row = 0;
    for row in 0..h {
        for col in 0..w {
            if state.grid.is_solid(col, row) {
                min_col = min_col.min(col);
                max_col = max_col.max(col);
                min_row = min_row.min(row);
                max_row = max_row.max(row);
            }
        }
    }
    let span_w = (max_col - min_col + 1) as f32;
    let span_h = (max_row - min_row + 1) as f32;
    let grid_center = Vec3::new(
        (min_col as f32 + max_col as f32) / 2.0,
        0.0,
        (min_row as f32 + max_row as f32) / 2.0,
    );
    let ortho_scale = span_w.max(span_h) * 1.2;
    jittered_isometric_camera(grid_center, 20.0, ortho_scale, &state.grid)
}

pub fn render(state: &BloxorzState, resolution: u32) -> RgbBuffer {
    let w = state.grid.width;
    let h = state.grid.height;
    let mut meshes = Vec::new();

    // Grid tiles: thin boxes (height 0.3) for each solid cell
    let (gc, gr) = state.grid.goal;
    for row in 0..h {
        for col in 0..w {
            if !state.grid.is_solid(col, row) {
                continue;
            }
            // Skip broken fragile tiles (they are void now).
            if let Some(idx) = state.grid.fragile_index(col, row) {
                if state.broken_mask & (1 << idx) != 0 {
                    continue;
                }
            }
            let is_goal = col == gc && row == gr;
            let is_fragile = state.grid.fragile_index(col, row).is_some();
            if is_fragile && !is_goal {
                // Render as 3 parallel amber stripes along the Z axis.
                let stripe_half_z: f32 = 0.11;
                for &z_offset in &[-0.26_f32, 0.0, 0.26] {
                    let center = Vec3::new(col as f32, 0.15, row as f32 + z_offset);
                    meshes.push(box_mesh(
                        center,
                        Vec3::new(0.48, 0.15, stripe_half_z),
                        FRAGILE_COLORS,
                    ));
                }
            } else {
                let colors = if is_goal { GOAL_COLORS } else { TILE_COLORS };
                let center = Vec3::new(col as f32, 0.15, row as f32);
                meshes.push(box_mesh(center, Vec3::new(0.48, 0.15, 0.48), colors));
            }
        }
    }

    // Block: 1×1×2 prism. Half-unit = half the cell-filling width.
    let u: f32 = 0.46; // half-unit (block nearly fills tile's 0.96 width)
    let tile_top: f32 = 0.3;
    let (bx, bz, half_extents) = match state.orientation {
        Orientation::Standing => (
            state.col as f32,
            state.row as f32,
            Vec3::new(u, 2.0 * u, u), // 1×2×1
        ),
        Orientation::LyingX => (
            state.col as f32 + 0.5,
            state.row as f32,
            Vec3::new(2.0 * u, u, u), // 2×1×1
        ),
        Orientation::LyingY => (
            state.col as f32,
            state.row as f32 + 0.5,
            Vec3::new(u, u, 2.0 * u), // 1×1×2
        ),
    };
    let center = Vec3::new(bx, tile_top + half_extents.y, bz);
    meshes.push(box_mesh(center, half_extents, BLOCK_COLORS));

    let camera = scene_camera(state);

    let scene = Scene {
        meshes,
        camera: camera.clone(),
        light: Light {
            dir: Vec3::new(0.3, 1.0, 0.2).normalize(),
            ambient: 0.35,
            diffuse: 0.65,
        },
        bg: [0x1A, 0x1A, 0x2E],
    };

    let buf = render_scene(&scene, resolution);
    draw_compass(buf, &camera, resolution)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::games::bloxorz::state::from_ascii;

    #[test]
    fn render_dimensions() {
        let state = from_ascii("....\n.S..\n..G.\n....");
        let buf = render(&state, 64);
        assert_eq!(buf.width, 64);
        assert_eq!(buf.height, 64);
        assert_eq!(buf.data.len(), 64 * 64 * 3);
    }

    #[test]
    fn render_nonzero() {
        let state = from_ascii("....\n.S..\n..G.\n....");
        let buf = render(&state, 64);
        assert!(buf.data.iter().any(|&b| b != 0));
    }

    #[test]
    fn render_standing_vs_lying_differ() {
        let standing = from_ascii("....\n.S..\n..G.\n....");
        let lying = from_ascii("....\n.=-.\n..G.\n....");
        let buf_s = render(&standing, 64);
        let buf_l = render(&lying, 64);
        assert_ne!(buf_s.data, buf_l.data);
    }

    #[test]
    fn render_goal_visible() {
        let state = from_ascii("....\n.S..\n..G.\n....");
        let buf = render(&state, 128);
        let bg = [0x1A, 0x1A, 0x2E];
        // Goal is red-ish — look for pixels with R > 0x80 and R > G and R > B
        let has_red = buf
            .data
            .chunks(3)
            .any(|p| p != bg && p[0] > 0x80 && p[0] > p[1] + 0x20 && p[0] > p[2] + 0x20);
        assert!(has_red, "Goal tile should produce distinct red pixels");
    }

    #[test]
    fn render_block_above_tiles() {
        let state = from_ascii("....\n.S..\n..G.\n....");
        let buf = render(&state, 128);
        let bg = [0x1A, 0x1A, 0x2E];
        // Check that there are blue-ish block pixels (B > R and B > G)
        let has_blue = buf
            .data
            .chunks(3)
            .any(|p| p != bg && p[2] > 0x30 && p[2] > p[0] + 0x10 && p[2] > p[1] + 0x10);
        assert!(has_blue, "Block should produce distinct blue pixels");
    }

    #[test]
    fn render_at_128() {
        let state = from_ascii("....\n.S..\n..G.\n....");
        let buf = render(&state, 128);
        assert_eq!(buf.width, 128);
        assert_eq!(buf.height, 128);
        assert_eq!(buf.data.len(), 128 * 128 * 3);
        // A reasonable number of non-background pixels
        let bg = [0x1A, 0x1A, 0x2E];
        let non_bg = buf.data.chunks(3).filter(|p| *p != bg).count();
        assert!(
            non_bg > 100,
            "At 128x128, should have many visible pixels, got {}",
            non_bg
        );
    }
}
