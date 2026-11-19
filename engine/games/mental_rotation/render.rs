//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
use glam::Vec3;
use image::Rgb;
use imageproc::drawing::draw_line_segment_mut;

use crate::base::game::RgbBuffer;
use crate::base::render::draw_text_centered;
use crate::base::render3d::{
    box_mesh, isometric_camera, project_to_screen, render_scene_rect, Light, Mesh, Scene,
};

use super::state::{rotation_matrix, MentalRotationState};

// Per-face-direction colors: [+x, -x, +y, -y, +z, -z]
const FACE_COLORS: [[u8; 3]; 6] = [
    [0xCC, 0x33, 0x33], // +X red
    [0x33, 0xAA, 0x33], // -X green
    [0x33, 0x55, 0xCC], // +Y blue
    [0xCC, 0xCC, 0x33], // -Y yellow
    [0xDD, 0x77, 0x22], // +Z orange
    [0x88, 0x33, 0xBB], // -Z purple
];

const BG: [u8; 3] = [0x1A, 0x1A, 0x2E];
const DIVIDER_COLOR: [u8; 3] = [0x44, 0x44, 0x66];

/// Build mesh for the polycube at the given voxel positions, centered on `centroid`.
fn build_polycube_mesh(voxels: &[(i32, i32, i32)], centroid: Vec3) -> Mesh {
    let half = 0.46;
    let mut all_tris = Vec::new();
    for &(x, y, z) in voxels {
        let center = Vec3::new(x as f32, y as f32, z as f32) - centroid;
        let m = box_mesh(center, Vec3::splat(half), FACE_COLORS);
        all_tris.extend(m.tris);
    }
    Mesh { tris: all_tris }
}

/// Compute bounding sphere radius from polycube voxels.
fn bounding_radius(voxels: &[(i32, i32, i32)]) -> f32 {
    let n = voxels.len() as f32;
    let cx: f32 = voxels.iter().map(|v| v.0 as f32).sum::<f32>() / n;
    let cy: f32 = voxels.iter().map(|v| v.1 as f32).sum::<f32>() / n;
    let cz: f32 = voxels.iter().map(|v| v.2 as f32).sum::<f32>() / n;

    let mut max_r2 = 0.0f32;
    for &(x, y, z) in voxels {
        let dx = x as f32 - cx;
        let dy = y as f32 - cy;
        let dz = z as f32 - cz;
        // Add 0.5 for voxel extent
        let r2 = (dx.abs() + 0.5).powi(2) + (dy.abs() + 0.5).powi(2) + (dz.abs() + 0.5).powi(2);
        max_r2 = max_r2.max(r2);
    }
    max_r2.sqrt()
}

const AXIS_COLORS: [(Rgb<u8>, char); 3] = [
    (Rgb([0xCC, 0x33, 0x33]), 'X'), // +X red
    (Rgb([0x33, 0x55, 0xCC]), 'Y'), // +Y blue
    (Rgb([0xDD, 0x77, 0x22]), 'Z'), // +Z orange
];

const AXIS_DIRS: [Vec3; 3] = [Vec3::X, Vec3::Y, Vec3::Z];

/// Draw a small XYZ axis gizmo in the bottom-left corner of a panel.
fn draw_axis_gizmo(buf: RgbBuffer, camera: &crate::base::render3d::Camera) -> RgbBuffer {
    let w = buf.width;
    let h = buf.height;
    let mut img = buf.into_image();

    let axis_len = h as f32 * 0.12;
    let margin = h as f32 * 0.14;
    let center_x = margin;
    let center_y = h as f32 - margin;

    let origin_screen = project_to_screen(camera, Vec3::ZERO, w, h);

    for (i, dir) in AXIS_DIRS.iter().enumerate() {
        let end_screen = project_to_screen(camera, *dir, w, h);

        let dx = end_screen.0 - origin_screen.0;
        let dy = end_screen.1 - origin_screen.1;
        let len = (dx * dx + dy * dy).sqrt().max(1e-6);
        let nx = dx / len * axis_len;
        let ny = dy / len * axis_len;

        let (color, label) = AXIS_COLORS[i];

        draw_line_segment_mut(
            &mut img,
            (center_x, center_y),
            (center_x + nx, center_y + ny),
            color,
        );

        let font_height = (h as f32 * 0.07).max(8.0);
        let label_x = center_x + nx * 1.25;
        let label_y = center_y + ny * 1.25;
        draw_text_centered(
            &mut img,
            label_x as i32,
            label_y as i32,
            label,
            font_height,
            color,
        );
    }

    RgbBuffer::from_image(img)
}

/// Render the dual-view mental rotation scene.
pub fn render(state: &MentalRotationState, resolution: u32, zoom_pct: u32) -> RgbBuffer {
    let puzzle = &state.puzzle;
    let voxels = &puzzle.polycube.voxels;

    let divider_width = (resolution / 64).max(2) as usize;
    let half_w = ((resolution as usize) - divider_width) / 2;
    let h = resolution as usize;

    // Compute ortho scale from bounding sphere, accounting for panel aspect ratio.
    // Each half-panel is half_w × h, so aspect < 1 means horizontal is the
    // tight dimension. Scale up so the sphere fits the narrower axis.
    let radius = bounding_radius(voxels);
    let aspect = half_w as f32 / h as f32;
    let ortho_scale = radius * 2.5 / aspect.min(1.0) / (zoom_pct as f32 / 100.0);

    // Center mesh on centroid so rotations spin in place
    let n = voxels.len() as f32;
    let centroid = Vec3::new(
        voxels.iter().map(|v| v.0 as f32).sum::<f32>() / n,
        voxels.iter().map(|v| v.1 as f32).sum::<f32>() / n,
        voxels.iter().map(|v| v.2 as f32).sum::<f32>() / n,
    );

    // --- Left view: player's current rotation ---
    let player_mat = rotation_matrix(state.rx, state.ry, state.rz);
    let mut player_mesh = build_polycube_mesh(voxels, centroid);
    player_mesh.transform(player_mat);

    let camera = isometric_camera(Vec3::ZERO, 20.0, ortho_scale);
    let light = Light {
        dir: Vec3::new(0.3, 1.0, 0.2).normalize(),
        ambient: 0.35,
        diffuse: 0.65,
    };

    let left_scene = Scene {
        meshes: vec![player_mesh],
        camera: camera.clone(),
        light: light.clone(),
        bg: BG,
    };
    let left_buf = render_scene_rect(&left_scene, half_w as u32, h as u32);
    let left_buf = draw_axis_gizmo(left_buf, &camera);

    // --- Right view: target rotation ---
    // When target_polycube is set (unsolvable), render the different shape on the right.
    let target_mat = rotation_matrix(puzzle.target_rx, puzzle.target_ry, puzzle.target_rz);
    let mut target_mesh = if let Some(ref target_poly) = puzzle.target_polycube {
        let tn = target_poly.voxels.len() as f32;
        let target_centroid = Vec3::new(
            target_poly.voxels.iter().map(|v| v.0 as f32).sum::<f32>() / tn,
            target_poly.voxels.iter().map(|v| v.1 as f32).sum::<f32>() / tn,
            target_poly.voxels.iter().map(|v| v.2 as f32).sum::<f32>() / tn,
        );
        build_polycube_mesh(&target_poly.voxels, target_centroid)
    } else {
        build_polycube_mesh(voxels, centroid)
    };
    target_mesh.transform(target_mat);

    let right_scene = Scene {
        meshes: vec![target_mesh],
        camera,
        light,
        bg: BG,
    };
    let right_buf = render_scene_rect(&right_scene, half_w as u32, h as u32);

    // --- Stitch together: left | divider | right ---
    let total_w = half_w + divider_width + half_w;
    let mut data = Vec::with_capacity(total_w * h * 3);

    for row in 0..h {
        // Left pixels
        let left_start = row * half_w * 3;
        let left_end = left_start + half_w * 3;
        data.extend_from_slice(&left_buf.data[left_start..left_end]);

        // Divider
        for _ in 0..divider_width {
            data.push(DIVIDER_COLOR[0]);
            data.push(DIVIDER_COLOR[1]);
            data.push(DIVIDER_COLOR[2]);
        }

        // Right pixels
        let right_start = row * half_w * 3;
        let right_end = right_start + half_w * 3;
        data.extend_from_slice(&right_buf.data[right_start..right_end]);
    }

    RgbBuffer {
        width: total_w as u32,
        height: h as u32,
        data,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::games::mental_rotation::gen::generate;
    use rand::SeedableRng;
    use rand_chacha::ChaChaRng;

    #[test]
    fn render_correct_dimensions() {
        let mut rng = ChaChaRng::seed_from_u64(42);
        let state =
            generate(&mut rng, 9, 90, 180, 45, 5, 100_000).expect("test seed should generate");
        let buf = render(&state, 128, 100);
        assert_eq!(buf.data.len() as u32, buf.width * buf.height * 3);
    }

    #[test]
    fn render_has_nonbackground_pixels() {
        let mut rng = ChaChaRng::seed_from_u64(42);
        let state =
            generate(&mut rng, 9, 90, 180, 45, 5, 100_000).expect("test seed should generate");
        let buf = render(&state, 128, 100);
        let bg = BG;
        let non_bg = buf.data.chunks(3).filter(|p| *p != bg).count();
        assert!(
            non_bg > 50,
            "Should have visible polycube pixels, got {}",
            non_bg
        );
    }

    #[test]
    fn render_different_rotations_differ() {
        let mut rng = ChaChaRng::seed_from_u64(42);
        let state =
            generate(&mut rng, 9, 90, 180, 45, 5, 100_000).expect("test seed should generate");
        let buf1 = render(&state, 64, 100);

        let rotated = state.apply_action(super::super::state::RotationAction::RotateX(90));
        let buf2 = render(&rotated, 64, 100);
        assert_ne!(buf1.data, buf2.data);
    }
}
