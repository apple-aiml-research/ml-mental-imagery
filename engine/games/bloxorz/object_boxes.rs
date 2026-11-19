//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
//! Bloxorz object boxes (Boxes CoT): the scene is rendered in isometric 3D, so
//! each object's box is the screen extent of its 3D geometry projected through
//! the render's camera (`render::scene_camera`): the `block`, the `goal` tile
//! and each unbroken `fragile_k` tile (broken ones are void and not drawn).
//! The geometry matches render.rs.

use glam::Vec3;

use crate::base::object_boxes::{normalize_fraction, ObjectBox};
use crate::base::render3d::{project_to_screen, Camera};

use super::render::scene_camera;
use super::state::{BloxorzState, Orientation};

/// Square canvas the corners are projected onto before normalizing.
const PROJECTION_RESOLUTION: u32 = 1000;
/// Half of a block unit (render.rs).
const BLOCK_HALF_UNIT: f32 = 0.46;
/// Height of the top of the floor tiles (render.rs).
const TILE_TOP: f32 = 0.3;
const TILE_HALF_EXTENTS: Vec3 = Vec3::new(0.48, 0.15, 0.48);

/// Normalized `[x0,y0,x1,y1]` screen extent of an axis-aligned 3D box.
fn project_box(camera: &Camera, center: Vec3, half_extents: Vec3) -> Vec<i32> {
    let (mut min_x, mut min_y, mut max_x, mut max_y) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for sign_x in [-1.0f32, 1.0] {
        for sign_y in [-1.0f32, 1.0] {
            for sign_z in [-1.0f32, 1.0] {
                let corner = center
                    + Vec3::new(
                        sign_x * half_extents.x,
                        sign_y * half_extents.y,
                        sign_z * half_extents.z,
                    );
                let (screen_x, screen_y) =
                    project_to_screen(camera, corner, PROJECTION_RESOLUTION, PROJECTION_RESOLUTION);
                min_x = min_x.min(screen_x);
                max_x = max_x.max(screen_x);
                min_y = min_y.min(screen_y);
                max_y = max_y.max(screen_y);
            }
        }
    }
    [min_x, min_y, max_x, max_y]
        .iter()
        .map(|&pixel| normalize_fraction(pixel as f64 / PROJECTION_RESOLUTION as f64))
        .collect()
}

pub fn object_boxes(state: &BloxorzState) -> Vec<ObjectBox> {
    let camera = scene_camera(state);
    let mut object_boxes = Vec::new();

    let (block_x, block_z, block_half_extents) = match state.orientation {
        Orientation::Standing => (
            state.col as f32,
            state.row as f32,
            Vec3::new(BLOCK_HALF_UNIT, 2.0 * BLOCK_HALF_UNIT, BLOCK_HALF_UNIT),
        ),
        Orientation::LyingX => (
            state.col as f32 + 0.5,
            state.row as f32,
            Vec3::new(2.0 * BLOCK_HALF_UNIT, BLOCK_HALF_UNIT, BLOCK_HALF_UNIT),
        ),
        Orientation::LyingY => (
            state.col as f32,
            state.row as f32 + 0.5,
            Vec3::new(BLOCK_HALF_UNIT, BLOCK_HALF_UNIT, 2.0 * BLOCK_HALF_UNIT),
        ),
    };
    let block_center = Vec3::new(block_x, TILE_TOP + block_half_extents.y, block_z);
    object_boxes.push((
        "block".to_string(),
        project_box(&camera, block_center, block_half_extents),
    ));

    let (goal_column, goal_row) = state.grid.goal;
    let goal_center = Vec3::new(goal_column as f32, TILE_HALF_EXTENTS.y, goal_row as f32);
    object_boxes.push((
        "goal".to_string(),
        project_box(&camera, goal_center, TILE_HALF_EXTENTS),
    ));

    for (fragile_index, &(column, row)) in state.grid.fragile_tiles.iter().enumerate() {
        if state.broken_mask & (1 << fragile_index) != 0 {
            continue;
        }
        let tile_center = Vec3::new(column as f32, TILE_HALF_EXTENTS.y, row as f32);
        object_boxes.push((
            format!("fragile_{fragile_index}"),
            project_box(&camera, tile_center, TILE_HALF_EXTENTS),
        ));
    }
    object_boxes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::games::bloxorz::render::render;
    use crate::games::bloxorz::state::from_ascii;

    fn find<'a>(object_boxes: &'a [ObjectBox], name: &str) -> Option<&'a Vec<i32>> {
        object_boxes
            .iter()
            .find(|(object_name, _)| object_name == name)
            .map(|(_, coordinates)| coordinates)
    }

    #[test]
    fn lists_block_goal_and_unbroken_fragile_tiles() {
        let mut state = from_ascii(".F.\n.S.\n..G");
        let names: Vec<String> = object_boxes(&state)
            .into_iter()
            .map(|(name, _)| name)
            .collect();
        assert_eq!(names, vec!["block", "goal", "fragile_0"]);
        for (_, coordinates) in object_boxes(&state) {
            assert!(
                coordinates[0] < coordinates[2] && coordinates[1] < coordinates[3],
                "{coordinates:?}"
            );
        }
        state.broken_mask = 1;
        assert!(find(&object_boxes(&state), "fragile_0").is_none());
    }

    #[test]
    fn block_box_holds_the_drawn_block() {
        // The block is drawn on top, so its blue faces are visible.
        let state = from_ascii("......\n..S...\n....G.\n......");
        let block = find(&object_boxes(&state), "block").unwrap().clone();
        let image = render(&state, PROJECTION_RESOLUTION);
        let (width, height) = (image.width as i32, image.height as i32);
        let (mut sum_x, mut sum_y, mut count) = (0i64, 0i64, 0i64);
        for y in 0..height {
            for x in 0..width {
                let offset = ((y * width + x) * 3) as usize;
                let (red, green, blue) = (
                    image.data[offset] as i32,
                    image.data[offset + 1] as i32,
                    image.data[offset + 2] as i32,
                );
                if blue > 110 && blue > red + 40 && blue > green + 40 {
                    sum_x += x as i64;
                    sum_y += y as i64;
                    count += 1;
                }
            }
        }
        assert!(count > 50, "expected visible block pixels, got {count}");
        let (centroid_x, centroid_y) = ((sum_x / count) as i32, (sum_y / count) as i32);
        let tolerance = 12;
        assert!(
            centroid_x >= block[0] * width / 100 - tolerance
                && centroid_x <= block[2] * width / 100 + tolerance
                && centroid_y >= block[1] * height / 100 - tolerance
                && centroid_y <= block[3] * height / 100 + tolerance,
            "block centroid ({centroid_x},{centroid_y}) is outside {block:?}"
        );
    }
}
