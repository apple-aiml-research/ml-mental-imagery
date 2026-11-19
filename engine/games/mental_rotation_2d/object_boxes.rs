//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
//! 2D Mental Rotation object boxes (Boxes CoT): the `current` shape (left
//! panel, current rotation) and the `target` shape (right panel, target
//! rotation and flip), each as the box of the drawn polyomino on the whole
//! two-panel image. The framing matches render.rs: the shape's bounding radius
//! is fit to the panel width with `world_extent = max_radius * 2.5`, and the
//! panel is centred vertically.

use crate::base::object_boxes::{normalize_fraction, ObjectBox};

use super::state::MentalRotation2dState;

/// Width of the divider between the panels, as a fraction of the image width.
const DIVIDER_FRACTION: f32 = 1.0 / 64.0;

/// Normalized `[x0,y0,x1,y1]` of the polyomino drawn in the panel spanning
/// `panel_left..panel_right` (fractions of the image width) and the full height.
/// `panel_aspect` is the panel's width over the image height.
fn panel_box(
    cells: &[(i32, i32)],
    rotation_degrees: u16,
    flipped: bool,
    panel_left: f32,
    panel_right: f32,
    panel_aspect: f32,
) -> Vec<i32> {
    let count = cells.len() as f32;
    let center_x = cells.iter().map(|cell| cell.0 as f32).sum::<f32>() / count;
    let center_y = cells.iter().map(|cell| cell.1 as f32).sum::<f32>() / count;
    let mut max_radius = 0.0f32;
    for &(x, y) in cells {
        max_radius = max_radius
            .max(((x as f32 - center_x).abs() + 0.5).max((y as f32 - center_y).abs() + 0.5));
    }
    let world_extent = max_radius * 2.5;
    let angle = rotation_degrees as f32 * std::f32::consts::PI / 180.0;
    let (cosine, sine) = (angle.cos(), angle.sin());

    // Each cell is a unit square: transform its corners (rotate, then mirror x).
    let (mut left, mut top, mut right, mut bottom) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for &(cell_x, cell_y) in cells {
        for (corner_x, corner_y) in [(-0.5f32, -0.5f32), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)] {
            let offset_x = cell_x as f32 - center_x + corner_x;
            let offset_y = cell_y as f32 - center_y + corner_y;
            let mut world_x = cosine * offset_x - sine * offset_y;
            let world_y = sine * offset_x + cosine * offset_y;
            if flipped {
                world_x = -world_x;
            }
            let fraction_x = world_x / world_extent + 0.5;
            let fraction_y = world_y * panel_aspect / world_extent + 0.5;
            left = left.min(fraction_x);
            right = right.max(fraction_x);
            top = top.min(fraction_y);
            bottom = bottom.max(fraction_y);
        }
    }
    let panel_width = panel_right - panel_left;
    let normalize = |fraction: f32| normalize_fraction(fraction as f64);
    vec![
        normalize(panel_left + left.clamp(0.0, 1.0) * panel_width),
        normalize(top.clamp(0.0, 1.0)),
        normalize(panel_left + right.clamp(0.0, 1.0) * panel_width),
        normalize(bottom.clamp(0.0, 1.0)),
    ]
}

pub fn object_boxes(state: &MentalRotation2dState) -> Vec<ObjectBox> {
    let cells = &state.puzzle.polyomino.cells;
    let panel_width = (1.0 - DIVIDER_FRACTION) / 2.0;
    vec![
        (
            "current".to_string(),
            panel_box(cells, state.rotation, false, 0.0, panel_width, panel_width),
        ),
        (
            "target".to_string(),
            panel_box(
                cells,
                state.puzzle.target_rotation,
                state.puzzle.target_flipped,
                panel_width + DIVIDER_FRACTION,
                1.0,
                panel_width,
            ),
        ),
    ]
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_chacha::ChaChaRng;

    use super::*;
    use crate::games::mental_rotation_2d::gen::generate;
    use crate::games::mental_rotation_2d::render::render;

    const BACKGROUND_COLOR: [u8; 3] = [0x1A, 0x1A, 0x2E];
    const DIVIDER_COLOR: [u8; 3] = [0x44, 0x44, 0x66];

    fn sample_state() -> MentalRotation2dState {
        let mut rng = ChaChaRng::seed_from_u64(42);
        generate(&mut rng, 5, 90, 180, 180, 1, 100_000).expect("test seed should generate")
    }

    #[test]
    fn current_is_left_and_target_right() {
        let object_boxes = object_boxes(&sample_state());
        assert_eq!(object_boxes[0].0, "current");
        assert_eq!(object_boxes[1].0, "target");
        assert!(object_boxes[0].1[2] <= 50, "{:?}", object_boxes[0].1);
        assert!(object_boxes[1].1[0] >= 50, "{:?}", object_boxes[1].1);
    }

    #[test]
    fn boxes_hold_every_drawn_pixel() {
        // At 512 px the divider is exactly 1/64 of the width.
        let state = sample_state();
        let object_boxes = object_boxes(&state);
        let image = render(&state, 512);
        let (width, height) = (image.width as i32, image.height as i32);
        let tolerance = 3;
        let pixel_box = |coordinates: &Vec<i32>| {
            [
                coordinates[0] * width / 100 - tolerance,
                coordinates[1] * height / 100 - tolerance,
                coordinates[2] * width / 100 + tolerance,
                coordinates[3] * height / 100 + tolerance,
            ]
        };
        let pixel_boxes = [pixel_box(&object_boxes[0].1), pixel_box(&object_boxes[1].1)];
        let mut shape_pixels = 0;
        for y in 0..height {
            for x in 0..width {
                let offset = ((y * width + x) * 3) as usize;
                let pixel = [
                    image.data[offset],
                    image.data[offset + 1],
                    image.data[offset + 2],
                ];
                if pixel == BACKGROUND_COLOR || pixel == DIVIDER_COLOR {
                    continue;
                }
                shape_pixels += 1;
                assert!(
                    pixel_boxes.iter().any(|bounds| x >= bounds[0]
                        && x < bounds[2]
                        && y >= bounds[1]
                        && y < bounds[3]),
                    "shape pixel ({x},{y}) is outside both boxes {object_boxes:?}"
                );
            }
        }
        assert!(shape_pixels > 50, "too few shape pixels: {shape_pixels}");
    }
}
