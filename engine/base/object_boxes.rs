//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
//! Object boxes for Boxes CoT.
//!
//! Each game lists the task-relevant objects of a state with their positions in
//! the rendered image (`Game::object_boxes`), as integer coordinates normalized
//! to `0..=COORD_MAX` (origin top left, x to the right, y down). The number of
//! values per object depends on the game:
//! - 4 `[x0,y0,x1,y1]`: an image-space box (layout games);
//! - 6 `[x0,y0,z0,x1,y1,z1]`: a box in a fixed world frame (3D-rendered games);
//! - 2 `[column,row]`: the grid cell an object occupies (grid games);
//! - 1 `[slot]`: the slot a tile is in (Anagram).
//!
//! The episode writes the list as one compact JSON object between `<board>` and
//! `</board>`, e.g. `{"player":[6,4],"box_0":[2,6]}`.

use std::fmt::Write;

/// One object: its name and its normalized coordinates.
pub type ObjectBox = (String, Vec<i32>);

/// Largest normalized coordinate.
pub const COORD_MAX: i32 = 100;

/// Format objects as a compact JSON object, keeping their order: `{"name":[..],...}`.
pub fn format_object_boxes(object_boxes: &[ObjectBox]) -> String {
    let mut text = String::from("{");
    for (object_index, (name, coordinates)) in object_boxes.iter().enumerate() {
        if object_index > 0 {
            text.push(',');
        }
        let values: Vec<String> = coordinates.iter().map(|value| value.to_string()).collect();
        write!(text, "\"{name}\":[{}]", values.join(",")).unwrap();
    }
    text.push('}');
    text
}

/// Normalize a fraction of the image (0.0 to 1.0) to `0..=COORD_MAX`, clamped.
pub fn normalize_fraction(fraction: f64) -> i32 {
    (fraction * COORD_MAX as f64)
        .round()
        .clamp(0.0, COORD_MAX as f64) as i32
}

/// Normalized `[x0,y0,x1,y1]` of a pixel rectangle on a `canvas_width` x `canvas_height` image.
pub fn normalize_pixel_rect(
    canvas_width: u32,
    canvas_height: u32,
    x: f64,
    y: f64,
    width: f64,
    height: f64,
) -> [i32; 4] {
    let (canvas_width, canvas_height) = (canvas_width.max(1) as f64, canvas_height.max(1) as f64);
    [
        normalize_fraction(x / canvas_width),
        normalize_fraction(y / canvas_height),
        normalize_fraction((x + width) / canvas_width),
        normalize_fraction((y + height) / canvas_height),
    ]
}

/// Normalized `[x0,y0,z0,x1,y1,z1]` of a world-space box, mapping
/// `-reference_half_extent..=reference_half_extent` to `0..=COORD_MAX` on every axis.
pub fn normalize_world_box(
    minimum: [f32; 3],
    maximum: [f32; 3],
    reference_half_extent: f32,
) -> [i32; 6] {
    let normalize = |value: f32| {
        normalize_fraction((value / reference_half_extent.max(f32::EPSILON) * 0.5 + 0.5) as f64)
    };
    [
        normalize(minimum[0]),
        normalize(minimum[1]),
        normalize(minimum[2]),
        normalize(maximum[0]),
        normalize(maximum[1]),
        normalize(maximum[2]),
    ]
}

/// Test helper: every pixel of `color` lies inside one of the normalized `boxes`
/// (scaled to the image and grown by `tolerance` pixels), and at least one exists.
#[cfg(test)]
pub fn assert_color_within_boxes(
    image: &crate::base::game::RgbBuffer,
    color: [u8; 3],
    boxes: &[Vec<i32>],
    tolerance: i32,
) {
    let (width, height) = (image.width as i32, image.height as i32);
    let pixel_boxes: Vec<[i32; 4]> = boxes
        .iter()
        .map(|normalized| {
            [
                normalized[0] * width / COORD_MAX - tolerance,
                normalized[1] * height / COORD_MAX - tolerance,
                normalized[2] * width / COORD_MAX + tolerance,
                normalized[3] * height / COORD_MAX + tolerance,
            ]
        })
        .collect();
    let mut matching_pixels = 0usize;
    for y in 0..height {
        for x in 0..width {
            let offset = ((y * width + x) * 3) as usize;
            if image.data[offset..offset + 3] == color {
                matching_pixels += 1;
                let inside = pixel_boxes.iter().any(|pixel_box| {
                    x >= pixel_box[0] && x < pixel_box[2] && y >= pixel_box[1] && y < pixel_box[3]
                });
                assert!(
                    inside,
                    "pixel ({x},{y}) of color {color:?} is outside all boxes {boxes:?}"
                );
            }
        }
    }
    assert!(
        matching_pixels > 0,
        "no pixels of color {color:?}; boxes {boxes:?}"
    );
}

/// Test helper: the centroid of the pixels of `color` lies inside the normalized
/// box (scaled to the image and grown by `tolerance` pixels), and at least one exists.
#[cfg(test)]
pub fn assert_color_centroid_in_box(
    image: &crate::base::game::RgbBuffer,
    color: [u8; 3],
    normalized_box: &[i32],
    tolerance: i32,
) {
    let (width, height) = (image.width as i32, image.height as i32);
    let (mut sum_x, mut sum_y, mut count) = (0i64, 0i64, 0i64);
    for y in 0..height {
        for x in 0..width {
            let offset = ((y * width + x) * 3) as usize;
            if image.data[offset..offset + 3] == color {
                sum_x += x as i64;
                sum_y += y as i64;
                count += 1;
            }
        }
    }
    assert!(count > 0, "no pixels of color {color:?}");
    let (centroid_x, centroid_y) = ((sum_x / count) as i32, (sum_y / count) as i32);
    let left = normalized_box[0] * width / COORD_MAX - tolerance;
    let top = normalized_box[1] * height / COORD_MAX - tolerance;
    let right = normalized_box[2] * width / COORD_MAX + tolerance;
    let bottom = normalized_box[3] * height / COORD_MAX + tolerance;
    assert!(
        centroid_x >= left && centroid_x <= right && centroid_y >= top && centroid_y <= bottom,
        "centroid ({centroid_x},{centroid_y}) of color {color:?} is outside box {normalized_box:?} \
         (pixels [{left},{top},{right},{bottom}])"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_objects_in_order() {
        assert_eq!(format_object_boxes(&[]), "{}");
        let object_boxes = vec![
            ("target".to_string(), vec![29, 9, 29, 70, 91, 70]),
            ("current".to_string(), vec![9, 29]),
            ("tile".to_string(), vec![3]),
        ];
        assert_eq!(
            format_object_boxes(&object_boxes),
            r#"{"target":[29,9,29,70,91,70],"current":[9,29],"tile":[3]}"#
        );
    }

    #[test]
    fn normalizes_pixel_rectangles() {
        assert_eq!(
            normalize_pixel_rect(256, 256, 64.0, 64.0, 128.0, 128.0),
            [25, 25, 75, 75]
        );
        assert_eq!(
            normalize_pixel_rect(256, 256, -10.0, 0.0, 300.0, 256.0),
            [0, 0, 100, 100]
        );
    }

    #[test]
    fn normalizes_world_boxes_around_the_origin() {
        assert_eq!(
            normalize_world_box([-2.0, 0.0, -1.0], [2.0, 0.0, 1.0], 2.0),
            [0, 50, 25, 100, 50, 75]
        );
    }
}
