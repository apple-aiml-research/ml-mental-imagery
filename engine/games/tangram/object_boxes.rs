//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
//! Tangram object boxes (Boxes CoT; also the yes/no variant): one box per
//! piece, named by its colour and the label drawn on it (`blue_0`, `red_1`,
//! ...): the extent of its board cells once placed, otherwise its tray
//! rectangle.
//!
//! The boxes use the tray layout the renderer cached at its first render, and
//! the board origin at that resolution. Recomputing the layout at another
//! resolution would be wrong: the packer is not scale-invariant.

use crate::base::object_boxes::{normalize_pixel_rect, ObjectBox};

use super::TangramState;

/// Colour names in the order of `render.rs::TANGRAM_PIECE_COLORS`.
const COLOR_NAMES: [&str; 12] = [
    "blue", "red", "green", "yellow", "purple", "orange", "cyan", "magenta", "olive", "navy",
    "brown", "lime",
];

/// `<colour>_<label>`: the label is the digit (P0-P9) or letter (P10 on: A, B, ...) drawn on the piece.
fn piece_name(piece_id: &str, color_map: &[usize]) -> String {
    let piece_index: usize = piece_id
        .strip_prefix('P')
        .and_then(|number| number.parse().ok())
        .unwrap_or(0);
    let palette_index = if piece_index < color_map.len() {
        color_map[piece_index]
    } else {
        piece_index
    };
    let color = COLOR_NAMES[palette_index % COLOR_NAMES.len()];
    let label = if piece_index < 10 {
        (b'0' + piece_index as u8) as char
    } else {
        (b'A' + (piece_index - 10) as u8) as char
    };
    format!("{color}_{label}")
}

/// Pixel extent `(x, y, width, height)` of `cells` drawn at `(origin_x, origin_y)` with square cells.
fn cells_extent(
    cells: &[(i32, i32)],
    origin_x: f64,
    origin_y: f64,
    cell_size: f64,
) -> (f64, f64, f64, f64) {
    let min_x = cells.iter().map(|cell| cell.0).min().unwrap();
    let max_x = cells.iter().map(|cell| cell.0).max().unwrap();
    let min_y = cells.iter().map(|cell| cell.1).min().unwrap();
    let max_y = cells.iter().map(|cell| cell.1).max().unwrap();
    (
        origin_x + min_x as f64 * cell_size,
        origin_y + min_y as f64 * cell_size,
        (max_x - min_x + 1) as f64 * cell_size,
        (max_y - min_y + 1) as f64 * cell_size,
    )
}

pub fn object_boxes(state: &TangramState) -> Vec<ObjectBox> {
    let (layout_resolution, cell_size, tray_positions) = state
        .puzzle
        .tray_layout
        .get()
        .expect("Tangram object boxes need the tray layout of a rendered state");
    let layout_resolution = *layout_resolution;
    let cell_size = *cell_size as f64;
    // Board origin as in render.rs, at the layout's resolution.
    let margin = (layout_resolution as f32 * 0.02) as u32;
    let label_gap = ((layout_resolution as f32 * 0.04).max(8.0) * 1.2) as u32;
    let board_origin = (margin + label_gap) as f64;

    let mut object_boxes = Vec::with_capacity(state.puzzle.pieces.len());
    for (piece_index, piece) in state.puzzle.pieces.iter().enumerate() {
        let name = piece_name(&piece.id, &state.puzzle.color_map);
        let (x, y, width, height) = if let Some(placement) = state
            .placed
            .iter()
            .find(|placement| placement.piece_id == piece.id)
        {
            cells_extent(
                &placement.occupied_cells(piece),
                board_origin,
                board_origin,
                cell_size,
            )
        } else if state.remaining_piece_ids.contains(&piece.id) {
            let (slot_x, slot_y) = tray_positions[piece_index];
            let (_, _, width, height) = cells_extent(
                &piece.rotated_cells(state.queue_rotation(&piece.id)),
                0.0,
                0.0,
                cell_size,
            );
            (slot_x as f64, slot_y as f64, width, height)
        } else {
            continue;
        };
        let coordinates =
            normalize_pixel_rect(layout_resolution, layout_resolution, x, y, width, height);
        object_boxes.push((name, coordinates.to_vec()));
    }
    object_boxes
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::sync::Arc;

    use super::super::render::render;
    use super::super::{PieceDef, Placement, TangramPuzzle};
    use super::*;
    use crate::base::object_boxes::assert_color_centroid_in_box;

    // The first three palette colours (render.rs).
    const BLUE: [u8; 3] = [0x44, 0x44, 0xFF];
    const RED: [u8; 3] = [0xFF, 0x44, 0x44];
    const GREEN: [u8; 3] = [0x44, 0xCC, 0x44];
    const RESOLUTION: u32 = 256;

    /// A 3x2 board with three dominoes: P0 placed, P1 and P2 in the tray.
    fn state_with_first_piece_placed() -> TangramState {
        let domino = |id: &str| PieceDef {
            id: id.to_string(),
            cells: vec![(0, 0), (1, 0)],
        };
        let puzzle = Arc::new(TangramPuzzle {
            canvas_width: 3,
            canvas_height: 2,
            pieces: vec![domino("P0"), domino("P1"), domino("P2")],
            target_set: [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)]
                .into_iter()
                .collect::<HashSet<_>>(),
            color_map: vec![0, 1, 2],
            target_placements: vec![],
            solvable: true,
            tray_layout: std::sync::OnceLock::new(),
        });
        TangramState {
            puzzle,
            placed: vec![Placement {
                piece_id: "P0".into(),
                x: 0,
                y: 0,
                rotation: 0,
            }],
            remaining_piece_ids: vec!["P1".to_string(), "P2".to_string()],
            queue_rotations: vec![("P1".to_string(), 0), ("P2".to_string(), 0)],
            occupied_cells: [(0, 0), (1, 0)].into_iter().collect(),
        }
    }

    fn coordinates<'a>(object_boxes: &'a [ObjectBox], name: &str) -> &'a Vec<i32> {
        &object_boxes
            .iter()
            .find(|(object_name, _)| object_name == name)
            .unwrap()
            .1
    }

    #[test]
    fn boxes_hold_the_drawn_pieces() {
        let state = state_with_first_piece_placed();
        let image = render(&state, RESOLUTION);
        let object_boxes = object_boxes(&state);
        let names: Vec<&str> = object_boxes.iter().map(|(name, _)| name.as_str()).collect();
        assert_eq!(names, vec!["blue_0", "red_1", "green_2"]);
        assert_color_centroid_in_box(&image, BLUE, coordinates(&object_boxes, "blue_0"), 6);
        assert_color_centroid_in_box(&image, RED, coordinates(&object_boxes, "red_1"), 6);
        assert_color_centroid_in_box(&image, GREEN, coordinates(&object_boxes, "green_2"), 6);
    }

    #[test]
    fn boxes_follow_the_rendered_layout() {
        let state_256 = state_with_first_piece_placed();
        let state_1000 = state_with_first_piece_placed();
        render(&state_256, RESOLUTION);
        render(&state_1000, 1000);
        assert_eq!(state_256.puzzle.tray_layout.get().unwrap().0, RESOLUTION);
        // Precondition: the layouts differ once normalized.
        assert_ne!(object_boxes(&state_256), object_boxes(&state_1000));
    }
}
