//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
//! Jigsaw object boxes (Boxes CoT): one box per piece, named by the number
//! drawn on it (`p0`, `p1`, ...): its grid cell once placed, otherwise its tray
//! slot. The tabs overhanging the square are not included.
//!
//! The boxes use the tray layout the renderer cached at its first render,
//! normalized by the resolution it was packed for. Recomputing the layout at
//! another resolution would be wrong: the packer is not scale-invariant.

use crate::base::object_boxes::{normalize_pixel_rect, ObjectBox};

use super::state::JigsawState;

pub fn object_boxes(state: &JigsawState) -> Vec<ObjectBox> {
    let grid_size = state.puzzle.grid_size;
    let (layout_resolution, cell_size, tray_positions) = state
        .puzzle
        .tray_layout
        .get()
        .expect("Jigsaw object boxes need the tray layout of a rendered state");
    let cell_size = *cell_size as f64;

    let mut grid_cell_of_piece: Vec<Option<usize>> = vec![None; grid_size * grid_size];
    for (cell_index, cell) in state.grid.iter().enumerate() {
        if let Some(piece_id) = *cell {
            grid_cell_of_piece[piece_id] = Some(cell_index);
        }
    }

    (0..grid_size * grid_size)
        .map(|piece_id| {
            let (x, y) = match grid_cell_of_piece[piece_id] {
                Some(cell_index) => (
                    (cell_index % grid_size) as f64 * cell_size,
                    (cell_index / grid_size) as f64 * cell_size,
                ),
                None => {
                    let slot = state
                        .puzzle
                        .tray_order
                        .iter()
                        .position(|&tray_piece| tray_piece == piece_id)
                        .expect("every piece has a tray slot");
                    let (slot_x, slot_y) = tray_positions[slot];
                    (slot_x as f64, slot_y as f64)
                }
            };
            let coordinates = normalize_pixel_rect(
                *layout_resolution,
                *layout_resolution,
                x,
                y,
                cell_size,
                cell_size,
            );
            (format!("p{piece_id}"), coordinates.to_vec())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_chacha::ChaChaRng;

    use super::*;
    use crate::games::jigsaw::gen::generate;
    use crate::games::jigsaw::render::{compute_tray_layout, render};

    const RESOLUTION: u32 = 256;
    /// Fill of an empty grid cell (render.rs).
    const EMPTY_CELL_COLOR: [u8; 3] = [0xD0, 0xD0, 0xD0];

    #[test]
    fn placed_piece_is_boxed_at_its_grid_cell() {
        let mut state = generate(3, &mut ChaChaRng::seed_from_u64(0));
        // Place piece 0 at column 1, row 1.
        state.grid[4] = Some(0);
        state.unplaced.retain(|&piece_id| piece_id != 0);
        let image = render(&state, RESOLUTION);
        let (layout_resolution, cell_size, _) = state.puzzle.tray_layout.get().unwrap().clone();
        assert_eq!(layout_resolution, RESOLUTION);

        let object_boxes = object_boxes(&state);
        let names: Vec<String> = object_boxes.iter().map(|(name, _)| name.clone()).collect();
        assert_eq!(
            names,
            (0..9)
                .map(|piece_id| format!("p{piece_id}"))
                .collect::<Vec<_>>()
        );
        let size = cell_size as f64;
        assert_eq!(
            object_boxes[0].1,
            normalize_pixel_rect(RESOLUTION, RESOLUTION, size, size, size, size).to_vec()
        );

        // The placed cell is covered by the piece; an empty cell shows the empty fill.
        let cell_size = cell_size as i32;
        let empty_pixels_in_cell = |column: i32, row: i32| {
            let mut count = 0;
            for y in (row * cell_size + 2)..((row + 1) * cell_size - 2) {
                for x in (column * cell_size + 2)..((column + 1) * cell_size - 2) {
                    let offset = ((y * image.width as i32 + x) * 3) as usize;
                    if image.data[offset..offset + 3] == EMPTY_CELL_COLOR {
                        count += 1;
                    }
                }
            }
            count
        };
        assert!(empty_pixels_in_cell(1, 1) < empty_pixels_in_cell(0, 0) / 2);
    }

    #[test]
    fn tray_boxes_follow_the_rendered_layout() {
        let state = generate(3, &mut ChaChaRng::seed_from_u64(0));
        render(&state, RESOLUTION);
        let (_, cell_size, tray_positions) = state.puzzle.tray_layout.get().unwrap().clone();
        let normalized_slot = |resolution: u32, position: (u32, u32), size: u32| {
            normalize_pixel_rect(
                resolution,
                resolution,
                position.0 as f64,
                position.1 as f64,
                size as f64,
                size as f64,
            )
        };
        // Precondition: the layout packed at another resolution differs once normalized.
        let (cell_size_1000, tray_positions_1000) = compute_tray_layout(3, 1000);
        assert_ne!(
            normalized_slot(RESOLUTION, tray_positions[0], cell_size),
            normalized_slot(1000, tray_positions_1000[0], cell_size_1000)
        );
        for (name, coordinates) in object_boxes(&state) {
            let piece_id: usize = name.strip_prefix('p').unwrap().parse().unwrap();
            let slot = state
                .puzzle
                .tray_order
                .iter()
                .position(|&tray_piece| tray_piece == piece_id)
                .unwrap();
            assert_eq!(
                coordinates,
                normalized_slot(RESOLUTION, tray_positions[slot], cell_size).to_vec(),
                "{name}"
            );
        }
    }

    #[test]
    #[should_panic(expected = "rendered state")]
    fn requires_a_rendered_state() {
        object_boxes(&generate(3, &mut ChaChaRng::seed_from_u64(0)));
    }
}
