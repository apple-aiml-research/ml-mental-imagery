//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
//! Character Recognition object boxes (Boxes CoT): one box per piece, named by
//! the number drawn on it (`p0`, `p1`, ...). A placed piece is boxed where it
//! is drawn in the assembly grid: its home cells when it is in its matching
//! slot, otherwise its cells rotated onto the chosen slot's centroid (as in
//! render.rs). Boxing it at its home cells regardless of the slot would leak
//! the piece-to-slot solution. An unplaced piece is boxed in its palette
//! section. The layout (assembly in the top half, palette in the bottom) follows
//! render.rs, evaluated at `LAYOUT_RESOLUTION`.

use crate::base::object_boxes::{normalize_pixel_rect, ObjectBox};

use super::state::CharacterRecognitionState;

/// Resolution the render layout is evaluated at before normalizing.
const LAYOUT_RESOLUTION: u32 = 1000;

/// Assembly grid of render.rs at `LAYOUT_RESOLUTION`: `(cell_size, grid_x0, grid_y0)` in pixels.
fn assembly_grid(grid_size: u32) -> (u32, u32, u32) {
    let resolution = LAYOUT_RESOLUTION;
    let assembly_height = resolution / 2;
    let margin = resolution / 14;
    let available_width = resolution.saturating_sub(2 * margin);
    let available_height = assembly_height.saturating_sub(2 * margin);
    let cell_size = (available_width / grid_size)
        .min(available_height / grid_size)
        .max(1);
    let grid_pixels = cell_size * grid_size;
    (
        cell_size,
        (resolution - grid_pixels) / 2,
        (assembly_height - grid_pixels) / 2,
    )
}

pub fn object_boxes(state: &CharacterRecognitionState) -> Vec<ObjectBox> {
    let puzzle = &state.puzzle;
    let resolution = LAYOUT_RESOLUTION;
    let (cell_size, grid_x0, grid_y0) = assembly_grid(puzzle.grid_size as u32);
    let cell_size = cell_size as f64;
    let palette_top = resolution * 55 / 100;
    let palette_height = resolution - palette_top;
    let section_width = resolution / puzzle.num_pieces as u32;
    let label_space = (palette_height as f64 * 0.18) as u32;
    let palette_center_y = (palette_top + label_space + (palette_height - label_space) / 2) as f64;

    let mut object_boxes = Vec::with_capacity(puzzle.num_pieces);
    for (piece_index, piece) in puzzle.pieces.iter().enumerate().take(puzzle.num_pieces) {
        if piece.cells.is_empty() {
            continue; // render.rs draws nothing for an empty piece
        }
        let (x, y, width, height) = if let Some(slot_index) = state.placements[piece_index] {
            let drawn_cells = if state.piece_correct(piece_index) {
                piece.cells.clone()
            } else {
                piece.rendered_cells(
                    state.rotations[piece_index],
                    puzzle.slots[slot_index].centroid,
                )
            };
            if drawn_cells.is_empty() {
                continue;
            }
            let min_x = drawn_cells.iter().map(|cell| cell.0).min().unwrap();
            let max_x = drawn_cells.iter().map(|cell| cell.0).max().unwrap();
            let min_y = drawn_cells.iter().map(|cell| cell.1).min().unwrap();
            let max_y = drawn_cells.iter().map(|cell| cell.1).max().unwrap();
            (
                grid_x0 as f64 + min_x as f64 * cell_size,
                grid_y0 as f64 + min_y as f64 * cell_size,
                (max_x - min_x + 1) as f64 * cell_size,
                (max_y - min_y + 1) as f64 * cell_size,
            )
        } else {
            // Centred in its palette section, sized by its rotated extent.
            let piece_width = (piece.cells.iter().map(|cell| cell.0).max().unwrap()
                - piece.cells.iter().map(|cell| cell.0).min().unwrap()
                + 1) as f64;
            let piece_height = (piece.cells.iter().map(|cell| cell.1).max().unwrap()
                - piece.cells.iter().map(|cell| cell.1).min().unwrap()
                + 1) as f64;
            let (rotated_width, rotated_height) = if state.rotations[piece_index] % 2 == 0 {
                (piece_width, piece_height)
            } else {
                (piece_height, piece_width)
            };
            let (drawn_width, drawn_height) =
                (rotated_width * cell_size, rotated_height * cell_size);
            let section_center_x = (piece_index as u32 * section_width + section_width / 2) as f64;
            (
                section_center_x - drawn_width / 2.0,
                palette_center_y - drawn_height / 2.0,
                drawn_width,
                drawn_height,
            )
        };
        object_boxes.push((
            format!("p{piece_index}"),
            normalize_pixel_rect(resolution, resolution, x, y, width, height).to_vec(),
        ));
    }
    object_boxes
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_chacha::ChaChaRng;

    use super::*;
    use crate::base::game::Game;
    use crate::games::character_recognition::CharacterRecognition;

    fn sample_state() -> CharacterRecognitionState {
        let game = CharacterRecognition {
            num_pieces_range: (3, 4),
            grid_size_range: (5, 5),
            slot_grid_dim: 8,
            solve_budget: 100_000,
        };
        game.generate(&mut ChaChaRng::seed_from_u64(1))
            .expect("generate")
    }

    fn piece_box(state: &CharacterRecognitionState, piece_index: usize) -> Vec<i32> {
        object_boxes(state)
            .into_iter()
            .find(|(name, _)| *name == format!("p{piece_index}"))
            .expect("box")
            .1
    }

    /// Index of the first piece that is drawn (empty pieces have no box).
    fn first_drawn_piece(state: &CharacterRecognitionState) -> usize {
        object_boxes(state)[0]
            .0
            .strip_prefix('p')
            .unwrap()
            .parse()
            .unwrap()
    }

    #[test]
    fn unplaced_pieces_are_in_the_palette_and_placed_ones_in_the_assembly() {
        let mut state = sample_state();
        for (_, coordinates) in object_boxes(&state) {
            assert!(coordinates[1] >= 55, "{coordinates:?}");
        }
        let piece_index = first_drawn_piece(&state);
        state.placements[piece_index] = Some(0);
        assert!(piece_box(&state, piece_index)[3] <= 50);
    }

    #[test]
    fn placed_box_follows_the_chosen_slot() {
        // Boxing a piece at its home cells for every slot would leak the solution.
        let state = sample_state();
        let piece_index = first_drawn_piece(&state);
        let box_in_slot = |slot_index: usize| {
            let mut placed = state.clone();
            placed.placements[piece_index] = Some(slot_index);
            piece_box(&placed, piece_index)
        };
        let first = box_in_slot(0);
        assert!((1..state.puzzle.slots.len()).any(|slot_index| box_in_slot(slot_index) != first));
    }

    #[test]
    fn misplaced_box_spans_the_drawn_cells_over_the_chosen_slot() {
        let state = sample_state();
        let piece_index = first_drawn_piece(&state);
        let (cell_size, grid_x0, grid_y0) = assembly_grid(state.puzzle.grid_size as u32);
        let mut checked_slots = 0;
        for slot_index in 0..state.puzzle.slots.len() {
            if state.puzzle.slots[slot_index].matching_piece_idx == Some(piece_index) {
                continue;
            }
            let mut placed = state.clone();
            placed.placements[piece_index] = Some(slot_index);
            let coordinates = piece_box(&placed, piece_index);
            let drawn_cells = placed.puzzle.pieces[piece_index].rendered_cells(
                placed.rotations[piece_index],
                placed.puzzle.slots[slot_index].centroid,
            );
            let middle = |values: Vec<i32>| {
                (values.iter().min().unwrap() + values.iter().max().unwrap()) as f64 / 2.0
            };
            let middle_x = middle(drawn_cells.iter().map(|cell| cell.0).collect());
            let middle_y = middle(drawn_cells.iter().map(|cell| cell.1).collect());
            let normalized_x = (grid_x0 as f64 + (middle_x + 0.5) * cell_size as f64) * 100.0
                / LAYOUT_RESOLUTION as f64;
            let normalized_y = (grid_y0 as f64 + (middle_y + 0.5) * cell_size as f64) * 100.0
                / LAYOUT_RESOLUTION as f64;
            assert!(
                coordinates[0] as f64 <= normalized_x + 1.0
                    && normalized_x <= coordinates[2] as f64 + 1.0
            );
            assert!(
                coordinates[1] as f64 <= normalized_y + 1.0
                    && normalized_y <= coordinates[3] as f64 + 1.0
            );
            checked_slots += 1;
        }
        assert!(checked_slots > 1);
    }
}
