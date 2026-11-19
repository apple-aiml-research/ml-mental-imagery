//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
//! Sokoban object boxes (Boxes CoT): the grid cell `[column,row]` of the
//! `player`, of each `box_k` (in position order) and of each `goal_k`. A box on
//! a goal shares its cell and is told apart by name.

use crate::base::object_boxes::ObjectBox;

use super::state::SokobanState;

pub fn object_boxes(state: &SokobanState) -> Vec<ObjectBox> {
    let mut object_boxes =
        Vec::with_capacity(1 + state.box_positions.len() + state.layout.goal_positions.len());
    object_boxes.push((
        "player".to_string(),
        vec![state.player_col as i32, state.player_row as i32],
    ));
    for (box_index, &(column, row)) in state.box_positions.iter().enumerate() {
        object_boxes.push((format!("box_{box_index}"), vec![column as i32, row as i32]));
    }
    for (goal_index, &(column, row)) in state.layout.goal_positions.iter().enumerate() {
        object_boxes.push((
            format!("goal_{goal_index}"),
            vec![column as i32, row as i32],
        ));
    }
    object_boxes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::object_boxes::assert_color_within_boxes;
    use crate::games::sokoban::render::render;
    use crate::games::sokoban::state::from_ascii;

    // Fill colors in render.rs.
    const PLAYER_COLOR: [u8; 3] = [0x44, 0x44, 0xFF];
    const BOX_COLOR: [u8; 3] = [0x8B, 0x45, 0x13];
    const GOAL_COLOR: [u8; 3] = [0xFF, 0x44, 0x44];
    const BOARD: &str = "#####\n#@..#\n#.O.#\n#..X#\n#####";

    fn coordinates<'a>(object_boxes: &'a [ObjectBox], name: &str) -> &'a Vec<i32> {
        &object_boxes
            .iter()
            .find(|(object_name, _)| object_name == name)
            .expect("missing object")
            .1
    }

    #[test]
    fn lists_player_boxes_and_goals_by_cell() {
        let object_boxes = object_boxes(&from_ascii(BOARD));
        assert_eq!(
            object_boxes,
            vec![
                ("player".to_string(), vec![1, 1]),
                ("box_0".to_string(), vec![2, 2]),
                ("goal_0".to_string(), vec![3, 3]),
            ]
        );
    }

    #[test]
    fn cells_hold_the_drawn_objects() {
        // A 5x5 board at 200 px fills the image with 40 px cells.
        let state = from_ascii(BOARD);
        let object_boxes = object_boxes(&state);
        let (width, height) = (state.layout.width as i32, state.layout.height as i32);
        let cell_box = |cell: &[i32]| {
            vec![
                cell[0] * 100 / width,
                cell[1] * 100 / height,
                (cell[0] + 1) * 100 / width,
                (cell[1] + 1) * 100 / height,
            ]
        };
        let image = render(&state, 200);
        assert_color_within_boxes(
            &image,
            PLAYER_COLOR,
            &[cell_box(coordinates(&object_boxes, "player"))],
            1,
        );
        assert_color_within_boxes(
            &image,
            BOX_COLOR,
            &[cell_box(coordinates(&object_boxes, "box_0"))],
            1,
        );
        assert_color_within_boxes(
            &image,
            GOAL_COLOR,
            &[cell_box(coordinates(&object_boxes, "goal_0"))],
            1,
        );
    }
}
