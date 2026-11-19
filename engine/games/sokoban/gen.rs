//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
use std::sync::Arc;

use rand::Rng;
use rand_chacha::ChaChaRng;

use super::state::{BoardLayout, Cell, SokobanState};

/// Generate a candidate Sokoban puzzle.
///
/// 1. Pick num_boxes from range
/// 2. Generate border walls, random interior walls (~20% density)
/// 3. Place player, boxes, goals on distinct floor cells
/// 4. Quick-reject corner deadlocks (box in corner not on goal)
fn generate_candidate(
    rng: &mut ChaChaRng,
    grid_size: usize,
    num_boxes_range: (usize, usize),
    wall_density: u32,
) -> Option<SokobanState> {
    let num_boxes = rng.gen_range(num_boxes_range.0..=num_boxes_range.1);

    // Build grid with border walls and random interior walls
    let mut grid = vec![Cell::Floor; grid_size * grid_size];
    for row in 0..grid_size {
        for col in 0..grid_size {
            if row == 0 || row == grid_size - 1 || col == 0 || col == grid_size - 1 {
                grid[row * grid_size + col] = Cell::Wall;
            } else if rng.gen::<f64>() < wall_density as f64 / 100.0 {
                grid[row * grid_size + col] = Cell::Wall;
            }
        }
    }

    // Collect floor cells
    let mut floor_cells: Vec<(usize, usize)> = Vec::new();
    for row in 1..grid_size - 1 {
        for col in 1..grid_size - 1 {
            if grid[row * grid_size + col] == Cell::Floor {
                floor_cells.push((col, row));
            }
        }
    }

    // Need at least 1 (player) + num_boxes (boxes) + num_boxes (goals) distinct cells
    let needed = 1 + num_boxes * 2;
    if floor_cells.len() < needed {
        return None;
    }

    // Shuffle and pick positions
    // Fisher-Yates shuffle on needed portion
    for i in 0..needed.min(floor_cells.len()) {
        let j = rng.gen_range(i..floor_cells.len());
        floor_cells.swap(i, j);
    }

    let player_pos = floor_cells[0];
    let box_positions: Vec<(usize, usize)> = floor_cells[1..1 + num_boxes].to_vec();
    let goal_positions: Vec<(usize, usize)> =
        floor_cells[1 + num_boxes..1 + 2 * num_boxes].to_vec();

    // Mark goal cells in grid
    for &(col, row) in &goal_positions {
        grid[row * grid_size + col] = Cell::Goal;
    }

    // Quick-reject: corner deadlock (box in corner not on goal)
    for &(box_col, box_row) in &box_positions {
        if goal_positions.contains(&(box_col, box_row)) {
            continue;
        }
        let wall_up = box_row == 0 || grid[(box_row - 1) * grid_size + box_col] == Cell::Wall;
        let wall_down =
            box_row + 1 >= grid_size || grid[(box_row + 1) * grid_size + box_col] == Cell::Wall;
        let wall_left = box_col == 0 || grid[box_row * grid_size + box_col - 1] == Cell::Wall;
        let wall_right =
            box_col + 1 >= grid_size || grid[box_row * grid_size + box_col + 1] == Cell::Wall;

        if (wall_up || wall_down) && (wall_left || wall_right) {
            return None;
        }
    }

    let mut sorted_goals = goal_positions;
    sorted_goals.sort();

    let layout = Arc::new(BoardLayout {
        width: grid_size,
        height: grid_size,
        grid,
        goal_positions: sorted_goals,
    });

    Some(SokobanState::new(
        layout,
        player_pos.0,
        player_pos.1,
        box_positions,
    ))
}

/// Generate a solvable Sokoban puzzle.
///
/// Uses generate-then-verify: generates candidate puzzles and checks
/// solvability via BFS. Retries up to 1000 times. Returns None if all fail.
pub fn generate(
    rng: &mut ChaChaRng,
    grid_size: usize,
    num_boxes_range: (usize, usize),
    solve_budget: usize,
    wall_density: u32,
) -> Option<SokobanState> {
    for _ in 0..1000 {
        let candidate = match generate_candidate(rng, grid_size, num_boxes_range, wall_density) {
            Some(c) => c,
            None => continue,
        };

        // Verify solvability with the BFS solver of a temporary Sokoban game
        let game = super::Sokoban {
            grid_size_range: (grid_size, grid_size),
            num_boxes_range,
            solve_budget,
            wall_density,
        };

        if crate::solver::bfs::bfs_solve(&game, &candidate, solve_budget).is_some() {
            return Some(candidate);
        }
    }

    None
}
