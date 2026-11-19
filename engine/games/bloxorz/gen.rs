//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use rand::seq::SliceRandom;
use rand::Rng;
use rand_chacha::ChaChaRng;

use crate::base::game::Game;

use super::state::{BloxorzGrid, BloxorzState, Orientation, ALL_DIRECTIONS};
use super::Bloxorz;

/// Generate a solvable Bloxorz puzzle using BFS-from-goal.
///
/// 1. Build a random connected grid shape by flood-filling from a random goal cell.
/// 2. Place block Standing on goal and BFS outward over all directions.
/// 3. Pick the farthest reachable state as the start.
/// 4. Retry if puzzle is too trivial (solution length < `grid_size.min(12)`).
///
/// Generates multiple valid candidates and scores them on 4 metrics with
/// randomly sampled weights, creating diverse puzzle types:
/// - **Solution length**: raw number of moves (longer = harder)
/// - **Cell utilization**: fraction of board cells used by the solution path (higher = tighter)
/// - **Compactness**: solution length relative to board size (higher = more winding)
/// - **Orientation changes**: how often the block switches standing/lying (higher = more complex)
pub fn generate(
    rng: &mut ChaChaRng,
    grid_size: usize,
    solve_budget: usize,
    fill_range: (usize, usize),
    num_fragile_tiles: usize,
) -> Option<BloxorzState> {
    let min_solution_len = grid_size.min(12);

    // Sample random weights for this generation call.
    let w_depth: f64 = rng.gen();
    let w_utilization: f64 = rng.gen();
    let w_compactness: f64 = rng.gen();
    let w_orientation: f64 = rng.gen();

    // Collect up to `keep` valid candidates.
    let keep = 50;
    let mut candidates: Vec<(BloxorzState, usize, f64, f64, f64, Vec<Key>)> = Vec::new();

    for _ in 0..10_000 {
        if let Some((state, filled, depth, cells_on_path, orientation_changes, path)) =
            generate_candidate(rng, grid_size, min_solution_len, fill_range)
        {
            let utilization = cells_on_path as f64 / filled.max(1) as f64;
            let compactness = depth as f64 / filled.max(1) as f64;
            let orient_rate = orientation_changes as f64 / depth.max(1) as f64;
            candidates.push((state, depth, utilization, compactness, orient_rate, path));
            if candidates.len() >= keep {
                break;
            }
        }
    }

    if candidates.is_empty() {
        return None;
    }

    // Normalize depth to [0, 1] across candidates.
    let max_depth = candidates.iter().map(|c| c.1).max().unwrap_or(1).max(1);

    // Score each candidate and pick the best.
    let (state, _depth, _util, _compact, _orient, path) =
        candidates.into_iter().max_by(|a, b| {
            let score_a = w_depth * (a.1 as f64 / max_depth as f64)
                + w_utilization * a.2
                + w_compactness * a.3
                + w_orientation * a.4;
            let score_b = w_depth * (b.1 as f64 / max_depth as f64)
                + w_utilization * b.2
                + w_compactness * b.3
                + w_orientation * b.4;
            score_a
                .partial_cmp(&score_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        })?;

    if num_fragile_tiles == 0 {
        return Some(state);
    }

    place_fragile_tiles(rng, state, &path, num_fragile_tiles, solve_budget)
}

type Key = (usize, usize, Orientation);

/// Returns (state, filled_cell_count, bfs_depth, cells_on_path, orientation_changes, path) on success.
fn generate_candidate(
    rng: &mut ChaChaRng,
    grid_size: usize,
    min_solution_len: usize,
    fill_range: (usize, usize),
) -> Option<(BloxorzState, usize, usize, usize, usize, Vec<Key>)> {
    // Step 1: Build random grid shape via flood-fill from a random interior cell.
    let total = grid_size * grid_size;
    let target_fill = total * rng.gen_range(fill_range.0..=fill_range.1) / 100;

    let mut cells = vec![false; total];

    // Pick a random interior cell as the goal seed.
    let goal_col = rng.gen_range(1..grid_size - 1);
    let goal_row = rng.gen_range(1..grid_size - 1);
    cells[goal_row * grid_size + goal_col] = true;

    // Grow outward from the goal using randomized BFS.
    let mut frontier: Vec<(usize, usize)> = vec![(goal_col, goal_row)];
    let mut filled = 1usize;

    while filled < target_fill && !frontier.is_empty() {
        let idx = rng.gen_range(0..frontier.len());
        let (fc, fr) = frontier[idx];

        // Collect unvisited neighbors
        let mut neighbors = Vec::new();
        if fc > 0 && !cells[fr * grid_size + fc - 1] {
            neighbors.push((fc - 1, fr));
        }
        if fc + 1 < grid_size && !cells[fr * grid_size + fc + 1] {
            neighbors.push((fc + 1, fr));
        }
        if fr > 0 && !cells[(fr - 1) * grid_size + fc] {
            neighbors.push((fc, fr - 1));
        }
        if fr + 1 < grid_size && !cells[(fr + 1) * grid_size + fc] {
            neighbors.push((fc, fr + 1));
        }

        if neighbors.is_empty() {
            frontier.swap_remove(idx);
            continue;
        }

        let ni = rng.gen_range(0..neighbors.len());
        let (nc, nr) = neighbors[ni];
        cells[nr * grid_size + nc] = true;
        frontier.push((nc, nr));
        filled += 1;
    }

    let grid = Arc::new(BloxorzGrid {
        width: grid_size,
        height: grid_size,
        cells,
        goal: (goal_col, goal_row),
        start: (goal_col, goal_row, Orientation::Standing), // placeholder; overwritten after BFS
        fragile_tiles: vec![],
    });

    // Step 2: BFS from goal (block standing on goal).
    let start_state = BloxorzState {
        grid: grid.clone(),
        col: goal_col,
        row: goal_row,
        orientation: Orientation::Standing,
        broken_mask: 0,
    };

    // BFS: state key = (col, row, orientation). Track parents for path reconstruction.
    let mut visited: HashMap<Key, Option<Key>> = HashMap::new();
    let mut queue: VecDeque<(BloxorzState, usize)> = VecDeque::new();
    let mut farthest_state = start_state.clone();
    let mut farthest_key: Key = (start_state.col, start_state.row, start_state.orientation);
    let mut max_depth = 0usize;

    let start_key: Key = (start_state.col, start_state.row, start_state.orientation);
    visited.insert(start_key, None);
    queue.push_back((start_state, 0));

    while let Some((current, depth)) = queue.pop_front() {
        let current_key: Key = (current.col, current.row, current.orientation);
        if depth > max_depth {
            max_depth = depth;
            farthest_state = current.clone();
            farthest_key = current_key;
        }

        for dir in &ALL_DIRECTIONS {
            if let Some(next) = current.apply_direction(*dir) {
                let key: Key = (next.col, next.row, next.orientation);
                if !visited.contains_key(&key) {
                    visited.insert(key, Some(current_key));
                    queue.push_back((next, depth + 1));
                }
            }
        }
    }

    // Step 3: Require minimum solution length.
    if max_depth < min_solution_len {
        return None;
    }

    // Reconstruct optimal path: collect keys, count unique grid cells and orientation changes.
    let mut solution_path: Vec<Key> = Vec::new();
    let mut path_cells: HashSet<(usize, usize)> = HashSet::new();
    let mut orientation_changes = 0usize;
    let mut prev_orient: Option<Orientation> = None;
    let mut walk_key = farthest_key;
    loop {
        let (col, row, orient) = walk_key;
        solution_path.push(walk_key);
        path_cells.insert((col, row));
        match orient {
            Orientation::LyingX => {
                path_cells.insert((col + 1, row));
            }
            Orientation::LyingY => {
                path_cells.insert((col, row + 1));
            }
            Orientation::Standing => {}
        }
        if let Some(prev) = prev_orient {
            if prev != orient {
                orientation_changes += 1;
            }
        }
        prev_orient = Some(orient);
        match visited.get(&walk_key) {
            Some(Some(parent)) => walk_key = *parent,
            _ => break,
        }
    }
    let cells_on_path = path_cells.len();

    // Step 4: Return the farthest state as the starting position.
    // Rebuild grid with the actual start position set.
    let mut grid_data = (*grid).clone();
    grid_data.start = (
        farthest_state.col,
        farthest_state.row,
        farthest_state.orientation,
    );
    let grid = Arc::new(grid_data);
    Some((
        BloxorzState {
            grid,
            col: farthest_state.col,
            row: farthest_state.row,
            orientation: farthest_state.orientation,
            broken_mask: 0,
        },
        filled,
        max_depth,
        cells_on_path,
        orientation_changes,
        solution_path,
    ))
}

/// Place fragile tiles on a generated puzzle.
///
/// Identifies cells visited exactly once on the solution path (real fragile candidates)
/// and cells never visited (distractor candidates). Picks from visited-once first,
/// then fills remaining from distractors. Verifies solvability after placement.
fn place_fragile_tiles(
    rng: &mut ChaChaRng,
    state: BloxorzState,
    solution_path: &[Key],
    num_fragile: usize,
    solve_budget: usize,
) -> Option<BloxorzState> {
    let num_fragile = num_fragile.min(32);
    let grid = &state.grid;

    // Count how many times each cell is visited along the solution path.
    let mut visit_counts: HashMap<(usize, usize), usize> = HashMap::new();
    for &(col, row, orient) in solution_path {
        *visit_counts.entry((col, row)).or_insert(0) += 1;
        match orient {
            Orientation::LyingX => {
                *visit_counts.entry((col + 1, row)).or_insert(0) += 1;
            }
            Orientation::LyingY => {
                *visit_counts.entry((col, row + 1)).or_insert(0) += 1;
            }
            Orientation::Standing => {}
        }
    }

    // Cells occupied at the start position — exclude from fragile candidates.
    let start_cells: HashSet<(usize, usize)> = {
        let (sc, sr, so) = grid.start;
        let mut s = HashSet::new();
        s.insert((sc, sr));
        match so {
            Orientation::LyingX => {
                s.insert((sc + 1, sr));
            }
            Orientation::LyingY => {
                s.insert((sc, sr + 1));
            }
            Orientation::Standing => {}
        }
        s
    };

    let mut visited_once: Vec<(usize, usize)> = Vec::new();
    let mut never_visited: Vec<(usize, usize)> = Vec::new();

    for row in 0..grid.height {
        for col in 0..grid.width {
            if !grid.is_solid(col, row) {
                continue;
            }
            if (col, row) == grid.goal || start_cells.contains(&(col, row)) {
                continue;
            }
            let count = visit_counts.get(&(col, row)).copied().unwrap_or(0);
            if count == 1 {
                visited_once.push((col, row));
            } else if count == 0 {
                never_visited.push((col, row));
            }
        }
    }

    visited_once.shuffle(rng);
    never_visited.shuffle(rng);

    // Pick from visited-once first, fill remainder from distractors.
    let mut fragile_tiles: Vec<(usize, usize)> = Vec::new();
    let from_path = num_fragile.min(visited_once.len());
    fragile_tiles.extend_from_slice(&visited_once[..from_path]);
    let remaining = num_fragile - from_path;
    let from_distractor = remaining.min(never_visited.len());
    fragile_tiles.extend_from_slice(&never_visited[..from_distractor]);

    if fragile_tiles.is_empty() {
        return Some(state);
    }

    // Rebuild grid with fragile tiles.
    let mut grid_data = grid.as_ref().clone();
    grid_data.fragile_tiles = fragile_tiles;
    let new_grid = Arc::new(grid_data);
    let new_state = BloxorzState {
        grid: new_grid,
        col: state.col,
        row: state.row,
        orientation: state.orientation,
        broken_mask: 0,
    };

    // Verify solvability with fragile mechanics active.
    let game = Bloxorz::default();
    if game.solve(&new_state, solve_budget).is_some() {
        Some(new_state)
    } else {
        None
    }
}
