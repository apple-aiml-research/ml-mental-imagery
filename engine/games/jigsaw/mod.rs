//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
pub mod gen;
pub mod object_boxes;
pub mod render;
pub mod state;

use rand::Rng;
use rand_chacha::ChaChaRng;

use crate::base::game::{Game, RgbBuffer};
use crate::base::state_text::{json_numbers, json_optional_numbers, StateField};
use crate::base::template::{ActionTemplate, ParamType};

use state::{cell_label, min_rotations, parse_cell_label, target_cell, JigsawAction, JigsawState};

// ---------------------------------------------------------------------------
// Game struct
// ---------------------------------------------------------------------------

pub struct Jigsaw {
    pub grid_size_range: (usize, usize),
    pub solve_budget: usize,
}

#[cfg(test)]
impl Default for Jigsaw {
    fn default() -> Self {
        Jigsaw {
            grid_size_range: (3, 4),
            solve_budget: 100_000,
        }
    }
}

// ---------------------------------------------------------------------------
// Factory + registry
// ---------------------------------------------------------------------------

const SOLVE_BUDGET: usize = 100_000;

pub fn build_factory(
    name: &str,
    params: &[super::registry::ConfigParam],
    overrides: &std::collections::HashMap<String, super::registry::ConfigValue>,
) -> Result<Box<dyn crate::stream::loader::GameFactory>, String> {
    let (lo, hi) = super::registry::resolve_int_range(params, overrides, "grid_size_range")?;
    if lo < 2 {
        return Err("grid_size_range min must be >= 2".to_string());
    }
    let game = Jigsaw {
        grid_size_range: (lo as usize, hi as usize),
        solve_budget: SOLVE_BUDGET,
    };
    Ok(crate::stream::loader::game_factory(
        game,
        name,
        SOLVE_BUDGET,
    ))
}

inventory::submit! {
    super::registry::GameEntry {
        name: "jigsaw",
        params: &[
            super::registry::ConfigParam {
                key: "grid_size_range",
                param_type: super::registry::ConfigParamType::IntRange,
                default: super::registry::ConfigValue::IntRange(3, 4),
            },
        ],
        build: build_factory,
    }
}

// ---------------------------------------------------------------------------
// Orientation helpers
// ---------------------------------------------------------------------------

/// Total cost to solve toward global orientation `theta`. Every orientation is
/// reachable because placed pieces can be removed.
fn orientation_cost(state: &JigsawState, theta: i32) -> usize {
    let n = state.puzzle.grid_size;
    let total = n * n;
    let mut cost = 0usize;

    for piece_id in 0..total {
        let target = target_cell(piece_id, n, theta);
        let target_rot = ((theta % 360) + 360) % 360;
        let rot_cost = min_rotations(state.rotations[piece_id], target_rot);

        if state.unplaced.contains(&piece_id) {
            // Needs rotation + place.
            cost += rot_cost + 1;
        } else {
            // Piece is on the grid — find where.
            let current_cell = state.grid.iter().position(|c| *c == Some(piece_id));
            if current_cell == Some(target) {
                // Correct cell, just needs rotation (possibly 0).
                cost += rot_cost;
            } else {
                // Wrong cell: remove + rotation + place.
                cost += rot_cost + 2;
            }
        }
    }
    cost
}

/// Pick the cheapest global orientation (0, 90, 180, 270) for the current state.
/// Returns (theta, cost).
fn best_orientation(state: &JigsawState) -> (i32, usize) {
    [0, 90, 180, 270]
        .iter()
        .map(|&theta| (theta, orientation_cost(state, theta)))
        .min_by_key(|&(_, cost)| cost)
        .unwrap()
}

/// Check whether the state is solved for a specific global orientation.
fn is_solved_for_theta(state: &JigsawState, theta: i32) -> bool {
    let n = state.puzzle.grid_size;
    let target_rot = ((theta % 360) + 360) % 360;
    if !state.unplaced.is_empty() {
        return false;
    }
    (0..n * n).all(|piece_id| {
        let target = target_cell(piece_id, n, theta);
        state.grid[target] == Some(piece_id) && state.rotations[piece_id] == target_rot
    })
}

// ---------------------------------------------------------------------------
// Expert solver
// ---------------------------------------------------------------------------

/// Return the optimal rotation action to move `current_rot` toward `target_rot`.
fn optimal_rotation_action(piece_id: usize, current_rot: i32, target_rot: i32) -> JigsawAction {
    let diff = ((target_rot - current_rot) % 360 + 360) % 360;
    // diff is how many degrees CW to reach target: 90 → 1 CW, 180 → 2 either, 270 → 1 CCW
    if diff <= 180 {
        JigsawAction::RotateCW { piece_id }
    } else {
        JigsawAction::RotateCCW { piece_id }
    }
}

/// Expert heuristic: picks the next action toward the best orientation.
///
/// Strategy:
/// 1. Remove any piece that's at the wrong cell for the target orientation.
/// 2. Among unplaced pieces, pick the best one to handle (corners first, then
///    most-constrained adjacent). Rotate it to target rotation, then place it.
fn expert_next_action(state: &JigsawState) -> Option<JigsawAction> {
    let n = state.puzzle.grid_size;
    let total = n * n;
    let (theta, _) = best_orientation(state);
    let target_rot = ((theta % 360) + 360) % 360;

    // Step 1: Remove any misplaced piece (placed at wrong cell for theta).
    for piece_id in 0..total {
        if state.unplaced.contains(&piece_id) {
            continue;
        }
        let current_cell = state
            .grid
            .iter()
            .position(|c| *c == Some(piece_id))
            .unwrap();
        let target = target_cell(piece_id, n, theta);
        if current_cell != target {
            return Some(JigsawAction::Remove { piece_id });
        }
    }

    // Step 2: Fix rotation of any placed piece at correct cell but wrong rotation.
    for piece_id in 0..total {
        if state.unplaced.contains(&piece_id) {
            continue;
        }
        if state.rotations[piece_id] != target_rot {
            return Some(optimal_rotation_action(
                piece_id,
                state.rotations[piece_id],
                target_rot,
            ));
        }
    }

    // Step 3: Pick the next unplaced piece to handle.
    if state.unplaced.is_empty() {
        return None; // solved
    }

    // Build priority from tray_order for tie-breaking.
    let mut priority = vec![total; total];
    for (rank, &piece_id) in state.puzzle.tray_order.iter().enumerate() {
        if piece_id < total {
            priority[piece_id] = rank;
        }
    }

    // Find unplaced piece to handle: corners first, then most-constrained.
    let target_cell_for = |pid: usize| target_cell(pid, n, theta);

    // Corner target cells.
    let corners = [0, n - 1, total - n, total - 1];

    // Priority 1: any unplaced piece whose target cell is a corner.
    let best_corner = state
        .unplaced
        .iter()
        .filter(|&&pid| corners.contains(&target_cell_for(pid)))
        .min_by_key(|&&pid| priority[pid]);
    let chosen = if let Some(&pid) = best_corner {
        pid
    } else {
        // Priority 2: unplaced piece whose target cell is adjacent to a correctly-placed piece.
        let mut best: Option<(usize, usize, usize)> = None; // (pid, neighbours, priority)
        for &pid in &state.unplaced {
            let tc = target_cell_for(pid);
            let row = tc / n;
            let col = tc % n;
            let mut placed_neighbours = 0u32;
            if row > 0 && state.grid[tc - n].is_some() {
                placed_neighbours += 1;
            }
            if row < n - 1 && state.grid[tc + n].is_some() {
                placed_neighbours += 1;
            }
            if col > 0 && state.grid[tc - 1].is_some() {
                placed_neighbours += 1;
            }
            if col < n - 1 && state.grid[tc + 1].is_some() {
                placed_neighbours += 1;
            }
            if placed_neighbours == 0 {
                continue;
            }
            let key = (usize::MAX - placed_neighbours as usize, priority[pid]);
            let is_better = match best {
                None => true,
                Some((_, best_n, best_p)) => key < (usize::MAX - best_n, best_p),
            };
            if is_better {
                best = Some((pid, placed_neighbours as usize, priority[pid]));
            }
        }
        if let Some((pid, _, _)) = best {
            pid
        } else {
            // Fallback: highest tray_order priority.
            *state.unplaced.iter().min_by_key(|&&pid| priority[pid])?
        }
    };

    // If chosen piece needs rotation, rotate first.
    if state.rotations[chosen] != target_rot {
        return Some(optimal_rotation_action(
            chosen,
            state.rotations[chosen],
            target_rot,
        ));
    }

    // Place at target cell (empty, since misplaced pieces are removed first).
    let tc = target_cell_for(chosen);
    Some(JigsawAction::Place {
        piece_id: chosen,
        cell: tc,
    })
}

// ---------------------------------------------------------------------------
// Game trait
// ---------------------------------------------------------------------------

impl Game for Jigsaw {
    type State = JigsawState;
    type Action = JigsawAction;

    fn generate(&self, rng: &mut ChaChaRng) -> Option<JigsawState> {
        let (lo, hi) = self.grid_size_range;
        let grid_size = if lo == hi { lo } else { rng.gen_range(lo..=hi) };
        Some(gen::generate(grid_size, rng))
    }

    fn puzzle_fingerprint(&self, state: &JigsawState) -> u64 {
        use std::hash::Hasher;
        // Identity = grid_size + the image generating parameters (which
        // deterministically reproduce piece_images) + the cut (h/v tabs) +
        // the scramble (tray_order, initial rotations). Pixels are NOT hashed;
        // gen_params reproduce them.
        let p = &*state.puzzle;
        let gp = &p.gen_params;
        crate::base::fingerprint::fingerprint(|h| {
            h.write_u64(p.grid_size as u64);

            // Image primitives (reproduce the source image). Background gradient
            // endpoints, then rectangles then circles in DRAW order (never
            // sorted: later shapes overwrite earlier pixels).
            h.write(&gp.bg_a);
            h.write(&gp.bg_b);
            h.write_u64(gp.rects.len() as u64);
            for r in &gp.rects {
                h.write_u8(r.color_idx);
                h.write_u32(r.x);
                h.write_u32(r.y);
                h.write_u32(r.w);
                h.write_u32(r.h);
            }
            h.write_u64(gp.circles.len() as u64);
            for c in &gp.circles {
                h.write_u8(c.color_idx);
                h.write_u32(c.cx as u32);
                h.write_u32(c.cy as u32);
                h.write_u32(c.radius as u32);
            }

            // Cut: horizontal / vertical edge tabs, in index order.
            h.write_u64(p.h_tabs.len() as u64);
            for &b in &p.h_tabs {
                h.write_u8(b as u8);
            }
            h.write_u64(p.v_tabs.len() as u64);
            for &b in &p.v_tabs {
                h.write_u8(b as u8);
            }

            // Scramble: tray slot assignment.
            h.write_u64(p.tray_order.len() as u64);
            for &t in &p.tray_order {
                h.write_u64(t as u64);
            }

            // Scramble: initial per-piece rotations, normalized to {0,1,2,3}.
            h.write_u64(state.rotations.len() as u64);
            for &r in &state.rotations {
                let norm = (((r % 360) + 360) % 360) as u32;
                h.write_u8((norm / 90) as u8);
            }
        })
    }

    fn trivial(&self) -> JigsawState {
        gen::trivial(self.grid_size_range.0)
    }

    fn actions(&self, state: &JigsawState) -> Vec<JigsawAction> {
        let n = state.puzzle.grid_size;
        let total = n * n;
        let mut result = Vec::new();

        // Place actions: unplaced pieces at empty cells.
        let empty_cells: Vec<usize> = state
            .grid
            .iter()
            .enumerate()
            .filter_map(|(i, c)| if c.is_none() { Some(i) } else { None })
            .collect();
        for &piece_id in &state.unplaced {
            for &cell in &empty_cells {
                result.push(JigsawAction::Place { piece_id, cell });
            }
        }

        // Rotate actions: all pieces.
        for piece_id in 0..total {
            result.push(JigsawAction::RotateCW { piece_id });
            result.push(JigsawAction::RotateCCW { piece_id });
        }

        // Remove actions: placed pieces only.
        for piece_id in 0..total {
            if !state.unplaced.contains(&piece_id) {
                result.push(JigsawAction::Remove { piece_id });
            }
        }

        result
    }

    fn apply(&self, state: &JigsawState, action: &JigsawAction) -> Option<JigsawState> {
        let n = state.puzzle.grid_size;
        match action {
            JigsawAction::Place { piece_id, cell } => {
                if *cell >= n * n {
                    return None;
                }
                if state.grid[*cell].is_some() {
                    return None;
                }
                if !state.unplaced.contains(piece_id) {
                    return None;
                }
                let mut new_grid = state.grid.clone();
                new_grid[*cell] = Some(*piece_id);
                let new_unplaced: Vec<usize> = state
                    .unplaced
                    .iter()
                    .copied()
                    .filter(|&p| p != *piece_id)
                    .collect();
                Some(JigsawState {
                    puzzle: state.puzzle.clone(),
                    grid: new_grid,
                    unplaced: new_unplaced,
                    rotations: state.rotations.clone(),
                })
            }
            JigsawAction::RotateCW { piece_id } => {
                if *piece_id >= n * n {
                    return None;
                }
                let mut new_rotations = state.rotations.clone();
                new_rotations[*piece_id] = (new_rotations[*piece_id] + 90) % 360;
                Some(JigsawState {
                    puzzle: state.puzzle.clone(),
                    grid: state.grid.clone(),
                    unplaced: state.unplaced.clone(),
                    rotations: new_rotations,
                })
            }
            JigsawAction::RotateCCW { piece_id } => {
                if *piece_id >= n * n {
                    return None;
                }
                let mut new_rotations = state.rotations.clone();
                new_rotations[*piece_id] = (new_rotations[*piece_id] + 270) % 360;
                Some(JigsawState {
                    puzzle: state.puzzle.clone(),
                    grid: state.grid.clone(),
                    unplaced: state.unplaced.clone(),
                    rotations: new_rotations,
                })
            }
            JigsawAction::Remove { piece_id } => {
                if *piece_id >= n * n {
                    return None;
                }
                // Piece must be on the grid (not in tray).
                if state.unplaced.contains(piece_id) {
                    return None;
                }
                let mut new_grid = state.grid.clone();
                // Find and clear the piece's cell.
                for cell in new_grid.iter_mut() {
                    if *cell == Some(*piece_id) {
                        *cell = None;
                        break;
                    }
                }
                let mut new_unplaced = state.unplaced.clone();
                new_unplaced.push(*piece_id);
                new_unplaced.sort();
                Some(JigsawState {
                    puzzle: state.puzzle.clone(),
                    grid: new_grid,
                    unplaced: new_unplaced,
                    rotations: state.rotations.clone(),
                })
            }
        }
    }

    fn is_solved(&self, state: &JigsawState) -> bool {
        [0, 90, 180, 270]
            .iter()
            .any(|&theta| is_solved_for_theta(state, theta))
    }

    fn render(&self, state: &JigsawState, resolution: u32) -> RgbBuffer {
        render::render(state, resolution)
    }

    fn action_to_str(&self, action: &JigsawAction) -> String {
        match action {
            JigsawAction::Place { piece_id, cell } => {
                format!("place({}, {})", piece_id, cell_label(*cell))
            }
            JigsawAction::RotateCW { piece_id } => {
                format!("rotatecw({})", piece_id)
            }
            JigsawAction::RotateCCW { piece_id } => {
                format!("rotateccw({})", piece_id)
            }
            JigsawAction::Remove { piece_id } => {
                format!("remove({})", piece_id)
            }
        }
    }

    fn parse_action(&self, state: &JigsawState, s: &str) -> Option<JigsawAction> {
        let n = state.puzzle.grid_size;
        if let Some(inner) = s.strip_prefix("place(") {
            let inner = inner.strip_suffix(')')?;
            let parts: Vec<&str> = inner.splitn(2, ", ").collect();
            if parts.len() != 2 {
                return None;
            }
            let piece_id: usize = parts[0].parse().ok()?;
            let cell = parse_cell_label(parts[1])?;
            if piece_id >= n * n || cell >= n * n {
                return None;
            }
            Some(JigsawAction::Place { piece_id, cell })
        } else if let Some(inner) = s.strip_prefix("rotatecw(") {
            let piece_id: usize = inner.strip_suffix(')')?.parse().ok()?;
            if piece_id >= n * n {
                return None;
            }
            Some(JigsawAction::RotateCW { piece_id })
        } else if let Some(inner) = s.strip_prefix("rotateccw(") {
            let piece_id: usize = inner.strip_suffix(')')?.parse().ok()?;
            if piece_id >= n * n {
                return None;
            }
            Some(JigsawAction::RotateCCW { piece_id })
        } else if let Some(inner) = s.strip_prefix("remove(") {
            let piece_id: usize = inner.strip_suffix(')')?.parse().ok()?;
            if piece_id >= n * n {
                return None;
            }
            Some(JigsawAction::Remove { piece_id })
        } else {
            None
        }
    }

    fn action_templates(&self, state: &JigsawState) -> Vec<ActionTemplate> {
        let n = state.puzzle.grid_size;
        let piece_ids: Vec<String> = (0..(n * n)).map(|i| i.to_string()).collect();
        let cell_labels: Vec<String> = (0..(n * n)).map(cell_label).collect();
        vec![
            ActionTemplate {
                segments: vec![
                    ParamType::Literal("place(".into()),
                    ParamType::OneOf(piece_ids.clone()),
                    ParamType::Literal(", ".into()),
                    ParamType::OneOf(cell_labels),
                    ParamType::Literal(")".into()),
                ],
            },
            ActionTemplate {
                segments: vec![
                    ParamType::Literal("rotatecw(".into()),
                    ParamType::OneOf(piece_ids.clone()),
                    ParamType::Literal(")".into()),
                ],
            },
            ActionTemplate {
                segments: vec![
                    ParamType::Literal("rotateccw(".into()),
                    ParamType::OneOf(piece_ids.clone()),
                    ParamType::Literal(")".into()),
                ],
            },
            ActionTemplate {
                segments: vec![
                    ParamType::Literal("remove(".into()),
                    ParamType::OneOf(piece_ids),
                    ParamType::Literal(")".into()),
                ],
            },
        ]
    }

    fn prompt(&self, _state: &JigsawState) -> String {
        "Reconstruct the original image by rotating each piece to the correct \
         orientation and placing it at its correct grid position."
            .to_string()
    }

    fn action_description(&self, _state: &JigsawState) -> String {
        "place(piece_id, cell) \u{2014} place a piece at a grid cell (A, B, C, ...); \
         rotatecw(piece_id) \u{2014} rotate 90\u{00b0} clockwise; \
         rotateccw(piece_id) \u{2014} rotate 90\u{00b0} counter-clockwise; \
         remove(piece_id) \u{2014} remove a piece from the grid back to the tray"
            .to_string()
    }

    fn object_boxes(&self, state: &JigsawState) -> Vec<crate::base::object_boxes::ObjectBox> {
        object_boxes::object_boxes(state)
    }

    /// `g`: piece id at each grid cell, row-major (`null` = empty); `r`:
    /// rotation of every piece in degrees; `unplaced`: pieces in the tray;
    /// `n_placed`: pieces on the grid.
    fn state_fields(&self, state: &JigsawState) -> Vec<StateField> {
        let num_placed = state.grid.iter().filter(|cell| cell.is_some()).count();
        vec![
            ("g", json_optional_numbers(state.grid.iter().copied())),
            ("r", json_numbers(state.rotations.iter())),
            ("unplaced", json_numbers(state.unplaced.iter())),
            ("n_placed", num_placed.to_string()),
        ]
    }

    // --- custom solver ---

    fn solve(&self, state: &JigsawState, _budget: usize) -> Option<(JigsawAction, usize)> {
        let action = expert_next_action(state)?;
        let (_, cost) = best_orientation(state);
        Some((action, cost))
    }

    fn solve_path(
        &self,
        state: &JigsawState,
        _budget: usize,
    ) -> Option<Vec<(JigsawAction, usize)>> {
        let (_, total_cost) = best_orientation(state);
        if total_cost == 0 {
            return Some(vec![]);
        }

        let mut path = Vec::with_capacity(total_cost);
        let mut sim_state = state.clone();
        let mut cost_remaining = total_cost;

        for _ in 0..total_cost {
            let action = expert_next_action(&sim_state)?;
            path.push((action.clone(), cost_remaining));
            cost_remaining -= 1;
            sim_state = self.apply(&sim_state, &action)?;
            if self.is_solved(&sim_state) {
                break;
            }
        }

        Some(path)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::episode::{Episode, GameEpisode};
    use rand::SeedableRng;
    use rand_chacha::ChaChaRng;
    use std::sync::Arc;

    fn default_game() -> Jigsaw {
        Jigsaw::default()
    }

    fn make_episode_from_state(state: JigsawState) -> GameEpisode<Jigsaw> {
        let game = Arc::new(default_game());
        GameEpisode::new(game, state, 100_000)
    }

    /// Generate a state with all rotations set to 0 (for tests that focus on placement).
    fn generate_no_rotation(game: &Jigsaw, seed: u64) -> JigsawState {
        let mut rng = ChaChaRng::seed_from_u64(seed);
        let mut state = game.generate(&mut rng).unwrap();
        for r in state.rotations.iter_mut() {
            *r = 0;
        }
        state
    }

    #[test]
    fn state_text_lists_grid_rotations_and_tray() {
        let mut state = generate_no_rotation(&default_game(), 0);
        let num_pieces = state.rotations.len();
        state.grid = vec![None; num_pieces];
        state.grid[0] = Some(2);
        state.unplaced = (0..num_pieces).filter(|&piece| piece != 2).collect();
        state.rotations[2] = 90;
        let fields = default_game().state_fields(&state);
        let text = crate::base::state_text::format_state_json(&fields, 4);
        let empty_cells = vec!["null"; num_pieces - 1].join(",");
        assert!(
            text.starts_with(&format!(r#"{{"g":[2,{empty_cells}],"r":[0,0,90,"#)),
            "{text}"
        );
        assert!(text.ends_with(r#""n_placed":1,"moves":4}"#), "{text}");
    }

    // --- registry ---

    #[test]
    fn registered_in_registry() {
        assert!(crate::games::registry::find_game("jigsaw").is_some());
    }

    // --- trivial ---

    #[test]
    fn trivial_is_solvable() {
        let game = default_game();
        let state = game.trivial();
        let result = game.solve(&state, 100_000);
        assert!(result.is_some(), "trivial state should be solvable");
        let (action, cost) = result.unwrap();
        assert!(
            matches!(
                action,
                JigsawAction::Place {
                    piece_id: 0,
                    cell: 0
                }
            ),
            "expected place(0, 0), got {:?}",
            action
        );
        assert_eq!(cost, 1);
    }

    // --- generate ---

    #[test]
    fn generate_produces_solvable() {
        let game = Jigsaw {
            grid_size_range: (3, 3),
            solve_budget: 100_000,
        };
        for seed in 0..20u64 {
            let mut rng = ChaChaRng::seed_from_u64(seed);
            let state = game.generate(&mut rng).expect("should generate");
            let result = game.solve(&state, 100_000);
            assert!(result.is_some(), "seed {} produced unsolvable puzzle", seed);
        }
    }

    #[test]
    fn generate_4x4_solvable() {
        let game = Jigsaw {
            grid_size_range: (4, 4),
            solve_budget: 100_000,
        };
        for seed in 0..10u64 {
            let mut rng = ChaChaRng::seed_from_u64(seed);
            let state = game.generate(&mut rng).expect("should generate");
            assert!(game.solve(&state, 100_000).is_some());
        }
    }

    #[test]
    fn generate_has_rotations() {
        let game = Jigsaw {
            grid_size_range: (3, 3),
            solve_budget: 100_000,
        };
        // Over 20 seeds, at least some pieces should have non-zero rotation.
        let mut saw_nonzero = false;
        for seed in 0..20u64 {
            let mut rng = ChaChaRng::seed_from_u64(seed);
            let state = game.generate(&mut rng).unwrap();
            if state.rotations.iter().any(|&r| r != 0) {
                saw_nonzero = true;
                break;
            }
        }
        assert!(
            saw_nonzero,
            "expected some non-zero rotations across 20 seeds"
        );
    }

    // --- actions ---

    #[test]
    fn actions_initial_state() {
        let game = Jigsaw {
            grid_size_range: (2, 2),
            solve_budget: 100_000,
        };
        let state = generate_no_rotation(&game, 0);
        let actions = game.actions(&state);
        // 4 unplaced * 4 empty = 16 place + 2*4 rotate + 0 remove = 24
        assert_eq!(actions.len(), 24);
    }

    #[test]
    fn actions_decrease_after_placement() {
        let game = Jigsaw {
            grid_size_range: (2, 2),
            solve_budget: 100_000,
        };
        let state = generate_no_rotation(&game, 0);
        let action = JigsawAction::Place {
            piece_id: 0,
            cell: 0,
        };
        let new = game.apply(&state, &action).unwrap();
        let actions = game.actions(&new);
        // 3 unplaced * 3 empty = 9 place + 2*4 rotate + 1 remove = 18
        assert_eq!(actions.len(), 18);
    }

    // --- apply ---

    #[test]
    fn apply_valid_placement() {
        let game = Jigsaw {
            grid_size_range: (3, 3),
            solve_budget: 100_000,
        };
        let state = generate_no_rotation(&game, 42);
        let action = JigsawAction::Place {
            piece_id: 0,
            cell: 0,
        };
        let new = game.apply(&state, &action).unwrap();
        assert_eq!(new.grid[0], Some(0));
        assert!(!new.unplaced.contains(&0));
    }

    #[test]
    fn apply_occupied_cell_returns_none() {
        let game = default_game();
        let state = game.trivial();
        // Cell 1 is occupied in trivial state.
        let action = JigsawAction::Place {
            piece_id: 0,
            cell: 1,
        };
        assert!(game.apply(&state, &action).is_none());
    }

    #[test]
    fn apply_already_placed_piece_returns_none() {
        let game = default_game();
        let state = game.trivial();
        // Piece 1 is already placed.
        let action = JigsawAction::Place {
            piece_id: 1,
            cell: 0,
        };
        assert!(game.apply(&state, &action).is_none());
    }

    #[test]
    fn apply_out_of_bounds_returns_none() {
        let game = Jigsaw {
            grid_size_range: (3, 3),
            solve_budget: 100_000,
        };
        let state = generate_no_rotation(&game, 0);
        let action = JigsawAction::Place {
            piece_id: 0,
            cell: 99,
        };
        assert!(game.apply(&state, &action).is_none());
    }

    #[test]
    fn apply_rotatecw() {
        let game = Jigsaw {
            grid_size_range: (2, 2),
            solve_budget: 100_000,
        };
        let state = generate_no_rotation(&game, 0);
        assert_eq!(state.rotations[0], 0);
        let new = game
            .apply(&state, &JigsawAction::RotateCW { piece_id: 0 })
            .unwrap();
        assert_eq!(new.rotations[0], 90);
        let new = game
            .apply(&new, &JigsawAction::RotateCW { piece_id: 0 })
            .unwrap();
        assert_eq!(new.rotations[0], 180);
    }

    #[test]
    fn apply_rotateccw() {
        let game = Jigsaw {
            grid_size_range: (2, 2),
            solve_budget: 100_000,
        };
        let state = generate_no_rotation(&game, 0);
        let new = game
            .apply(&state, &JigsawAction::RotateCCW { piece_id: 0 })
            .unwrap();
        assert_eq!(new.rotations[0], 270);
    }

    #[test]
    fn apply_rotate_wraps() {
        let game = Jigsaw {
            grid_size_range: (2, 2),
            solve_budget: 100_000,
        };
        let state = generate_no_rotation(&game, 0);
        let mut s = state;
        for _ in 0..4 {
            s = game
                .apply(&s, &JigsawAction::RotateCW { piece_id: 0 })
                .unwrap();
        }
        assert_eq!(s.rotations[0], 0);
    }

    #[test]
    fn apply_remove() {
        let game = default_game();
        let state = game.trivial();
        // Piece 1 is placed. Remove it.
        let new = game
            .apply(&state, &JigsawAction::Remove { piece_id: 1 })
            .unwrap();
        assert!(new.unplaced.contains(&1));
        assert!(new.grid.iter().all(|c| *c != Some(1)));
    }

    #[test]
    fn apply_remove_unplaced_returns_none() {
        let game = Jigsaw {
            grid_size_range: (2, 2),
            solve_budget: 100_000,
        };
        let state = generate_no_rotation(&game, 0);
        // All pieces are unplaced.
        assert!(game
            .apply(&state, &JigsawAction::Remove { piece_id: 0 })
            .is_none());
    }

    // --- is_solved ---

    #[test]
    fn is_solved_theta_0() {
        let game = Jigsaw {
            grid_size_range: (2, 2),
            solve_budget: 100_000,
        };
        let state = generate_no_rotation(&game, 0);
        let mut s = state;
        for i in 0..4 {
            s = game
                .apply(
                    &s,
                    &JigsawAction::Place {
                        piece_id: i,
                        cell: i,
                    },
                )
                .unwrap();
        }
        assert!(game.is_solved(&s));
    }

    #[test]
    fn is_solved_wrong_rotation() {
        let game = Jigsaw {
            grid_size_range: (2, 2),
            solve_budget: 100_000,
        };
        let state = generate_no_rotation(&game, 0);
        let mut s = state;
        for i in 0..4 {
            s = game
                .apply(
                    &s,
                    &JigsawAction::Place {
                        piece_id: i,
                        cell: i,
                    },
                )
                .unwrap();
        }
        // Rotate one piece — should no longer be solved for theta=0.
        s = game
            .apply(&s, &JigsawAction::RotateCW { piece_id: 0 })
            .unwrap();
        assert!(!game.is_solved(&s));
    }

    #[test]
    fn is_solved_wrong_placement() {
        let game = Jigsaw {
            grid_size_range: (2, 2),
            solve_budget: 100_000,
        };
        let state = generate_no_rotation(&game, 0);
        // Place piece 1 at cell 0 (wrong).
        let s = game
            .apply(
                &state,
                &JigsawAction::Place {
                    piece_id: 1,
                    cell: 0,
                },
            )
            .unwrap();
        assert!(!game.is_solved(&s));
    }

    #[test]
    fn is_solved_alternate_orientation() {
        // Verify that a 2x2 puzzle solved with all pieces rotated 90° at the
        // corresponding target cells is accepted.
        let game = Jigsaw {
            grid_size_range: (2, 2),
            solve_budget: 100_000,
        };
        let state = generate_no_rotation(&game, 0);
        let n = 2;
        let theta = 90;
        let mut s = state;
        for piece_id in 0..4 {
            // Rotate to 90°.
            s = game
                .apply(&s, &JigsawAction::RotateCW { piece_id })
                .unwrap();
            let tc = target_cell(piece_id, n, theta);
            s = game
                .apply(&s, &JigsawAction::Place { piece_id, cell: tc })
                .unwrap();
        }
        assert!(
            game.is_solved(&s),
            "puzzle with all pieces at theta=90 targets should be solved"
        );
    }

    // --- action_to_str / parse_action roundtrip ---

    #[test]
    fn action_to_str_parse_roundtrip() {
        let game = Jigsaw {
            grid_size_range: (3, 3),
            solve_budget: 100_000,
        };
        let state = generate_no_rotation(&game, 0);
        let actions = game.actions(&state);
        assert!(!actions.is_empty());
        for action in &actions {
            let s = game.action_to_str(action);
            let parsed = game
                .parse_action(&state, &s)
                .unwrap_or_else(|| panic!("failed to parse '{}'", s));
            assert_eq!(
                game.action_to_str(action),
                game.action_to_str(&parsed),
                "roundtrip failed for '{}'",
                s
            );
        }
    }

    #[test]
    fn parse_action_rotate() {
        let game = default_game();
        let state = game.trivial();
        assert!(matches!(
            game.parse_action(&state, "rotatecw(0)"),
            Some(JigsawAction::RotateCW { piece_id: 0 })
        ));
        assert!(matches!(
            game.parse_action(&state, "rotateccw(2)"),
            Some(JigsawAction::RotateCCW { piece_id: 2 })
        ));
        assert!(matches!(
            game.parse_action(&state, "remove(1)"),
            Some(JigsawAction::Remove { piece_id: 1 })
        ));
    }

    #[test]
    fn parse_action_invalid() {
        let game = default_game();
        let state = game.trivial();
        assert!(game.parse_action(&state, "garbage").is_none());
        assert!(game.parse_action(&state, "place(99, A)").is_none());
        assert!(game.parse_action(&state, "place(0, ZZ)").is_none());
    }

    // --- action_templates ---

    #[test]
    fn action_templates_cover_all_valid_actions() {
        let game = Jigsaw {
            grid_size_range: (3, 3),
            solve_budget: 100_000,
        };
        let state = generate_no_rotation(&game, 0);
        let templates = game.action_templates(&state);
        assert_eq!(templates.len(), 4); // place, rotatecw, rotateccw, remove
        let expanded: Vec<String> = templates.iter().flat_map(|t| t.expand()).collect();
        let actions = game.actions(&state);
        for action in &actions {
            let s = game.action_to_str(action);
            assert!(
                expanded.contains(&s),
                "valid action '{}' not in template expansion",
                s
            );
        }
    }

    // --- solver ---

    #[test]
    fn solve_returns_action() {
        let game = Jigsaw {
            grid_size_range: (3, 3),
            solve_budget: 100_000,
        };
        for seed in 0..10u64 {
            let mut rng = ChaChaRng::seed_from_u64(seed);
            let state = game.generate(&mut rng).unwrap();
            let result = game.solve(&state, 100_000);
            assert!(result.is_some(), "seed {} should be solvable", seed);
        }
    }

    #[test]
    fn solve_path_reaches_solved() {
        let game = Jigsaw {
            grid_size_range: (2, 2),
            solve_budget: 100_000,
        };
        for seed in 0..10u64 {
            let mut rng = ChaChaRng::seed_from_u64(seed);
            let state = game.generate(&mut rng).unwrap();
            let path = game.solve_path(&state, 100_000).unwrap();
            assert!(!path.is_empty(), "seed {} empty path but not solved", seed);

            // Replay the path.
            let mut s = state;
            for (action, _) in &path {
                s = game.apply(&s, action).unwrap();
            }
            assert!(
                game.is_solved(&s),
                "seed {} path did not reach solved",
                seed
            );
        }
    }

    #[test]
    fn solve_path_costs_decrease() {
        let game = Jigsaw {
            grid_size_range: (3, 3),
            solve_budget: 100_000,
        };
        let state = generate_no_rotation(&game, 0);
        let path = game.solve_path(&state, 100_000).unwrap();
        for window in path.windows(2) {
            assert!(
                window[0].1 > window[1].1,
                "costs should strictly decrease: {} -> {}",
                window[0].1,
                window[1].1
            );
        }
    }

    // --- GameEpisode integration tests ---

    #[test]
    fn episode_expert_action_returns_optimal() {
        let game = Jigsaw {
            grid_size_range: (3, 3),
            solve_budget: 100_000,
        };
        for seed in 0..5u64 {
            let mut rng = ChaChaRng::seed_from_u64(seed);
            let state = game.generate(&mut rng).unwrap();
            let mut ep = make_episode_from_state(state);
            let mut steps = 0;
            while !ep.is_solved() {
                let action = ep.expert_action();
                assert!(
                    ep.apply(&action),
                    "expert action '{}' failed at step {} (seed {})",
                    action,
                    steps,
                    seed
                );
                steps += 1;
                assert!(steps <= 100, "too many steps (seed {})", seed);
            }
        }
    }

    #[test]
    fn episode_expert_action_done_when_solved() {
        let game = Jigsaw {
            grid_size_range: (2, 2),
            solve_budget: 100_000,
        };
        let state = generate_no_rotation(&game, 0);
        let mut s = state;
        for i in 0..4 {
            s = game
                .apply(
                    &s,
                    &JigsawAction::Place {
                        piece_id: i,
                        cell: i,
                    },
                )
                .unwrap();
        }
        let ep = make_episode_from_state(s);
        assert!(ep.is_solved());
        assert_eq!(ep.expert_action(), "done");
    }

    #[test]
    fn episode_apply_undo_roundtrip() {
        let game = Jigsaw {
            grid_size_range: (3, 3),
            solve_budget: 100_000,
        };
        let state = generate_no_rotation(&game, 0);
        let original_grid = state.grid.clone();
        let original_rotations = state.rotations.clone();
        let mut ep = make_episode_from_state(state);

        assert!(ep.apply("place(0, A)"));
        assert_ne!(ep.state.grid, original_grid);

        ep.undo();
        assert_eq!(ep.state.grid, original_grid);
        assert_eq!(ep.state.rotations, original_rotations);
    }

    #[test]
    fn episode_reset_restores_initial() {
        let game = Jigsaw {
            grid_size_range: (3, 3),
            solve_budget: 100_000,
        };
        let state = generate_no_rotation(&game, 0);
        let initial_grid = state.grid.clone();
        let mut ep = make_episode_from_state(state);

        ep.apply("place(0, A)");
        ep.apply("place(4, E)");
        assert_ne!(ep.state.grid, initial_grid);

        ep.reset();
        assert_eq!(ep.state.grid, initial_grid);
        assert!(ep.history.is_empty());
    }

    #[test]
    fn episode_enumerate_includes_meta() {
        let game = Jigsaw {
            grid_size_range: (2, 2),
            solve_budget: 100_000,
        };
        let state = generate_no_rotation(&game, 0);
        let ep = make_episode_from_state(state);
        let actions = ep.enumerate_actions();
        assert!(actions.contains(&"back".to_string()));
        assert!(actions.contains(&"reset".to_string()));
        assert!(actions.contains(&"done".to_string()));
        assert!(actions.contains(&"place(0, A)".to_string()));
        assert!(actions.contains(&"rotatecw(0)".to_string()));
        assert!(actions.contains(&"rotateccw(0)".to_string()));
    }

    // --- render ---

    #[test]
    fn render_does_not_panic() {
        let game = Jigsaw {
            grid_size_range: (3, 3),
            solve_budget: 100_000,
        };
        let mut rng = ChaChaRng::seed_from_u64(0);
        let state = game.generate(&mut rng).unwrap();
        let buf = game.render(&state, 256);
        assert_eq!(buf.width, buf.height);
        assert!(buf.width > 0);
        assert_eq!(buf.data.len(), (buf.width * buf.height * 3) as usize);
    }

    #[test]
    fn render_solved_state() {
        let game = Jigsaw {
            grid_size_range: (2, 2),
            solve_budget: 100_000,
        };
        let state = generate_no_rotation(&game, 0);
        let mut s = state;
        for i in 0..4 {
            s = game
                .apply(
                    &s,
                    &JigsawAction::Place {
                        piece_id: i,
                        cell: i,
                    },
                )
                .unwrap();
        }
        let buf = game.render(&s, 128);
        assert_eq!(buf.width, buf.height);
        assert!(buf.width > 0);
    }

    // --- edge cases ---

    #[test]
    fn solve_path_on_trivial() {
        let game = default_game();
        let state = game.trivial();
        let path = game.solve_path(&state, 100_000).unwrap();
        assert_eq!(path.len(), 1);
        assert!(matches!(
            path[0].0,
            JigsawAction::Place {
                piece_id: 0,
                cell: 0
            }
        ));
        assert_eq!(path[0].1, 1);
    }

    #[test]
    fn solve_path_on_solved() {
        let game = Jigsaw {
            grid_size_range: (2, 2),
            solve_budget: 100_000,
        };
        let state = generate_no_rotation(&game, 0);
        let mut s = state;
        for i in 0..4 {
            s = game
                .apply(
                    &s,
                    &JigsawAction::Place {
                        piece_id: i,
                        cell: i,
                    },
                )
                .unwrap();
        }
        let path = game.solve_path(&s, 100_000).unwrap();
        assert!(path.is_empty());
    }

    #[test]
    fn min_rotations_helper() {
        assert_eq!(min_rotations(0, 0), 0);
        assert_eq!(min_rotations(0, 90), 1);
        assert_eq!(min_rotations(0, 180), 2);
        assert_eq!(min_rotations(0, 270), 1);
        assert_eq!(min_rotations(90, 270), 2);
        assert_eq!(min_rotations(270, 0), 1);
    }

    #[test]
    fn target_cell_theta_0_identity() {
        for n in 2..=4 {
            for pid in 0..n * n {
                assert_eq!(target_cell(pid, n, 0), pid);
            }
        }
    }

    #[test]
    fn target_cell_four_rotations_cycle() {
        // Applying theta=90 four times should return to original.
        let n = 3;
        for pid in 0..n * n {
            let mut cell = pid;
            for _ in 0..4 {
                cell = target_cell(cell, n, 90);
            }
            assert_eq!(cell, pid, "piece {} did not cycle back", pid);
        }
    }
}
