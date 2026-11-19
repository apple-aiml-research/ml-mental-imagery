//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
pub mod object_boxes;
pub mod render;

use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use rand::Rng;
use rand_chacha::ChaChaRng;

use crate::base::game::{Game, RgbBuffer};
use crate::base::state_text::{json_string, StateField};
use crate::base::template::{ActionTemplate, ParamType};
use crate::solver::dlx::DLX;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A tangram piece with its shape (relative cells at rotation=0).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PieceDef {
    pub id: String,
    pub cells: Vec<(i32, i32)>,
}

impl PieceDef {
    /// Rotate piece cells by the given angle (0, 90, 180, 270).
    pub fn rotated_cells(&self, rotation: i32) -> Vec<(i32, i32)> {
        let rot = ((rotation % 360) + 360) % 360;
        self.cells
            .iter()
            .map(|&(x, y)| match rot {
                0 => (x, y),
                90 => (-y, x),
                180 => (-x, -y),
                270 => (y, -x),
                _ => (x, y),
            })
            .collect()
    }
}

/// A placed piece on the canvas.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Placement {
    pub piece_id: String,
    pub x: i32,
    pub y: i32,
    pub rotation: i32,
}

impl Placement {
    pub fn occupied_cells(&self, piece: &PieceDef) -> Vec<(i32, i32)> {
        piece
            .rotated_cells(self.rotation)
            .into_iter()
            .map(|(dx, dy)| (self.x + dx, self.y + dy))
            .collect()
    }
}

/// Tangram action: place a piece, or rotate it CW/CCW.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TangramAction {
    Place { piece_id: String, x: i32, y: i32 },
    RotateCW { piece_id: String },
    RotateCCW { piece_id: String },
}

/// Shared puzzle data (immutable across states derived from the same puzzle).
#[derive(Debug, Clone)]
pub struct TangramPuzzle {
    pub canvas_width: usize,
    pub canvas_height: usize,
    pub pieces: Vec<PieceDef>,
    pub target_set: HashSet<(i32, i32)>,
    pub color_map: Vec<usize>,
    pub target_placements: Vec<Placement>,
    // --- certificate mode fields ---
    /// Whether this puzzle is solvable (true for normal puzzles).
    pub solvable: bool,
    /// Tray layout `(resolution, cell_size, positions)`, computed at the first
    /// render's resolution and reused by later renders. The packer is not
    /// scale-invariant, so `object_boxes` needs the resolution it was packed for.
    pub tray_layout: std::sync::OnceLock<(u32, u32, Vec<(u32, u32)>)>,
}

/// Tangram game state.
///
/// Hash/Eq only on `placed`, `remaining_piece_ids`, `queue_rotations` —
/// all stored as sorted Vecs for deterministic hashing.
#[derive(Debug, Clone)]
pub struct TangramState {
    pub puzzle: Arc<TangramPuzzle>,
    pub placed: Vec<Placement>,
    pub remaining_piece_ids: Vec<String>,
    pub queue_rotations: Vec<(String, i32)>,
    pub occupied_cells: HashSet<(i32, i32)>,
}

impl PartialEq for TangramState {
    fn eq(&self, other: &Self) -> bool {
        self.placed == other.placed
            && self.remaining_piece_ids == other.remaining_piece_ids
            && self.queue_rotations == other.queue_rotations
    }
}

impl Eq for TangramState {}

impl Hash for TangramState {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        self.placed.hash(hasher);
        self.remaining_piece_ids.hash(hasher);
        self.queue_rotations.hash(hasher);
    }
}

impl TangramState {
    /// Check if any placed pieces overlap.
    pub fn has_overlap(&self) -> bool {
        let mut seen = HashSet::new();
        for placement in &self.placed {
            if let Some(piece) = self
                .puzzle
                .pieces
                .iter()
                .find(|p| p.id == placement.piece_id)
            {
                for cell in placement.occupied_cells(piece) {
                    if !seen.insert(cell) {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// Get the queue rotation for a piece (default 0).
    pub fn queue_rotation(&self, piece_id: &str) -> i32 {
        self.queue_rotations
            .iter()
            .find(|(k, _)| k == piece_id)
            .map(|(_, v)| *v)
            .unwrap_or(0)
    }

    /// Set the queue rotation for a piece, maintaining sorted order.
    pub fn set_queue_rotation(&mut self, piece_id: &str, rotation: i32) {
        if let Some(entry) = self.queue_rotations.iter_mut().find(|(k, _)| k == piece_id) {
            entry.1 = rotation;
        } else {
            self.queue_rotations.push((piece_id.to_string(), rotation));
            self.queue_rotations.sort_by(|a, b| a.0.cmp(&b.0));
        }
    }

    /// Rebuild occupied_cells from current placements.
    pub fn recompute_occupied(&mut self) {
        self.occupied_cells.clear();
        for placement in &self.placed {
            if let Some(piece) = self
                .puzzle
                .pieces
                .iter()
                .find(|p| p.id == placement.piece_id)
            {
                for cell in placement.occupied_cells(piece) {
                    self.occupied_cells.insert(cell);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Rotation utilities
// ---------------------------------------------------------------------------

/// Minimum number of 90-degree rotations to go from current to target (bidirectional).
pub fn min_rotations(current: i32, target: i32) -> usize {
    let diff = ((target - current) % 360 + 360) % 360;
    let cw = (diff / 90) as usize;
    let ccw = ((360 - diff) / 90) as usize;
    cw.min(ccw)
}

/// Find the rotation closest to `current` that produces the same raw cell set as `target`.
/// Two rotations are equivalent only if rotated_cells() returns the same multiset,
/// ensuring identical absolute cells at any (x,y) position.
pub fn best_equivalent_rotation(piece: &PieceDef, current: i32, target: i32) -> i32 {
    let mut target_cells = piece.rotated_cells(target);
    target_cells.sort();
    let mut best_rot = target;
    let mut best_cost = min_rotations(current, target);
    for &rot in &[0, 90, 180, 270] {
        let mut cells = piece.rotated_cells(rot);
        cells.sort();
        if cells == target_cells {
            let cost = min_rotations(current, rot);
            if cost < best_cost {
                best_cost = cost;
                best_rot = rot;
            }
        }
    }
    best_rot
}

/// Minimum rotations from `current` to reach a rotation equivalent to `target`,
/// accounting for the piece's rotational symmetry.
pub fn effective_min_rotations(piece: &PieceDef, current: i32, target: i32) -> usize {
    let best = best_equivalent_rotation(piece, current, target);
    min_rotations(current, best)
}

/// Emit minimal RotateCW/RotateCCW actions to go from current to target rotation.
fn emit_rotations(actions: &mut Vec<TangramAction>, piece_id: &str, current: i32, target: i32) {
    let diff = ((target - current) % 360 + 360) % 360;
    if diff == 0 {
        return;
    }
    let cw_steps = diff / 90;
    let ccw_steps = (360 - diff) / 90;
    if cw_steps <= ccw_steps {
        for _ in 0..cw_steps {
            actions.push(TangramAction::RotateCW {
                piece_id: piece_id.to_string(),
            });
        }
    } else {
        for _ in 0..ccw_steps {
            actions.push(TangramAction::RotateCCW {
                piece_id: piece_id.to_string(),
            });
        }
    }
}

// ---------------------------------------------------------------------------
// DLX integration
// ---------------------------------------------------------------------------

/// Maps a DLX row back to a tangram placement.
#[derive(Debug, Clone)]
struct PlacementInfo {
    piece_id: String,
    x: i32,
    y: i32,
    rotation: i32,
}

/// Build the exact cover matrix for the current tangram state.
///
/// Columns:
///   - 0..num_uncovered_cells: one column per uncovered target cell
///   - num_uncovered_cells..: one "piece-used" column per remaining piece
///
/// Returns (num_columns, rows, placement_infos).
fn build_cover_matrix(state: &TangramState) -> (usize, Vec<Vec<usize>>, Vec<PlacementInfo>) {
    let uncovered: Vec<(i32, i32)> = {
        let mut cells: Vec<(i32, i32)> = state
            .puzzle
            .target_set
            .iter()
            .filter(|c| !state.occupied_cells.contains(c))
            .copied()
            .collect();
        cells.sort();
        cells
    };

    let num_cell_cols = uncovered.len();
    let remaining_pieces: Vec<&String> = state.remaining_piece_ids.iter().collect();
    let num_piece_cols = remaining_pieces.len();
    let num_columns = num_cell_cols + num_piece_cols;

    let cell_to_col: std::collections::HashMap<(i32, i32), usize> =
        uncovered.iter().enumerate().map(|(i, &c)| (c, i)).collect();

    let piece_to_col: std::collections::HashMap<&String, usize> = remaining_pieces
        .iter()
        .enumerate()
        .map(|(i, pid)| (*pid, num_cell_cols + i))
        .collect();

    let uncovered_set: HashSet<(i32, i32)> = uncovered.iter().copied().collect();

    let mut rows = Vec::new();
    let mut placements = Vec::new();
    let rotations = [0, 90, 180, 270];

    for piece_id in &remaining_pieces {
        let piece = match state.puzzle.pieces.iter().find(|p| &p.id == *piece_id) {
            Some(p) => p,
            None => continue,
        };

        let mut seen_shapes: HashSet<Vec<(i32, i32)>> = HashSet::new();

        for &rotation in &rotations {
            let rotated_cells = piece.rotated_cells(rotation);
            let mut sorted = rotated_cells.clone();
            sorted.sort();
            if !seen_shapes.insert(sorted) {
                continue;
            }

            for y in 0..state.puzzle.canvas_height as i32 {
                for x in 0..state.puzzle.canvas_width as i32 {
                    let abs_cells: Vec<(i32, i32)> = rotated_cells
                        .iter()
                        .map(|&(dx, dy)| (x + dx, y + dy))
                        .collect();

                    let valid = abs_cells.iter().all(|c| uncovered_set.contains(c));
                    if !valid {
                        continue;
                    }

                    let mut cols: Vec<usize> = abs_cells.iter().map(|c| cell_to_col[c]).collect();
                    cols.push(piece_to_col[piece_id]);
                    cols.sort();

                    rows.push(cols);
                    placements.push(PlacementInfo {
                        piece_id: piece_id.to_string(),
                        x,
                        y,
                        rotation,
                    });
                }
            }
        }
    }

    (num_columns, rows, placements)
}

const MAX_DLX_SOLUTIONS: usize = 100;

/// Solve a tangram state using DLX. Returns target placements (one per piece).
///
/// Enumerates up to MAX_DLX_SOLUTIONS valid tilings and picks the one
/// with the lowest total rotation cost relative to current queue rotations.
pub fn dlx_solve(state: &TangramState, max_ops: usize) -> Option<Vec<Placement>> {
    if state.remaining_piece_ids.is_empty() {
        if state.occupied_cells == state.puzzle.target_set {
            return Some(vec![]);
        }
        return None;
    }

    let (num_cols, rows, infos) = build_cover_matrix(state);
    if num_cols == 0 {
        return Some(vec![]);
    }

    let mut dlx = DLX::new(num_cols, &rows, max_ops);
    let all = dlx.find_all(MAX_DLX_SOLUTIONS);
    if all.is_empty() {
        return None;
    }

    let best = all
        .into_iter()
        .min_by_key(|sol| rotation_cost(sol, &infos, state))
        .unwrap();

    Some(to_placements(&best, &infos, state))
}

/// Partial-cover DLX solve for unsolvable certificate puzzles.
///
/// Uses the original (unperturbed) pieces to find the best partial tiling
/// of the uncovered target cells. Piece columns are primary (each piece
/// must be placed or explicitly skipped), cell columns are secondary
/// (covered at most once). Skip rows are added last so DLX prefers placing.
///
/// Returns placements for pieces that were placed (not skipped), filtered
/// to only pieces that exist in the actual game.
pub fn dlx_partial_solve(
    state: &TangramState,
    original_pieces: &[PieceDef],
    max_ops: usize,
) -> Option<Vec<Placement>> {
    // Uncovered target cells in actual state
    let uncovered: Vec<(i32, i32)> = {
        let mut cells: Vec<(i32, i32)> = state
            .puzzle
            .target_set
            .iter()
            .filter(|c| !state.occupied_cells.contains(c))
            .copied()
            .collect();
        cells.sort();
        cells
    };
    let uncovered_set: HashSet<(i32, i32)> = uncovered.iter().copied().collect();

    // Figure out which original pieces are still remaining (not placed on board).
    // Only include original pieces that:
    // 1. Are still remaining (not yet placed) in the actual game
    // 2. Have the same shape as the actual piece (not corrupted)
    // Corrupted pieces are auto-skipped — DLX won't try to place them.
    let actual_ids: HashSet<&String> = state.remaining_piece_ids.iter().collect();
    let remaining_original: Vec<&PieceDef> = original_pieces
        .iter()
        .filter(|orig| {
            if !actual_ids.contains(&orig.id) {
                return false;
            }
            // Check shape matches actual piece
            match state.puzzle.pieces.iter().find(|p| p.id == orig.id) {
                Some(actual) => {
                    let mut orig_sorted = orig.cells.clone();
                    orig_sorted.sort();
                    let mut actual_sorted = actual.cells.clone();
                    actual_sorted.sort();
                    orig_sorted == actual_sorted
                }
                None => false,
            }
        })
        .collect();

    if remaining_original.is_empty() {
        return Some(vec![]);
    }

    let num_pieces = remaining_original.len();
    let num_cells = uncovered.len();
    // Primary: piece columns (0..num_pieces)
    // Secondary: cell columns (num_pieces..num_pieces+num_cells)
    let num_primary = num_pieces;
    let num_secondary = num_cells;

    let cell_to_col: std::collections::HashMap<(i32, i32), usize> = uncovered
        .iter()
        .enumerate()
        .map(|(i, &c)| (c, num_pieces + i))
        .collect();

    let piece_to_col: std::collections::HashMap<&String, usize> = remaining_original
        .iter()
        .enumerate()
        .map(|(i, p)| (&p.id, i))
        .collect();

    let rotations = [0, 90, 180, 270];
    let mut rows = Vec::new();
    let mut infos: Vec<Option<PlacementInfo>> = Vec::new(); // None for skip rows

    // Placement rows for each piece
    for piece in &remaining_original {
        let piece_col = piece_to_col[&piece.id];
        let mut seen_shapes: HashSet<Vec<(i32, i32)>> = HashSet::new();

        for &rotation in &rotations {
            let rotated_cells = piece.rotated_cells(rotation);
            let mut sorted = rotated_cells.clone();
            sorted.sort();
            if !seen_shapes.insert(sorted) {
                continue;
            }

            for y in 0..state.puzzle.canvas_height as i32 {
                for x in 0..state.puzzle.canvas_width as i32 {
                    let abs_cells: Vec<(i32, i32)> = rotated_cells
                        .iter()
                        .map(|&(dx, dy)| (x + dx, y + dy))
                        .collect();

                    // All cells must be within uncovered target
                    let valid = abs_cells.iter().all(|c| uncovered_set.contains(c));
                    if !valid {
                        continue;
                    }

                    let mut cols: Vec<usize> = abs_cells.iter().map(|c| cell_to_col[c]).collect();
                    cols.push(piece_col);
                    cols.sort();

                    rows.push(cols);
                    infos.push(Some(PlacementInfo {
                        piece_id: piece.id.clone(),
                        x,
                        y,
                        rotation,
                    }));
                }
            }
        }
    }

    // Skip rows (added LAST so DLX prefers placing)
    for piece in &remaining_original {
        let piece_col = piece_to_col[&piece.id];
        rows.push(vec![piece_col]);
        infos.push(None); // marker: this is a skip row
    }

    let mut dlx = DLX::with_secondary(num_primary, num_secondary, &rows, max_ops);
    let solution = dlx.solve()?;

    // Extract placements (non-skip rows only)
    let mut placements: Vec<Placement> = solution
        .iter()
        .filter_map(|&row_id| {
            infos[row_id].as_ref().map(|p| Placement {
                piece_id: p.piece_id.clone(),
                x: p.x,
                y: p.y,
                rotation: p.rotation,
            })
        })
        .collect();
    sort_placements_by_piece_size(&mut placements, original_pieces);
    Some(placements)
}

fn rotation_cost(sol: &[usize], infos: &[PlacementInfo], state: &TangramState) -> usize {
    sol.iter()
        .map(|&i| {
            let p = &infos[i];
            let piece = state
                .puzzle
                .pieces
                .iter()
                .find(|pd| pd.id == p.piece_id)
                .unwrap();
            let qr = state.queue_rotation(&p.piece_id);
            effective_min_rotations(piece, qr, p.rotation)
        })
        .sum()
}

fn to_placements(sol: &[usize], infos: &[PlacementInfo], state: &TangramState) -> Vec<Placement> {
    let mut result: Vec<Placement> = sol
        .iter()
        .map(|&i| {
            let p = &infos[i];
            Placement {
                piece_id: p.piece_id.clone(),
                x: p.x,
                y: p.y,
                rotation: p.rotation,
            }
        })
        .collect();
    sort_placements_by_piece_size(&mut result, &state.puzzle.pieces);
    result
}

/// Sort placements so largest pieces come first.
fn sort_placements_by_piece_size(placements: &mut [Placement], pieces: &[PieceDef]) {
    placements.sort_by(|a, b| {
        let size_a = pieces
            .iter()
            .find(|p| p.id == a.piece_id)
            .map_or(0, |p| p.cells.len());
        let size_b = pieces
            .iter()
            .find(|p| p.id == b.piece_id)
            .map_or(0, |p| p.cells.len());
        size_b.cmp(&size_a)
    });
}

/// Convert target placements to a sequence of Rotate+Place actions from a given state.
pub fn decompose_to_actions(state: &TangramState, targets: &[Placement]) -> Vec<TangramAction> {
    let mut actions = Vec::new();

    for target in targets {
        let piece = match state.puzzle.pieces.iter().find(|p| p.id == target.piece_id) {
            Some(p) => p,
            None => continue,
        };

        if let Some(placed) = state.placed.iter().find(|p| p.piece_id == target.piece_id) {
            let effective_target_rot =
                best_equivalent_rotation(piece, placed.rotation, target.rotation);
            if placed.x == target.x
                && placed.y == target.y
                && placed.rotation == effective_target_rot
            {
                continue;
            }
            if placed.x == target.x && placed.y == target.y {
                emit_rotations(
                    &mut actions,
                    &target.piece_id,
                    placed.rotation,
                    effective_target_rot,
                );
            } else {
                actions.push(TangramAction::Place {
                    piece_id: target.piece_id.clone(),
                    x: target.x,
                    y: target.y,
                });
                emit_rotations(
                    &mut actions,
                    &target.piece_id,
                    placed.rotation,
                    effective_target_rot,
                );
            }
        } else if state.remaining_piece_ids.contains(&target.piece_id) {
            let current_rot = state.queue_rotation(&target.piece_id);
            let effective_target_rot =
                best_equivalent_rotation(piece, current_rot, target.rotation);
            emit_rotations(
                &mut actions,
                &target.piece_id,
                current_rot,
                effective_target_rot,
            );
            actions.push(TangramAction::Place {
                piece_id: target.piece_id.clone(),
                x: target.x,
                y: target.y,
            });
        }
    }

    actions
}

// ---------------------------------------------------------------------------
// Partition-based piece generation helpers
// ---------------------------------------------------------------------------

/// Allocate piece sizes that sum to `total_cells`.
/// Each piece gets at least `min_size`. Remaining cells distributed randomly.
/// Auto-expands `max_size` if needed for feasibility.
fn allocate_piece_sizes(
    rng: &mut ChaChaRng,
    num_pieces: usize,
    total_cells: usize,
    piece_size_range: (usize, usize),
) -> Option<Vec<usize>> {
    let min_size = piece_size_range.0;
    let max_size = piece_size_range.1;

    if num_pieces * min_size > total_cells {
        return None;
    }

    // Auto-expand max_size if needed
    let effective_max = if num_pieces * max_size < total_cells {
        (total_cells + num_pieces - 1) / num_pieces + 1
    } else {
        max_size
    };

    let mut sizes = vec![min_size; num_pieces];
    let mut remaining = total_cells - num_pieces * min_size;

    while remaining > 0 {
        let idx = rng.gen_range(0..num_pieces);
        if sizes[idx] < effective_max {
            sizes[idx] += 1;
            remaining -= 1;
        }
    }

    Some(sizes)
}

/// Partition a `width`×`height` grid into connected regions of given target sizes.
/// Uses round-robin frontier growth from random seed cells.
fn partition_grid(
    rng: &mut ChaChaRng,
    width: usize,
    height: usize,
    target_sizes: &[usize],
) -> Option<Vec<Vec<(i32, i32)>>> {
    let num_regions = target_sizes.len();
    let total_cells = width * height;

    // Pick distinct random seed cells
    let mut seeds = Vec::with_capacity(num_regions);
    let mut used: HashSet<(i32, i32)> = HashSet::new();
    for _ in 0..num_regions {
        let mut attempts = 0;
        loop {
            let x = rng.gen_range(0..width as i32);
            let y = rng.gen_range(0..height as i32);
            let cell = (x, y);
            if !used.contains(&cell) {
                seeds.push(cell);
                used.insert(cell);
                break;
            }
            attempts += 1;
            if attempts > total_cells * 2 {
                return None;
            }
        }
    }

    // Initialize regions
    let mut regions: Vec<Vec<(i32, i32)>> = seeds.iter().map(|&s| vec![s]).collect();
    let mut assigned: HashSet<(i32, i32)> = used;

    let deltas: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
    let max_iters = total_cells * 4;
    let mut iter_count = 0;

    loop {
        let mut any_grew = false;

        for r in 0..num_regions {
            if regions[r].len() >= target_sizes[r] {
                continue;
            }

            // Collect frontier: unassigned neighbors of current region
            let mut frontier: Vec<(i32, i32)> = Vec::new();
            for &(cx, cy) in &regions[r] {
                for &(dx, dy) in &deltas {
                    let nx = cx + dx;
                    let ny = cy + dy;
                    if nx >= 0
                        && ny >= 0
                        && nx < width as i32
                        && ny < height as i32
                        && !assigned.contains(&(nx, ny))
                    {
                        frontier.push((nx, ny));
                    }
                }
            }
            frontier.sort();
            frontier.dedup();

            if frontier.is_empty() {
                return None;
            }

            let idx = rng.gen_range(0..frontier.len());
            let cell = frontier[idx];
            regions[r].push(cell);
            assigned.insert(cell);
            any_grew = true;
        }

        let all_done = regions.iter().zip(target_sizes).all(|(r, &t)| r.len() >= t);
        if all_done {
            break;
        }

        if !any_grew {
            return None;
        }

        iter_count += 1;
        if iter_count > max_iters {
            return None;
        }
    }

    Some(regions)
}

// ---------------------------------------------------------------------------
// Tangram struct and Game trait impl
// ---------------------------------------------------------------------------

pub struct Tangram {
    pub num_pieces_range: (usize, usize),
    pub piece_size_range: (usize, usize),
    pub canvas_size_range: (usize, usize),
    pub solve_budget: usize,
    pub certificate_game: bool,
}

#[cfg(test)]
impl Default for Tangram {
    fn default() -> Self {
        Tangram {
            num_pieces_range: (3, 7),
            piece_size_range: (2, 12),
            canvas_size_range: (5, 5),
            solve_budget: 1_000_000,
            certificate_game: false,
        }
    }
}

/// Solver state budget.
const SOLVE_BUDGET: usize = 1_000_000;
/// Cells per piece.
const PIECE_SIZE_RANGE: (usize, usize) = (2, 12);

pub fn build_factory(
    name: &str,
    params: &[super::registry::ConfigParam],
    overrides: &std::collections::HashMap<String, super::registry::ConfigValue>,
) -> Result<Box<dyn crate::stream::loader::GameFactory>, String> {
    let (lo, hi) = super::registry::resolve_int_range(params, overrides, "num_pieces_range")?;
    let (cs_lo, cs_hi) =
        super::registry::resolve_int_range(params, overrides, "canvas_size_range")?;
    let certificate_game =
        super::registry::resolve_int(params, overrides, "certificate_game")? != 0;
    let game = Tangram {
        num_pieces_range: (lo as usize, hi as usize),
        piece_size_range: PIECE_SIZE_RANGE,
        canvas_size_range: (cs_lo as usize, cs_hi as usize),
        solve_budget: SOLVE_BUDGET,
        certificate_game,
    };
    Ok(crate::stream::loader::game_factory(
        game,
        name,
        SOLVE_BUDGET,
    ))
}

inventory::submit! {
    super::registry::GameEntry {
        name: "tangram",
        params: &[
            super::registry::ConfigParam {
                key: "num_pieces_range",
                param_type: super::registry::ConfigParamType::IntRange,
                default: super::registry::ConfigValue::IntRange(3, 7),
            },
            super::registry::ConfigParam {
                key: "canvas_size_range",
                param_type: super::registry::ConfigParamType::IntRange,
                default: super::registry::ConfigValue::IntRange(5, 5),
            },
            super::registry::ConfigParam {
                key: "certificate_game",
                param_type: super::registry::ConfigParamType::Int,
                default: super::registry::ConfigValue::Int(0),
            },
        ],
        build: build_factory,
    }
}

impl Game for Tangram {
    type State = TangramState;
    type Action = TangramAction;

    fn generate(&self, rng: &mut ChaChaRng) -> Option<TangramState> {
        use rand::Rng;
        let canvas_size = rng.gen_range(self.canvas_size_range.0..=self.canvas_size_range.1);
        if self.certificate_game {
            generate_certificate(
                rng,
                canvas_size,
                self.num_pieces_range,
                self.piece_size_range,
                self.solve_budget,
            )
        } else {
            generate(
                rng,
                canvas_size,
                self.num_pieces_range,
                self.piece_size_range,
                self.solve_budget,
            )
        }
    }

    /// Deterministic puzzle-identity fingerprint.
    ///
    /// Raw (no symmetry canonicalization); includes colors (`color_map`) and
    /// `solvable`. Excludes `target_placements`, `tray_layout` and
    /// `occupied_cells`. HashSets are collected and sorted; no floats or
    /// pointers reach the hash.
    fn puzzle_fingerprint(&self, state: &TangramState) -> u64 {
        use std::hash::Hasher;
        crate::base::fingerprint::fingerprint(|h| {
            let puzzle = &*state.puzzle;

            // 1. Board geometry.
            h.write_u64(puzzle.canvas_width as u64);
            h.write_u64(puzzle.canvas_height as u64);

            // 2. Target silhouette (HashSet -> sorted Vec).
            let mut target: Vec<(i32, i32)> = puzzle.target_set.iter().copied().collect();
            target.sort_unstable();
            h.write_u64(target.len() as u64);
            for (x, y) in target {
                h.write_u32(x as u32);
                h.write_u32(y as u32);
            }

            // 3. Pieces, in stored order (raw identity: ids + as-stored cells,
            //    sorted only for representation determinism, not normalized).
            h.write_u64(puzzle.pieces.len() as u64);
            for piece in &puzzle.pieces {
                h.write_u64(piece.id.len() as u64);
                h.write(piece.id.as_bytes());
                let mut cells = piece.cells.clone();
                cells.sort_unstable();
                h.write_u64(cells.len() as u64);
                for (x, y) in cells {
                    h.write_u32(x as u32);
                    h.write_u32(y as u32);
                }
            }

            // 4. Colors: per-puzzle shuffled palette (index i <-> piece P{i}).
            h.write_u64(puzzle.color_map.len() as u64);
            for &c in &puzzle.color_map {
                h.write_u64(c as u64);
            }

            // 5. Solvable flag.
            h.write_u8(puzzle.solvable as u8);

            // 6. Placed pieces (sorted by (piece_id, x, y, rotation)).
            let mut placed = state.placed.clone();
            placed.sort_by(|a, b| {
                a.piece_id
                    .cmp(&b.piece_id)
                    .then(a.x.cmp(&b.x))
                    .then(a.y.cmp(&b.y))
                    .then(a.rotation.cmp(&b.rotation))
            });
            h.write_u64(placed.len() as u64);
            for p in &placed {
                h.write_u64(p.piece_id.len() as u64);
                h.write(p.piece_id.as_bytes());
                h.write_u32(p.x as u32);
                h.write_u32(p.y as u32);
                h.write_u32(p.rotation as u32);
            }

            // 7. Remaining piece ids (set semantics -> sorted).
            let mut remaining = state.remaining_piece_ids.clone();
            remaining.sort();
            h.write_u64(remaining.len() as u64);
            for id in &remaining {
                h.write_u64(id.len() as u64);
                h.write(id.as_bytes());
            }

            // 8. Queue rotations (sorted by piece id).
            let mut queue = state.queue_rotations.clone();
            queue.sort_by(|a, b| a.0.cmp(&b.0));
            h.write_u64(queue.len() as u64);
            for (id, rot) in &queue {
                h.write_u64(id.len() as u64);
                h.write(id.as_bytes());
                h.write_u64(*rot as i64 as u64);
            }
        })
    }

    fn trivial(&self) -> TangramState {
        // 1-piece puzzle: single 1x1 piece, 1x1 target at (0,0).
        let pieces = vec![PieceDef {
            id: "P0".to_string(),
            cells: vec![(0, 0)],
        }];
        let mut target_set = HashSet::new();
        target_set.insert((0, 0));
        let puzzle = Arc::new(TangramPuzzle {
            canvas_width: 1,
            canvas_height: 1,
            pieces,
            target_set,
            color_map: vec![0],
            target_placements: vec![Placement {
                piece_id: "P0".to_string(),
                x: 0,
                y: 0,
                rotation: 0,
            }],
            solvable: true,
            tray_layout: std::sync::OnceLock::new(),
        });
        TangramState {
            puzzle,
            placed: vec![],
            remaining_piece_ids: vec!["P0".to_string()],
            queue_rotations: vec![("P0".to_string(), 0)],
            occupied_cells: HashSet::new(),
        }
    }

    fn actions(&self, state: &TangramState) -> Vec<TangramAction> {
        let mut result = Vec::new();
        let w = state.puzzle.canvas_width as i32;
        let h = state.puzzle.canvas_height as i32;

        // For all pieces (placed + remaining): rotate CW/CCW by 90°
        for piece in &state.puzzle.pieces {
            result.push(TangramAction::RotateCW {
                piece_id: piece.id.clone(),
            });
            result.push(TangramAction::RotateCCW {
                piece_id: piece.id.clone(),
            });
        }

        // For all pieces: place at all (x, y) positions
        for piece in &state.puzzle.pieces {
            for y in 0..h {
                for x in 0..w {
                    result.push(TangramAction::Place {
                        piece_id: piece.id.clone(),
                        x,
                        y,
                    });
                }
            }
        }

        result
    }

    fn apply(&self, state: &TangramState, action: &TangramAction) -> Option<TangramState> {
        match action {
            TangramAction::Place { piece_id, x, y } => {
                let piece = state.puzzle.pieces.iter().find(|p| p.id == *piece_id)?;

                // Determine rotation: from queue if in queue, from current placement if on board
                let rotation;
                let mut new_placed = state.placed.clone();
                let mut new_remaining = state.remaining_piece_ids.clone();
                let mut new_queue = state.queue_rotations.clone();

                if let Some(idx) = state
                    .remaining_piece_ids
                    .iter()
                    .position(|id| id == piece_id)
                {
                    // Piece is in queue
                    rotation = state.queue_rotation(piece_id);
                    new_remaining.remove(idx);
                    new_queue.retain(|(k, _)| k != piece_id);
                } else if let Some(idx) = state.placed.iter().position(|p| p.piece_id == *piece_id)
                {
                    // Piece is on board — pick up and re-place
                    rotation = state.placed[idx].rotation;
                    new_placed.remove(idx);
                } else {
                    return None;
                }

                // Compute occupied cells for this placement
                let abs_cells: Vec<(i32, i32)> = piece
                    .rotated_cells(rotation)
                    .into_iter()
                    .map(|(dx, dy)| (x + dx, y + dy))
                    .collect();

                // Bounds check
                let w = state.puzzle.canvas_width as i32;
                let h = state.puzzle.canvas_height as i32;
                for &(cx, cy) in &abs_cells {
                    if cx < 0 || cx >= w || cy < 0 || cy >= h {
                        return None;
                    }
                }

                let new_placement = Placement {
                    piece_id: piece_id.clone(),
                    x: *x,
                    y: *y,
                    rotation,
                };
                new_placed.push(new_placement);
                new_placed.sort_by(|a, b| a.piece_id.cmp(&b.piece_id));

                let mut new_state = TangramState {
                    puzzle: state.puzzle.clone(),
                    placed: new_placed,
                    remaining_piece_ids: new_remaining,
                    queue_rotations: new_queue,
                    occupied_cells: HashSet::new(),
                };
                new_state.recompute_occupied();

                Some(new_state)
            }
            TangramAction::RotateCW { piece_id } => {
                let step = 90;
                if state.remaining_piece_ids.contains(piece_id) {
                    // Rotate in queue
                    let current = state.queue_rotation(piece_id);
                    let new_rot = (current + step) % 360;
                    let mut new_state = state.clone();
                    new_state.set_queue_rotation(piece_id, new_rot);
                    Some(new_state)
                } else if let Some(idx) = state.placed.iter().position(|p| p.piece_id == *piece_id)
                {
                    // Rotate on board (always 90°)
                    let piece = state.puzzle.pieces.iter().find(|p| p.id == *piece_id)?;
                    let mut new_placed = state.placed.clone();
                    let old_rot = new_placed[idx].rotation;
                    let new_rot = (old_rot + step) % 360;
                    new_placed[idx].rotation = new_rot;

                    // Check bounds after rotation
                    let abs_cells = new_placed[idx].occupied_cells(piece);
                    let w = state.puzzle.canvas_width as i32;
                    let h = state.puzzle.canvas_height as i32;
                    for &(cx, cy) in &abs_cells {
                        if cx < 0 || cx >= w || cy < 0 || cy >= h {
                            return None;
                        }
                    }

                    let mut new_state = TangramState {
                        puzzle: state.puzzle.clone(),
                        placed: new_placed,
                        remaining_piece_ids: state.remaining_piece_ids.clone(),
                        queue_rotations: state.queue_rotations.clone(),
                        occupied_cells: HashSet::new(),
                    };
                    new_state.recompute_occupied();
                    Some(new_state)
                } else {
                    None
                }
            }
            TangramAction::RotateCCW { piece_id } => {
                let step = 90;
                if state.remaining_piece_ids.contains(piece_id) {
                    let current = state.queue_rotation(piece_id);
                    let new_rot = (current + 360 - step) % 360;
                    let mut new_state = state.clone();
                    new_state.set_queue_rotation(piece_id, new_rot);
                    Some(new_state)
                } else if let Some(idx) = state.placed.iter().position(|p| p.piece_id == *piece_id)
                {
                    let piece = state.puzzle.pieces.iter().find(|p| p.id == *piece_id)?;
                    let mut new_placed = state.placed.clone();
                    let old_rot = new_placed[idx].rotation;
                    let new_rot = (old_rot + 360 - step) % 360;
                    new_placed[idx].rotation = new_rot;

                    let abs_cells = new_placed[idx].occupied_cells(piece);
                    let w = state.puzzle.canvas_width as i32;
                    let h = state.puzzle.canvas_height as i32;
                    for &(cx, cy) in &abs_cells {
                        if cx < 0 || cx >= w || cy < 0 || cy >= h {
                            return None;
                        }
                    }

                    let mut new_state = TangramState {
                        puzzle: state.puzzle.clone(),
                        placed: new_placed,
                        remaining_piece_ids: state.remaining_piece_ids.clone(),
                        queue_rotations: state.queue_rotations.clone(),
                        occupied_cells: HashSet::new(),
                    };
                    new_state.recompute_occupied();
                    Some(new_state)
                } else {
                    None
                }
            }
        }
    }

    fn is_solved(&self, state: &TangramState) -> bool {
        state.remaining_piece_ids.is_empty()
            && !state.has_overlap()
            && state.occupied_cells == state.puzzle.target_set
    }

    fn render(&self, state: &TangramState, resolution: u32) -> RgbBuffer {
        render::render(state, resolution)
    }

    fn action_to_str(&self, action: &TangramAction) -> String {
        match action {
            TangramAction::Place { piece_id, x, y } => {
                format!("place('{}', {}, {})", piece_id, x, y)
            }
            TangramAction::RotateCW { piece_id } => {
                format!("rotatecw('{}')", piece_id)
            }
            TangramAction::RotateCCW { piece_id } => {
                format!("rotateccw('{}')", piece_id)
            }
        }
    }

    fn parse_action(&self, _state: &TangramState, s: &str) -> Option<TangramAction> {
        if let Some(inner) = s.strip_prefix("place('") {
            // place('P0', 3, 2)
            let rest = inner.strip_suffix(')')?;
            let parts: Vec<&str> = rest.splitn(2, "', ").collect();
            if parts.len() != 2 {
                return None;
            }
            let piece_id = parts[0].to_string();
            let nums: Vec<&str> = parts[1].split(", ").collect();
            if nums.len() != 2 {
                return None;
            }
            let x = nums[0].parse().ok()?;
            let y = nums[1].parse().ok()?;
            Some(TangramAction::Place { piece_id, x, y })
        } else if let Some(inner) = s.strip_prefix("rotatecw('") {
            let piece_id = inner.strip_suffix("')")?.to_string();
            Some(TangramAction::RotateCW { piece_id })
        } else if let Some(inner) = s.strip_prefix("rotateccw('") {
            let piece_id = inner.strip_suffix("')")?.to_string();
            Some(TangramAction::RotateCCW { piece_id })
        } else {
            None
        }
    }

    fn action_templates(&self, state: &TangramState) -> Vec<ActionTemplate> {
        let piece_ids: Vec<String> = state.puzzle.pieces.iter().map(|p| p.id.clone()).collect();

        if piece_ids.is_empty() {
            return vec![];
        }

        let w = state.puzzle.canvas_width as i32;
        let h = state.puzzle.canvas_height as i32;

        vec![
            // place('P0', x, y)
            ActionTemplate {
                segments: vec![
                    ParamType::Literal("place('".into()),
                    ParamType::OneOf(piece_ids.clone()),
                    ParamType::Literal("', ".into()),
                    ParamType::Int { min: 0, max: w - 1 },
                    ParamType::Literal(", ".into()),
                    ParamType::Int { min: 0, max: h - 1 },
                    ParamType::Literal(")".into()),
                ],
            },
            // rotatecw('P0')
            ActionTemplate {
                segments: vec![
                    ParamType::Literal("rotatecw('".into()),
                    ParamType::OneOf(piece_ids.clone()),
                    ParamType::Literal("')".into()),
                ],
            },
            // rotateccw('P0')
            ActionTemplate {
                segments: vec![
                    ParamType::Literal("rotateccw('".into()),
                    ParamType::OneOf(piece_ids),
                    ParamType::Literal("')".into()),
                ],
            },
        ]
    }

    fn prompt(&self, _state: &TangramState) -> String {
        if self.certificate_game {
            "Determine if the pieces can fill the silhouette. Say 'yes' if they can, 'no' if they can't.".to_string()
        } else {
            "Complete the tangram. Fill the grey silhouette using the colored pieces.".to_string()
        }
    }

    fn action_description(&self, _state: &TangramState) -> String {
        if self.certificate_game {
            "place('piece', x, y), rotatecw('piece'), rotateccw('piece'), yes, no".to_string()
        } else {
            "place('piece', x, y), rotatecw('piece'), rotateccw('piece')".to_string()
        }
    }

    fn object_boxes(&self, state: &TangramState) -> Vec<crate::base::object_boxes::ObjectBox> {
        object_boxes::object_boxes(state)
    }

    /// `c`: the canvas row-major without row breaks, with the last character
    /// of the piece id on cells covered by a placed piece, `T` on uncovered
    /// target cells and `.` elsewhere; `q_rot`: [id, rotation] of every
    /// unplaced piece.
    fn state_fields(&self, state: &TangramState) -> Vec<StateField> {
        let puzzle = &state.puzzle;
        let (width, height) = (puzzle.canvas_width, puzzle.canvas_height);
        let mut canvas: Vec<char> = (0..width * height)
            .map(|index| {
                let cell = ((index % width) as i32, (index / width) as i32);
                if puzzle.target_set.contains(&cell) {
                    'T'
                } else {
                    '.'
                }
            })
            .collect();
        for placement in &state.placed {
            let Some(piece) = puzzle
                .pieces
                .iter()
                .find(|piece| piece.id == placement.piece_id)
            else {
                continue;
            };
            let label = placement.piece_id.chars().last().unwrap_or('?');
            for (x, y) in placement.occupied_cells(piece) {
                if (0..width as i32).contains(&x) && (0..height as i32).contains(&y) {
                    canvas[y as usize * width + x as usize] = label;
                }
            }
        }
        let queue: Vec<String> = state
            .remaining_piece_ids
            .iter()
            .map(|piece_id| {
                format!(
                    "[{},{}]",
                    json_string(piece_id),
                    state.queue_rotation(piece_id)
                )
            })
            .collect();
        vec![
            ("c", json_string(&canvas.into_iter().collect::<String>())),
            ("q_rot", format!("[{}]", queue.join(","))),
        ]
    }

    fn solve(&self, state: &TangramState, budget: usize) -> Option<(TangramAction, usize)> {
        if self.certificate_game && !state.puzzle.solvable {
            // Dynamic partial-cover: DLX with actual pieces, cells are secondary
            let placements = dlx_partial_solve(state, &state.puzzle.pieces, budget)?;
            let actions = decompose_to_actions(state, &placements);
            if actions.is_empty() {
                return None;
            }
            Some((actions[0].clone(), actions.len()))
        } else {
            let target_placements = dlx_solve(state, budget)?;
            let actions = decompose_to_actions(state, &target_placements);
            if actions.is_empty() {
                return None;
            }
            Some((actions[0].clone(), actions.len()))
        }
    }

    fn solve_path(
        &self,
        state: &TangramState,
        budget: usize,
    ) -> Option<Vec<(TangramAction, usize)>> {
        if self.certificate_game && !state.puzzle.solvable {
            let placements = dlx_partial_solve(state, &state.puzzle.pieces, budget)?;
            let actions = decompose_to_actions(state, &placements);
            if actions.is_empty() {
                return None;
            }
            let n = actions.len();
            Some(
                actions
                    .into_iter()
                    .enumerate()
                    .map(|(i, a)| (a, n - i))
                    .collect(),
            )
        } else {
            let target_placements = dlx_solve(state, budget)?;
            let actions = decompose_to_actions(state, &target_placements);
            if actions.is_empty() {
                return None;
            }
            let n = actions.len();
            Some(
                actions
                    .into_iter()
                    .enumerate()
                    .map(|(i, a)| (a, n - i))
                    .collect(),
            )
        }
    }

    fn certificate_mode(&self) -> bool {
        self.certificate_game
    }

    fn is_solvable(&self, state: &TangramState) -> bool {
        state.puzzle.solvable
    }

    fn shadow_solve_cost(&self, state: &TangramState) -> Option<usize> {
        if state.puzzle.solvable {
            return None;
        }
        let placements = dlx_partial_solve(state, &state.puzzle.pieces, self.solve_budget)?;
        let actions = decompose_to_actions(state, &placements);
        if actions.is_empty() {
            Some(0)
        } else {
            Some(actions.len())
        }
    }
}

// ---------------------------------------------------------------------------
// Puzzle generation
// ---------------------------------------------------------------------------

fn generate(
    rng: &mut ChaChaRng,
    canvas_size: usize,
    num_pieces_range: (usize, usize),
    piece_size_range: (usize, usize),
    solve_budget: usize,
) -> Option<TangramState> {
    let total_cells = canvas_size * canvas_size;
    let rotations = [0, 90, 180, 270];

    for _ in 0..1000 {
        let num_pieces = rng.gen_range(num_pieces_range.0..=num_pieces_range.1);

        // 1. Allocate piece sizes summing to total grid cells
        let sizes = match allocate_piece_sizes(rng, num_pieces, total_cells, piece_size_range) {
            Some(s) => s,
            None => continue,
        };

        // 2. Partition grid into connected regions
        let regions = match partition_grid(rng, canvas_size, canvas_size, &sizes) {
            Some(r) => r,
            None => continue,
        };

        // 3. For each region → PieceDef with random rotation/anchor
        let mut pieces = Vec::with_capacity(num_pieces);
        for (i, region) in regions.iter().enumerate() {
            let id = format!("P{}", i);

            // Pick a random rotation for the piece definition
            let def_rot = rotations[rng.gen_range(0..4)];

            // Rotate the region cells, then normalize around a random anchor
            let rotated: Vec<(i32, i32)> = region
                .iter()
                .map(|&(x, y)| match def_rot {
                    0 => (x, y),
                    90 => (-y, x),
                    180 => (-x, -y),
                    270 => (y, -x),
                    _ => (x, y),
                })
                .collect();
            // Pick anchor cell nearest to centroid (most centered)
            let cx = rotated.iter().map(|c| c.0 as f32).sum::<f32>() / rotated.len() as f32;
            let cy = rotated.iter().map(|c| c.1 as f32).sum::<f32>() / rotated.len() as f32;
            let anchor_idx = rotated
                .iter()
                .enumerate()
                .min_by(|(_, &(ax, ay)), (_, &(bx, by))| {
                    let da = (ax as f32 - cx).powi(2) + (ay as f32 - cy).powi(2);
                    let db = (bx as f32 - cx).powi(2) + (by as f32 - cy).powi(2);
                    da.partial_cmp(&db).unwrap()
                })
                .map(|(i, _)| i)
                .unwrap();
            let (ax, ay) = rotated[anchor_idx];
            let normalized: Vec<(i32, i32)> =
                rotated.iter().map(|&(x, y)| (x - ax, y - ay)).collect();

            pieces.push(PieceDef {
                id,
                cells: normalized,
            });
        }

        // 4. target_set = full grid
        let target_set: HashSet<(i32, i32)> = (0..canvas_size as i32)
            .flat_map(|y| (0..canvas_size as i32).map(move |x| (x, y)))
            .collect();

        // 5. Shuffle color map
        let mut color_map: Vec<usize> = (0..num_pieces).collect();
        for i in (1..color_map.len()).rev() {
            let j = rng.gen_range(0..=i);
            color_map.swap(i, j);
        }

        // 6. Build queue_rotations (all at 0)
        let remaining_piece_ids: Vec<String> = pieces.iter().map(|p| p.id.clone()).collect();
        let queue_rotations: Vec<(String, i32)> = remaining_piece_ids
            .iter()
            .map(|id| (id.clone(), 0))
            .collect();

        let puzzle_data = TangramPuzzle {
            canvas_width: canvas_size,
            canvas_height: canvas_size,
            pieces,
            target_set,
            color_map,
            target_placements: vec![],
            solvable: true,
            tray_layout: std::sync::OnceLock::new(),
        };

        // 7. Build temporary state for DLX solve
        let tmp_state = TangramState {
            puzzle: Arc::new(puzzle_data),
            placed: vec![],
            remaining_piece_ids: remaining_piece_ids.clone(),
            queue_rotations: queue_rotations.clone(),
            occupied_cells: HashSet::new(),
        };

        // 8. DLX solve to get target_placements
        let target_placements = match dlx_solve(&tmp_state, solve_budget) {
            Some(p) => p,
            None => continue,
        };

        // Rebuild puzzle with target_placements populated
        let mut puzzle_final = (*tmp_state.puzzle).clone();
        puzzle_final.target_placements = target_placements;

        return Some(TangramState {
            puzzle: Arc::new(puzzle_final),
            placed: vec![],
            remaining_piece_ids,
            queue_rotations,
            occupied_cells: HashSet::new(),
        });
    }

    None
}

/// Perturb a solvable puzzle to make it unsolvable.
///
/// Uses the known solution to make targeted changes: picks a piece from the
/// solved tiling, then modifies its definition so it no longer fits in its
/// intended position. Verified unsolvable via DLX.
pub fn perturb_to_unsolvable(
    rng: &mut ChaChaRng,
    base_state: &TangramState,
    solve_budget: usize,
) -> Option<TangramState> {
    let base_puzzle = &base_state.puzzle;
    let deltas = [(1, 0), (-1, 0), (0, 1), (0, -1)];

    let is_connected = |cells: &[(i32, i32)]| -> bool {
        if cells.len() <= 1 {
            return true;
        }
        let set: HashSet<(i32, i32)> = cells.iter().copied().collect();
        let mut visited = HashSet::new();
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(cells[0]);
        visited.insert(cells[0]);
        while let Some((x, y)) = queue.pop_front() {
            for &(dx, dy) in &deltas {
                let n = (x + dx, y + dy);
                if set.contains(&n) && visited.insert(n) {
                    queue.push_back(n);
                }
            }
        }
        visited.len() == cells.len()
    };

    if base_puzzle.target_placements.is_empty() {
        return None;
    }

    // Build board-level occupied cells per piece from the known solution
    let placements = &base_puzzle.target_placements;
    let mut board_cells_per_piece: Vec<HashSet<(i32, i32)>> = Vec::new();
    for placement in placements {
        let piece = base_puzzle
            .pieces
            .iter()
            .find(|p| p.id == placement.piece_id)?;
        let cells: HashSet<(i32, i32)> = placement.occupied_cells(piece).into_iter().collect();
        board_cells_per_piece.push(cells);
    }

    for _ in 0..50 {
        // Pick a random piece from the solution
        let src_idx = rng.gen_range(0..placements.len());
        let placement = &placements[src_idx];
        let piece = base_puzzle
            .pieces
            .iter()
            .find(|p| p.id == placement.piece_id)?;
        let piece_list_idx = base_puzzle
            .pieces
            .iter()
            .position(|p| p.id == placement.piece_id)?;

        if piece.cells.len() <= 1 {
            continue;
        }

        // Find board-level border cells of this piece that are adjacent to another piece
        let my_board_cells = &board_cells_per_piece[src_idx];
        let other_board_cells: HashSet<(i32, i32)> = board_cells_per_piece
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != src_idx)
            .flat_map(|(_, cells)| cells.iter().copied())
            .collect();

        // Iterate in sorted order: HashSet order varies between instances, and the
        // RNG below indexes into `boundary_pairs`, so the order must be deterministic.
        let mut my_sorted_cells: Vec<(i32, i32)> = my_board_cells.iter().copied().collect();
        my_sorted_cells.sort_unstable();
        let mut boundary_pairs: Vec<((i32, i32), (i32, i32))> = Vec::new();
        for &(bx, by) in &my_sorted_cells {
            for dx in -2i32..=2 {
                for dy in -2i32..=2 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    if dx.abs() + dy.abs() > 2 {
                        continue;
                    }
                    let neighbor = (bx + dx, by + dy);
                    if other_board_cells.contains(&neighbor) {
                        boundary_pairs.push(((bx, by), neighbor));
                    }
                }
            }
        }

        if boundary_pairs.is_empty() {
            continue;
        }

        // Pick 1 or 2 adjacent boundary cells to perturb
        let num_cells = if piece.cells.len() <= 2 {
            1
        } else {
            rng.gen_range(1..=2)
        };

        let cells_to_move: Vec<((i32, i32), (i32, i32))>;

        if num_cells == 2 {
            // Find pairs of adjacent boundary cells where destinations are also adjacent
            let mut valid_pairs: Vec<[((i32, i32), (i32, i32)); 2]> = Vec::new();
            for (a_idx, &(a_my, a_nb)) in boundary_pairs.iter().enumerate() {
                for &(b_my, b_nb) in &boundary_pairs[a_idx + 1..] {
                    let src_adj = deltas
                        .iter()
                        .any(|&(dx, dy)| (a_my.0 + dx, a_my.1 + dy) == b_my);
                    let dst_adj = deltas
                        .iter()
                        .any(|&(dx, dy)| (a_nb.0 + dx, a_nb.1 + dy) == b_nb);
                    if src_adj && dst_adj {
                        valid_pairs.push([(a_my, a_nb), (b_my, b_nb)]);
                    }
                }
            }
            if !valid_pairs.is_empty() {
                let pair = valid_pairs[rng.gen_range(0..valid_pairs.len())];
                cells_to_move = pair.to_vec();
            } else {
                // Fall back to 1-cell
                let &(first_my, first_nb) = &boundary_pairs[rng.gen_range(0..boundary_pairs.len())];
                cells_to_move = vec![(first_my, first_nb)];
            }
        } else {
            let &(first_my, first_nb) = &boundary_pairs[rng.gen_range(0..boundary_pairs.len())];
            cells_to_move = vec![(first_my, first_nb)];
        }

        // Convert board coords to piece-local coords and build new cells
        let rot = ((placement.rotation % 360) + 360) % 360;
        let board_to_local = |bx: i32, by: i32| -> (i32, i32) {
            let lx = bx - placement.x;
            let ly = by - placement.y;
            match rot {
                0 => (lx, ly),
                90 => (ly, -lx),
                180 => (-lx, -ly),
                270 => (-ly, lx),
                _ => (lx, ly),
            }
        };

        let move_from: HashSet<(i32, i32)> = cells_to_move
            .iter()
            .map(|&(my, _)| board_to_local(my.0, my.1))
            .collect();
        let move_to: Vec<(i32, i32)> = cells_to_move
            .iter()
            .map(|&(_, nb)| board_to_local(nb.0, nb.1))
            .collect();

        let mut new_cells: Vec<(i32, i32)> = piece
            .cells
            .iter()
            .filter(|c| !move_from.contains(c))
            .copied()
            .collect();
        new_cells.extend(&move_to);

        if !is_connected(&new_cells) {
            continue;
        }

        // Reject pieces with interior holes: flood-fill from outside the bounding box
        // and check that all empty cells within the bbox are reachable from outside.
        {
            let cell_set: HashSet<(i32, i32)> = new_cells.iter().copied().collect();
            let min_x = new_cells.iter().map(|c| c.0).min().unwrap() - 1;
            let max_x = new_cells.iter().map(|c| c.0).max().unwrap() + 1;
            let min_y = new_cells.iter().map(|c| c.1).min().unwrap() - 1;
            let max_y = new_cells.iter().map(|c| c.1).max().unwrap() + 1;
            let mut exterior = HashSet::new();
            let mut queue = std::collections::VecDeque::new();
            queue.push_back((min_x, min_y));
            exterior.insert((min_x, min_y));
            while let Some((x, y)) = queue.pop_front() {
                for &(dx, dy) in &deltas {
                    let n = (x + dx, y + dy);
                    if n.0 >= min_x
                        && n.0 <= max_x
                        && n.1 >= min_y
                        && n.1 <= max_y
                        && !cell_set.contains(&n)
                        && exterior.insert(n)
                    {
                        queue.push_back(n);
                    }
                }
            }
            let bbox_size = (max_x - min_x + 1) * (max_y - min_y + 1);
            let filled = cell_set.len() as i32 + exterior.len() as i32;
            if filled < bbox_size {
                continue;
            }
        }

        // Check for duplicate cells
        let cell_set: HashSet<(i32, i32)> = new_cells.iter().copied().collect();
        if cell_set.len() != new_cells.len() {
            continue;
        }

        let mut new_pieces = base_puzzle.pieces.clone();
        new_pieces[piece_list_idx] = PieceDef {
            id: piece.id.clone(),
            cells: new_cells,
        };

        let remaining_ids: Vec<String> = new_pieces.iter().map(|p| p.id.clone()).collect();
        let queue_rotations: Vec<(String, i32)> = base_state
            .queue_rotations
            .iter()
            .filter(|(id, _)| remaining_ids.contains(id))
            .cloned()
            .collect();

        let puzzle = Arc::new(TangramPuzzle {
            canvas_width: base_puzzle.canvas_width,
            canvas_height: base_puzzle.canvas_height,
            pieces: new_pieces,
            target_set: base_puzzle.target_set.clone(),
            color_map: base_puzzle.color_map.clone(),
            target_placements: vec![],
            solvable: false,
            tray_layout: std::sync::OnceLock::new(),
        });

        let test_state = TangramState {
            puzzle: puzzle.clone(),
            placed: vec![],
            remaining_piece_ids: remaining_ids,
            queue_rotations,
            occupied_cells: HashSet::new(),
        };
        if dlx_solve(&test_state, solve_budget).is_none() {
            return Some(test_state);
        }
    }
    None
}

/// Generate an unsolvable tangram puzzle.
///
/// Creates a solvable puzzle, then perturbs one piece to make it unsolvable.
fn generate_unsolvable(
    rng: &mut ChaChaRng,
    canvas_size: usize,
    num_pieces_range: (usize, usize),
    piece_size_range: (usize, usize),
    solve_budget: usize,
) -> Option<TangramState> {
    for _ in 0..100 {
        let base_state = generate(
            rng,
            canvas_size,
            num_pieces_range,
            piece_size_range,
            solve_budget,
        )?;
        if let Some(unsolvable) = perturb_to_unsolvable(rng, &base_state, solve_budget) {
            return Some(unsolvable);
        }
    }
    None
}

/// Generate a certificate-mode puzzle: 50/50 solvable vs unsolvable.
fn generate_certificate(
    rng: &mut ChaChaRng,
    canvas_size: usize,
    num_pieces_range: (usize, usize),
    piece_size_range: (usize, usize),
    solve_budget: usize,
) -> Option<TangramState> {
    // Generate both variants first, then flip — guarantees exact 50/50.
    let solvable = generate(
        rng,
        canvas_size,
        num_pieces_range,
        piece_size_range,
        solve_budget,
    )?;
    let unsolvable = generate_unsolvable(
        rng,
        canvas_size,
        num_pieces_range,
        piece_size_range,
        solve_budget,
    )?;
    if rng.gen_bool(0.5) {
        Some(solvable)
    } else {
        Some(unsolvable)
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

    /// Create a known 3x2 puzzle with 3 dominoes for testing.
    fn make_test_state() -> TangramState {
        let target_set: HashSet<(i32, i32)> = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)]
            .iter()
            .copied()
            .collect();

        let pieces = vec![
            PieceDef {
                id: "P0".to_string(),
                cells: vec![(0, 0), (1, 0)],
            },
            PieceDef {
                id: "P1".to_string(),
                cells: vec![(0, 0), (1, 0)],
            },
            PieceDef {
                id: "P2".to_string(),
                cells: vec![(0, 0), (1, 0)],
            },
        ];

        let puzzle = Arc::new(TangramPuzzle {
            canvas_width: 3,
            canvas_height: 2,
            pieces,
            target_set,
            color_map: vec![0, 1, 2],
            target_placements: vec![
                Placement {
                    piece_id: "P0".to_string(),
                    x: 0,
                    y: 0,
                    rotation: 0,
                },
                Placement {
                    piece_id: "P1".to_string(),
                    x: 2,
                    y: 0,
                    rotation: 0,
                },
                Placement {
                    piece_id: "P2".to_string(),
                    x: 0,
                    y: 1,
                    rotation: 0,
                },
            ],
            solvable: true,
            tray_layout: std::sync::OnceLock::new(),
        });

        TangramState {
            puzzle,
            placed: vec![],
            remaining_piece_ids: vec!["P0".to_string(), "P1".to_string(), "P2".to_string()],
            queue_rotations: vec![
                ("P0".to_string(), 0),
                ("P1".to_string(), 0),
                ("P2".to_string(), 0),
            ],
            occupied_cells: HashSet::new(),
        }
    }

    fn make_tangram() -> Tangram {
        Tangram {
            num_pieces_range: (2, 4),
            piece_size_range: (2, 12),
            canvas_size_range: (5, 5),
            solve_budget: 1_000_000,
            certificate_game: false,
        }
    }

    #[test]
    fn state_text_marks_placed_pieces_and_target_cells() {
        let mut state = make_test_state();
        state.placed.push(Placement {
            piece_id: "P0".to_string(),
            x: 0,
            y: 0,
            rotation: 0,
        });
        state
            .remaining_piece_ids
            .retain(|piece_id| piece_id != "P0");
        state.set_queue_rotation("P2", 90);
        let fields = make_tangram().state_fields(&state);
        assert_eq!(
            crate::base::state_text::format_state_json(&fields, 2),
            r#"{"c":"00TTTT","q_rot":[["P1",0],["P2",90]],"moves":2}"#
        );
    }

    // --- Game trait tests ---

    #[test]
    fn apply_place_from_queue() {
        let game = make_tangram();
        let state = make_test_state();
        let action = TangramAction::Place {
            piece_id: "P0".to_string(),
            x: 0,
            y: 0,
        };
        let new_state = game.apply(&state, &action).unwrap();
        assert_eq!(new_state.remaining_piece_ids.len(), 2);
        assert!(!new_state.remaining_piece_ids.contains(&"P0".to_string()));
        assert_eq!(new_state.placed.len(), 1);
        assert!(new_state.occupied_cells.contains(&(0, 0)));
        assert!(new_state.occupied_cells.contains(&(1, 0)));
    }

    #[test]
    fn apply_place_out_of_bounds() {
        let game = make_tangram();
        let state = make_test_state();
        // Domino at (2, 1) would have cells (2,1) and (3,1) — (3,1) is out of bounds for 3x2
        let action = TangramAction::Place {
            piece_id: "P0".to_string(),
            x: 2,
            y: 1,
        };
        assert!(game.apply(&state, &action).is_none());
    }

    #[test]
    fn apply_rotate_cw_in_queue() {
        let game = make_tangram();
        let state = make_test_state();
        let action = TangramAction::RotateCW {
            piece_id: "P0".to_string(),
        };
        let new_state = game.apply(&state, &action).unwrap();
        assert_eq!(new_state.queue_rotation("P0"), 90);
    }

    #[test]
    fn apply_rotate_ccw_in_queue() {
        let game = make_tangram();
        let state = make_test_state();
        let action = TangramAction::RotateCCW {
            piece_id: "P0".to_string(),
        };
        let new_state = game.apply(&state, &action).unwrap();
        assert_eq!(new_state.queue_rotation("P0"), 270);
    }

    #[test]
    fn apply_rotate_on_board() {
        let game = make_tangram();
        let state = make_test_state();

        // Place piece first
        let action = TangramAction::Place {
            piece_id: "P0".to_string(),
            x: 0,
            y: 0,
        };
        let placed_state = game.apply(&state, &action).unwrap();

        // Rotate on board
        let rotate = TangramAction::RotateCW {
            piece_id: "P0".to_string(),
        };
        let rotated_state = game.apply(&placed_state, &rotate).unwrap();
        assert_eq!(rotated_state.placed[0].rotation, 90);
    }

    #[test]
    fn is_solved_positive() {
        let game = make_tangram();
        let state = make_test_state();

        // Place all 3 dominoes to cover the 3x2 target.
        // P0 horizontal at (0,0): covers (0,0),(1,0)
        let s1 = game
            .apply(
                &state,
                &TangramAction::Place {
                    piece_id: "P0".to_string(),
                    x: 0,
                    y: 0,
                },
            )
            .unwrap();

        // Rotate P1 to vertical, then place at (2,0): covers (2,0),(2,1)
        let s2 = game
            .apply(
                &s1,
                &TangramAction::RotateCW {
                    piece_id: "P1".to_string(),
                },
            )
            .unwrap();
        let s3 = game
            .apply(
                &s2,
                &TangramAction::Place {
                    piece_id: "P1".to_string(),
                    x: 2,
                    y: 0,
                },
            )
            .unwrap();

        // Place P2 horizontal at (0,1): covers (0,1),(1,1)
        let s4 = game
            .apply(
                &s3,
                &TangramAction::Place {
                    piece_id: "P2".to_string(),
                    x: 0,
                    y: 1,
                },
            )
            .unwrap();

        // Covers (0,0),(1,0),(2,0),(2,1),(0,1),(1,1) = full 3x2
        assert!(game.is_solved(&s4));
    }

    #[test]
    fn is_solved_negative() {
        let game = make_tangram();
        let state = make_test_state();
        assert!(!game.is_solved(&state));

        // Place only one piece
        let s1 = game
            .apply(
                &state,
                &TangramAction::Place {
                    piece_id: "P0".to_string(),
                    x: 0,
                    y: 0,
                },
            )
            .unwrap();
        assert!(!game.is_solved(&s1));
    }

    #[test]
    fn action_to_str_parse_roundtrip() {
        let game = make_tangram();
        let state = make_test_state();

        let actions = vec![
            TangramAction::Place {
                piece_id: "P0".to_string(),
                x: 1,
                y: 0,
            },
            TangramAction::RotateCW {
                piece_id: "P1".to_string(),
            },
            TangramAction::RotateCCW {
                piece_id: "P2".to_string(),
            },
        ];

        for action in &actions {
            let s = game.action_to_str(action);
            let parsed = game.parse_action(&state, &s).unwrap();
            assert_eq!(*action, parsed, "Roundtrip failed for: {}", s);
        }
    }

    #[test]
    fn action_templates_expand() {
        let game = make_tangram();
        let state = make_test_state();
        let templates = game.action_templates(&state);
        assert_eq!(templates.len(), 3); // place, rotatecw, rotateccw

        let all: Vec<String> = templates.iter().flat_map(|t| t.expand()).collect();
        // place: 3 pieces * 3 x * 2 y = 18
        // rotatecw: 3 pieces = 3
        // rotateccw: 3 pieces = 3
        assert_eq!(all.len(), 24);

        assert!(all.contains(&"place('P0', 0, 0)".to_string()));
        assert!(all.contains(&"place('P2', 2, 1)".to_string()));
        assert!(all.contains(&"rotatecw('P0')".to_string()));
        assert!(all.contains(&"rotateccw('P2')".to_string()));
    }

    // --- DLX integration tests ---

    #[test]
    fn dlx_solve_small_puzzle() {
        let state = make_test_state();
        let placements = dlx_solve(&state, 100_000).expect("should be solvable");
        assert_eq!(placements.len(), 3);
        let mut used: HashSet<String> = HashSet::new();
        for p in &placements {
            used.insert(p.piece_id.clone());
        }
        assert_eq!(used.len(), 3);
    }

    #[test]
    fn dlx_unsolvable_returns_none() {
        // 3x2 target but only 2 dominoes (4 cells < 6 cells needed)
        let target_set: HashSet<(i32, i32)> = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)]
            .iter()
            .copied()
            .collect();

        let puzzle = Arc::new(TangramPuzzle {
            canvas_width: 3,
            canvas_height: 2,
            pieces: vec![
                PieceDef {
                    id: "P0".to_string(),
                    cells: vec![(0, 0), (1, 0)],
                },
                PieceDef {
                    id: "P1".to_string(),
                    cells: vec![(0, 0), (1, 0)],
                },
            ],
            target_set,
            color_map: vec![0, 1],
            target_placements: vec![],
            solvable: true,
            tray_layout: std::sync::OnceLock::new(),
        });

        let state = TangramState {
            puzzle,
            placed: vec![],
            remaining_piece_ids: vec!["P0".to_string(), "P1".to_string()],
            queue_rotations: vec![("P0".to_string(), 0), ("P1".to_string(), 0)],
            occupied_cells: HashSet::new(),
        };

        assert!(dlx_solve(&state, 100_000).is_none());
    }

    #[test]
    fn decompose_to_actions_correct() {
        let state = make_test_state();
        let targets = vec![
            Placement {
                piece_id: "P0".to_string(),
                x: 0,
                y: 0,
                rotation: 90,
            },
            Placement {
                piece_id: "P1".to_string(),
                x: 1,
                y: 0,
                rotation: 0,
            },
        ];

        let actions = decompose_to_actions(&state, &targets);
        // P0: 1 CW rotation + Place. P1: Place.
        assert_eq!(actions.len(), 3);
        assert_eq!(
            actions[0],
            TangramAction::RotateCW {
                piece_id: "P0".to_string()
            }
        );
        assert_eq!(
            actions[1],
            TangramAction::Place {
                piece_id: "P0".to_string(),
                x: 0,
                y: 0
            }
        );
        assert_eq!(
            actions[2],
            TangramAction::Place {
                piece_id: "P1".to_string(),
                x: 1,
                y: 0
            }
        );
    }

    #[test]
    fn cover_matrix_construction() {
        let state = make_test_state();
        let (num_cols, rows, placements) = build_cover_matrix(&state);
        // 6 cell columns + 3 piece columns = 9
        assert_eq!(num_cols, 9);
        assert_eq!(rows.len(), placements.len());
        // Each row should cover exactly 3 columns (2 cells + 1 piece-used)
        for row in &rows {
            assert_eq!(row.len(), 3);
        }
    }

    #[test]
    fn solve_trivial() {
        let game = make_tangram();
        let state = make_test_state();
        let result = game.solve(&state, 100_000);
        assert!(result.is_some());
        let (first_action, cost) = result.unwrap();
        // Should be able to solve
        assert!(cost > 0);
        // First action should be parseable
        let _ = game.action_to_str(&first_action);
    }

    #[test]
    fn registered_in_registry() {
        assert!(crate::games::registry::find_game("tangram").is_some());
    }

    #[test]
    fn test_aspect_ratio_default() {
        let game = Tangram::default();
        assert_eq!(game.aspect_ratio(), (1, 1));
    }

    #[test]
    fn generate_produces_solvable() {
        let game = Tangram {
            num_pieces_range: (2, 3),
            piece_size_range: (2, 12),
            canvas_size_range: (4, 4),
            solve_budget: 1_000_000,
            certificate_game: false,
        };
        for seed in 0..10u64 {
            let mut rng = ChaChaRng::seed_from_u64(seed);
            let state = game.generate(&mut rng).expect("test seed should generate");
            let result = game.solve(&state, 1_000_000);
            assert!(result.is_some(), "Seed {} produced unsolvable puzzle", seed);
        }
    }

    #[test]
    fn trivial_is_solvable() {
        let game = Tangram::default();
        let state = game.trivial();
        let result = game.solve(&state, 1_000_000);
        assert!(result.is_some(), "trivial state should be solvable");
    }

    #[test]
    fn expert_does_not_always_pick_first_piece() {
        let game = make_tangram();
        let mut first_place_ids = Vec::new();
        for seed in 0..10u64 {
            let mut rng = ChaChaRng::seed_from_u64(seed);
            let state = game.generate(&mut rng).expect("test seed should generate");

            let path = game.solve_path(&state, 1_000_000).unwrap();
            let first_place = path
                .iter()
                .find(|(a, _)| matches!(a, TangramAction::Place { .. }));

            if let Some((TangramAction::Place { piece_id, .. }, _)) = first_place {
                first_place_ids.push(piece_id.clone());
            }
        }
        assert!(
            first_place_ids.iter().any(|id| id != "P0"),
            "expert always placed P0 first across 10 seeds — \
             largest piece should vary: {:?}",
            first_place_ids,
        );
    }

    // --- GameEpisode integration tests ---

    fn make_episode() -> GameEpisode<Tangram> {
        let state = make_test_state();
        let game = Arc::new(make_tangram());
        GameEpisode::new(game, state, 1_000_000)
    }

    #[test]
    fn episode_apply_undo_roundtrip() {
        let mut ep = make_episode();
        let original_remaining = ep.state.remaining_piece_ids.len();

        assert!(ep.apply("place('P0', 0, 0)"));
        assert_eq!(ep.state.remaining_piece_ids.len(), original_remaining - 1);

        ep.undo();
        assert_eq!(ep.state.remaining_piece_ids.len(), original_remaining);
    }

    #[test]
    fn episode_reset_restores_initial() {
        let mut ep = make_episode();
        let initial_remaining = ep.state.remaining_piece_ids.len();

        ep.apply("place('P0', 0, 0)");
        ep.apply("rotatecw('P1')");
        assert_ne!(ep.state.remaining_piece_ids.len(), initial_remaining);

        ep.reset();
        assert_eq!(ep.state.remaining_piece_ids.len(), initial_remaining);
        assert!(ep.history.is_empty());
    }

    #[test]
    fn episode_expert_action_returns_optimal() {
        let ep = make_episode();
        let action = ep.expert_action();
        // Should return a valid tangram action (not back/reset/done since puzzle is unsolved)
        assert!(action != "done", "Unsolved puzzle should not return 'done'");
        // Should be parseable
        let game = make_tangram();
        assert!(
            game.parse_action(&ep.state, &action).is_some()
                || action == "back"
                || action == "reset",
            "Expert action should be parseable: {}",
            action
        );
    }

    #[test]
    fn episode_expert_action_done_when_solved() {
        let mut ep = make_episode();
        // Solve the puzzle by following expert actions
        let mut steps = 0;
        while !ep.is_solved() && steps < 100 {
            let action = ep.expert_action();
            if action == "done" {
                break;
            }
            ep.apply(&action);
            steps += 1;
        }
        // After solving, expert should say "done"
        if ep.is_solved() {
            assert_eq!(ep.expert_action(), "done");
        }
    }

    #[test]
    fn episode_enumerate_includes_meta() {
        let ep = make_episode();
        let actions = ep.enumerate_actions();
        assert!(actions.contains(&"back".to_string()));
        assert!(actions.contains(&"reset".to_string()));
        assert!(actions.contains(&"done".to_string()));
        // Should have tangram actions too
        assert!(actions.contains(&"place('P0', 0, 0)".to_string()));
        assert!(actions.contains(&"rotatecw('P0')".to_string()));
        assert!(actions.contains(&"rotateccw('P0')".to_string()));
    }

    #[test]
    fn episode_cache_consistency() {
        let state = make_test_state();
        let game = Arc::new(make_tangram());
        let ep = GameEpisode::new(game.clone(), state.clone(), 1_000_000);

        let cached_cost = ep.cached_solve(&state);
        let fresh = game.solve(&state, 1_000_000).map(|(_, c)| c);
        assert_eq!(cached_cost, fresh);
    }

    // --- Rotation utility tests ---

    #[test]
    fn test_min_rotations() {
        assert_eq!(min_rotations(0, 0), 0);
        assert_eq!(min_rotations(0, 90), 1);
        assert_eq!(min_rotations(0, 180), 2);
        assert_eq!(min_rotations(0, 270), 1);
        assert_eq!(min_rotations(90, 0), 1);
        assert_eq!(min_rotations(270, 90), 2);
    }

    #[test]
    fn test_effective_min_rotations_symmetric() {
        // A 2x2 square has distinct raw cells under every rotation about (0,0);
        // a 1x1 piece is fully symmetric.
        let _square_centered = PieceDef {
            id: "S".to_string(),
            cells: vec![(-1, -1), (0, -1), (-1, 0), (0, 0)],
        };
        let dot = PieceDef {
            id: "D".to_string(),
            cells: vec![(0, 0)],
        };
        assert_eq!(effective_min_rotations(&dot, 0, 90), 0);
        assert_eq!(effective_min_rotations(&dot, 0, 180), 0);
        assert_eq!(effective_min_rotations(&dot, 0, 270), 0);

        // A 2x2 square NOT centered at origin has distinct rotations
        // (rot changes relative cell positions), so cost is nonzero
        let square = PieceDef {
            id: "S".to_string(),
            cells: vec![(0, 0), (1, 0), (0, 1), (1, 1)],
        };
        assert_eq!(effective_min_rotations(&square, 0, 90), 1);
        assert_eq!(effective_min_rotations(&square, 0, 180), 2);
    }

    #[test]
    fn test_effective_min_rotations_asymmetric() {
        let l_shape = PieceDef {
            id: "L".to_string(),
            cells: vec![(0, 0), (1, 0), (0, 1)],
        };
        assert_eq!(effective_min_rotations(&l_shape, 0, 90), 1);
        assert_eq!(effective_min_rotations(&l_shape, 0, 180), 2);
        assert_eq!(effective_min_rotations(&l_shape, 0, 270), 1);
    }

    #[test]
    fn expert_action_prefers_largest_piece() {
        // Verify expert_action() (used by teacher forcing) respects the
        // solver's largest-piece-first ordering, not iteration order.
        let mut first_place_ids = Vec::new();
        for seed in 0..10u64 {
            let mut rng = ChaChaRng::seed_from_u64(seed);
            let game = Arc::new(make_tangram());
            let state = game.generate(&mut rng).expect("test seed should generate");
            let mut ep = GameEpisode::new(game.clone(), state, 1_000_000);

            // Step through expert actions until first Place
            for _ in 0..20 {
                let action_str = ep.expert_action();
                if action_str == "done" {
                    break;
                }
                if action_str.starts_with("place(") {
                    // Extract piece_id from place('Pn', x, y)
                    let piece_id = action_str
                        .strip_prefix("place('")
                        .unwrap()
                        .split('\'')
                        .next()
                        .unwrap()
                        .to_string();
                    first_place_ids.push(piece_id);
                    break;
                }
                ep.apply(&action_str);
            }
        }
        assert!(
            first_place_ids.iter().any(|id| id != "P0"),
            "expert_action always placed P0 first across 10 seeds — \
             should prefer largest piece: {:?}",
            first_place_ids,
        );
    }

    // =========================================================================
    // Rotation equivalence
    // =========================================================================

    #[test]
    fn equivalent_rotation_bar_not_falsely_equivalent() {
        // Bar piece [(0,0), (1,0)]: rot 0 and rot 180 produce different raw cells
        // but the same normalized shape. They should NOT be considered equivalent
        // because at position (3,0) on a 4x4 canvas, rot 0 → [(3,0),(4,0)] OOB
        // while rot 180 → [(3,0),(2,0)] is valid.
        let bar = PieceDef {
            id: "B".to_string(),
            cells: vec![(0, 0), (1, 0)],
        };

        // Rot 0: [(0,0), (1,0)]  Rot 180: [(0,0), (-1,0)]
        // These are NOT the same raw cells, so best_equivalent_rotation
        // should NOT consider 0 equivalent to 180.
        let best = best_equivalent_rotation(&bar, 0, 180);
        assert_eq!(
            best, 180,
            "rot 0 and rot 180 of bar produce different cells; \
             best_equivalent_rotation(bar, 0, 180) should return 180, not {best}"
        );
    }

    #[test]
    fn cover_matrix_includes_all_distinct_rotations() {
        // On a 4x1 canvas with a bar piece, the DLX matrix should include
        // placements for BOTH horizontal orientations (rot 0 and rot 180)
        // because they produce different cells at the same position.
        let target_set: HashSet<(i32, i32)> =
            [(0, 0), (1, 0), (2, 0), (3, 0)].iter().copied().collect();
        let puzzle = TangramPuzzle {
            canvas_width: 4,
            canvas_height: 1,
            pieces: vec![
                PieceDef {
                    id: "P0".to_string(),
                    cells: vec![(0, 0), (1, 0)],
                },
                PieceDef {
                    id: "P1".to_string(),
                    cells: vec![(0, 0), (1, 0)],
                },
            ],
            target_set,
            color_map: vec![0, 1],
            target_placements: vec![],
            solvable: true,
            tray_layout: std::sync::OnceLock::new(),
        };
        let state = TangramState {
            puzzle: Arc::new(puzzle),
            placed: vec![],
            remaining_piece_ids: vec!["P0".to_string(), "P1".to_string()],
            queue_rotations: vec![("P0".to_string(), 0), ("P1".to_string(), 0)],
            occupied_cells: HashSet::new(),
        };

        // The bar at rot 0 has cells [(0,0),(1,0)] — can be placed at x=0,1,2
        // (x=3 would put cell at (4,0) OOB on 4-wide canvas)
        // The bar at rot 180 has cells [(0,0),(-1,0)] — can be placed at x=1,2,3
        // (x=0 would put cell at (-1,0) OOB)
        // So rot 180 at x=3 is valid but rot 0 at x=3 is not.
        // DLX must include both rotations to find solutions using x=3.
        let solution = dlx_solve(&state, 100_000);
        assert!(
            solution.is_some(),
            "DLX should find a solution for 2 bars on a 4x1 canvas"
        );
        let placements = solution.unwrap();
        // One valid tiling: P0 at (0,0) rot 0, P1 at (2,0) rot 0. Another: P0 at (3,0) rot 180, etc.
        // The key assertion: the DLX found SOME valid tiling.
        assert_eq!(placements.len(), 2);
    }

    #[test]
    fn centered_squares_need_no_rotations() {
        // A 3x3 square centered at origin has identical cells under all rotations,
        // so best_equivalent_rotation should always return the current rotation
        // (0 cost), and the expert should never emit rotate actions.
        let sq_cells: Vec<(i32, i32)> = (-1..=1)
            .flat_map(|y| (-1..=1).map(move |x| (x, y)))
            .collect();

        // Verify the cells are truly rotation-invariant
        let piece = PieceDef {
            id: "S".to_string(),
            cells: sq_cells.clone(),
        };
        for &rot in &[0, 90, 180, 270] {
            let mut rotated = piece.rotated_cells(rot);
            rotated.sort();
            let mut original = sq_cells.clone();
            original.sort();
            assert_eq!(
                rotated, original,
                "3x3 centered square should be invariant under rot {rot}"
            );
        }
        assert_eq!(effective_min_rotations(&piece, 0, 90), 0);
        assert_eq!(effective_min_rotations(&piece, 0, 180), 0);
        assert_eq!(effective_min_rotations(&piece, 0, 270), 0);

        // 4 such pieces tile a 6x6 grid. Expert should solve with only Place actions.
        let target_set: HashSet<(i32, i32)> =
            (0..6).flat_map(|y| (0..6).map(move |x| (x, y))).collect();
        let puzzle = TangramPuzzle {
            canvas_width: 6,
            canvas_height: 6,
            pieces: (0..4)
                .map(|i| PieceDef {
                    id: format!("P{i}"),
                    cells: sq_cells.clone(),
                })
                .collect(),
            target_set,
            color_map: vec![0, 1, 2, 3],
            target_placements: vec![],
            solvable: true,
            tray_layout: std::sync::OnceLock::new(),
        };
        let state = TangramState {
            puzzle: Arc::new(puzzle),
            placed: vec![],
            remaining_piece_ids: (0..4).map(|i| format!("P{i}")).collect(),
            queue_rotations: (0..4).map(|i| (format!("P{i}"), 0)).collect(),
            occupied_cells: HashSet::new(),
        };

        let game = Arc::new(Tangram {
            num_pieces_range: (4, 4),
            piece_size_range: (9, 9),
            canvas_size_range: (6, 6),
            solve_budget: 100_000,
            certificate_game: false,
        });
        let mut ep = GameEpisode::new(game, state, 100_000);

        let mut steps = 0;
        while !ep.is_solved() && steps < 50 {
            let action = ep.expert_action();
            if action == "done" {
                break;
            }
            assert!(
                action.starts_with("place("),
                "centered squares should need no rotations, got {action:?} at step {steps}"
            );
            assert!(
                ep.apply(&action),
                "apply({action:?}) failed at step {steps}"
            );
            steps += 1;
        }
        assert!(
            ep.is_solved(),
            "should solve 4 centered squares on 6x6 grid"
        );
        // Exactly 4 place actions + done
        assert_eq!(steps, 4, "should take exactly 4 place actions");
    }

    // --- Certificate mode tests ---

    fn make_cert_tangram() -> Tangram {
        Tangram {
            num_pieces_range: (2, 4),
            piece_size_range: (2, 12),
            canvas_size_range: (5, 5),
            solve_budget: 1_000_000,
            certificate_game: true,
        }
    }

    #[test]
    fn perturb_delete_piece_preserves_color_map_alignment() {
        let game = make_cert_tangram();
        for seed in 0..50u64 {
            let mut rng = ChaChaRng::seed_from_u64(seed);
            let base = match generate(
                &mut rng,
                game.canvas_size_range.0,
                game.num_pieces_range,
                game.piece_size_range,
                game.solve_budget,
            ) {
                Some(s) => s,
                None => continue,
            };

            let base_color_map = base.puzzle.color_map.clone();

            let mut rng2 = ChaChaRng::seed_from_u64(seed + 1000);
            if let Some(unsolvable) = perturb_to_unsolvable(&mut rng2, &base, game.solve_budget) {
                let remaining_ids: Vec<String> = unsolvable
                    .puzzle
                    .pieces
                    .iter()
                    .map(|p| p.id.clone())
                    .collect();

                for pid in &remaining_ids {
                    let numeric_idx: usize =
                        pid.strip_prefix('P').and_then(|s| s.parse().ok()).unwrap();
                    if numeric_idx < base_color_map.len()
                        && numeric_idx < unsolvable.puzzle.color_map.len()
                    {
                        assert_eq!(
                            unsolvable.puzzle.color_map[numeric_idx],
                            base_color_map[numeric_idx],
                            "seed {seed}: color_map[{numeric_idx}] changed for piece {pid} after deletion"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn perturb_unsolvable_preserves_base_queue_rotations() {
        // perturb_to_unsolvable keeps the base state's queue rotations.
        let game = make_cert_tangram();
        for seed in 0..50u64 {
            let mut rng = ChaChaRng::seed_from_u64(seed);
            let mut base = match generate(
                &mut rng,
                game.canvas_size_range.0,
                game.num_pieces_range,
                game.piece_size_range,
                game.solve_budget,
            ) {
                Some(s) => s,
                None => continue,
            };
            // Non-default rotations, so that a reset to 0 is detectable.
            for entry in &mut base.queue_rotations {
                entry.1 = 45;
            }
            if let Some(unsolvable) = perturb_to_unsolvable(&mut rng, &base, game.solve_budget) {
                for (_, rot) in &unsolvable.queue_rotations {
                    assert_eq!(
                        *rot, 45,
                        "seed {seed}: queue rotation was not preserved from base state"
                    );
                }
                return;
            }
        }
        panic!("no unsolvable puzzle generated in seeds 0..50");
    }

    #[test]
    fn certificate_generate_distribution() {
        let game = make_cert_tangram();
        let mut solvable = 0;
        let mut unsolvable = 0;
        for seed in 0..100u64 {
            let mut rng = ChaChaRng::seed_from_u64(seed);
            if let Some(state) = game.generate(&mut rng) {
                if state.puzzle.solvable {
                    solvable += 1;
                } else {
                    unsolvable += 1;
                }
            }
        }
        assert!(solvable > 20, "Too few solvable: {}", solvable);
        assert!(unsolvable > 20, "Too few unsolvable: {}", unsolvable);
    }

    #[test]
    fn certificate_unsolvable_dlx_confirms() {
        let game = make_cert_tangram();
        for seed in 0..20u64 {
            let mut rng = ChaChaRng::seed_from_u64(seed);
            if let Some(state) = generate_unsolvable(
                &mut rng,
                game.canvas_size_range.0,
                game.num_pieces_range,
                game.piece_size_range,
                game.solve_budget,
            ) {
                assert!(!state.puzzle.solvable);
                assert!(
                    dlx_solve(&state, game.solve_budget).is_none(),
                    "seed {}: DLX should find no solution for unsolvable puzzle",
                    seed
                );
            }
        }
    }

    #[test]
    fn certificate_expert_says_yes_when_solvable() {
        let game = Arc::new(make_cert_tangram());
        for seed in 0..10u64 {
            let mut rng = ChaChaRng::seed_from_u64(seed);
            let state = generate(
                &mut rng,
                game.canvas_size_range.0,
                game.num_pieces_range,
                game.piece_size_range,
                game.solve_budget,
            )
            .expect("seed should generate solvable");
            assert!(state.puzzle.solvable);

            let mut ep = GameEpisode::new(game.clone(), state, game.solve_budget);
            let mut steps = 0;
            while steps < 200 {
                let action = ep.expert_action();
                if action == "yes" {
                    break;
                }
                assert!(
                    ep.apply(&action),
                    "seed {}: action '{}' failed",
                    seed,
                    action
                );
                steps += 1;
            }
            assert_eq!(
                ep.expert_action(),
                "yes",
                "seed {}: expert should say yes when solved",
                seed
            );
        }
    }

    #[test]
    fn certificate_expert_says_no_when_unsolvable() {
        let game = Arc::new(make_cert_tangram());
        for seed in 0..10u64 {
            let mut rng = ChaChaRng::seed_from_u64(seed);
            if let Some(state) = generate_unsolvable(
                &mut rng,
                game.canvas_size_range.0,
                game.num_pieces_range,
                game.piece_size_range,
                game.solve_budget,
            ) {
                assert!(!state.puzzle.solvable);

                let mut ep = GameEpisode::new(game.clone(), state, game.solve_budget);
                let mut steps = 0;
                while steps < 200 {
                    let action = ep.expert_action();
                    if action == "no" {
                        break;
                    }
                    assert!(
                        ep.apply(&action),
                        "seed {}: action '{}' failed",
                        seed,
                        action
                    );
                    steps += 1;
                }
                assert_eq!(
                    ep.expert_action(),
                    "no",
                    "seed {}: expert should say no for unsolvable",
                    seed
                );
            }
        }
    }

    #[test]
    fn certificate_enumerate_actions_has_yes_no() {
        let game = Arc::new(make_cert_tangram());
        let mut rng = ChaChaRng::seed_from_u64(42);
        let state = game.generate(&mut rng).expect("seed should generate");
        let ep = GameEpisode::new(game, state, 1_000_000);
        let actions = ep.enumerate_actions();
        assert!(actions.contains(&"yes".to_string()));
        assert!(actions.contains(&"no".to_string()));
        assert!(!actions.contains(&"done".to_string()));
    }

    #[test]
    fn certificate_unsolvable_never_says_back() {
        // Expert on unsolvable puzzles should never recommend back/reset,
        // even if the model places the corrupted piece first.
        let game = Arc::new(make_cert_tangram());
        for seed in 0..10u64 {
            let mut rng = ChaChaRng::seed_from_u64(seed);
            if let Some(state) = generate_unsolvable(
                &mut rng,
                game.canvas_size_range.0,
                game.num_pieces_range,
                game.piece_size_range,
                game.solve_budget,
            ) {
                let mut ep = GameEpisode::new(game.clone(), state.clone(), game.solve_budget);

                // Place ALL pieces (including corrupted) in arbitrary valid positions
                let all_piece_ids: Vec<String> =
                    state.puzzle.pieces.iter().map(|p| p.id.clone()).collect();
                for pid in &all_piece_ids {
                    // Try placing at every position until one works
                    let w = state.puzzle.canvas_width as i32;
                    let h = state.puzzle.canvas_height as i32;
                    let mut placed = false;
                    for y in 0..h {
                        for x in 0..w {
                            let action_str = format!("place('{}', {}, {})", pid, x, y);
                            if ep.apply(&action_str) {
                                placed = true;
                                break;
                            }
                        }
                        if placed {
                            break;
                        }
                    }
                }

                // Now follow expert from here — should never say back/reset
                let mut steps = 0;
                while steps < 100 {
                    let action = ep.expert_action();
                    assert!(
                        action != "back" && action != "reset",
                        "seed {}: expert said '{}' on unsolvable puzzle (should never go back)",
                        seed,
                        action
                    );
                    if action == "no" {
                        break;
                    }
                    assert!(
                        ep.apply(&action),
                        "seed {}: action '{}' failed",
                        seed,
                        action
                    );
                    steps += 1;
                }
                assert_eq!(
                    ep.expert_action(),
                    "no",
                    "seed {}: expert should eventually say no",
                    seed
                );
            }
        }
    }

    #[test]
    fn certificate_unsolvable_expert_forward_after_corrupted_piece() {
        // Place the corrupted piece first, then verify expert places remaining
        // non-corrupted pieces and says "no".
        let game = Arc::new(make_cert_tangram());
        for seed in 0..10u64 {
            let mut rng = ChaChaRng::seed_from_u64(seed);
            let Some(base_state) = generate(
                &mut rng,
                game.canvas_size_range.0,
                game.num_pieces_range,
                game.piece_size_range,
                game.solve_budget,
            ) else {
                continue;
            };
            if let Some(state) = perturb_to_unsolvable(&mut rng, &base_state, game.solve_budget) {
                // Find the corrupted piece (shape differs from the unperturbed puzzle)
                let original = &base_state.puzzle.pieces;
                let corrupted_id: Option<String> = state.puzzle.pieces.iter().find_map(|actual| {
                    let orig = original.iter().find(|o| o.id == actual.id)?;
                    let mut a = actual.cells.clone();
                    a.sort();
                    let mut o = orig.cells.clone();
                    o.sort();
                    if a != o {
                        Some(actual.id.clone())
                    } else {
                        None
                    }
                });

                // Skip seeds where perturbation was delete-piece (no corrupted piece in game)
                let corrupted_id = match corrupted_id {
                    Some(id) => id,
                    None => continue,
                };

                let mut ep = GameEpisode::new(game.clone(), state.clone(), game.solve_budget);

                // Place the corrupted piece at the first valid position (try all rotations)
                let w = state.puzzle.canvas_width as i32;
                let h = state.puzzle.canvas_height as i32;
                let mut placed = false;
                'outer: for rot in 0..4 {
                    if rot > 0 {
                        let rot_str = format!("rotatecw('{}')", corrupted_id);
                        ep.apply(&rot_str);
                    }
                    for y in 0..h {
                        for x in 0..w {
                            let action_str = format!("place('{}', {}, {})", corrupted_id, x, y);
                            if ep.apply(&action_str) {
                                placed = true;
                                break 'outer;
                            }
                        }
                    }
                }
                if !placed {
                    continue;
                } // skip seed if can't place

                // Follow expert — should place remaining pieces then say "no"
                let mut steps = 0;
                while steps < 200 {
                    let action = ep.expert_action();
                    if action == "no" {
                        break;
                    }
                    assert!(
                        action != "back" && action != "reset",
                        "seed {}: expert said '{}' after placing corrupted piece",
                        seed,
                        action
                    );
                    assert!(
                        ep.apply(&action),
                        "seed {}: action '{}' failed",
                        seed,
                        action
                    );
                    steps += 1;
                }
                assert_eq!(
                    ep.expert_action(),
                    "no",
                    "seed {}: expert should say no after placing remaining pieces",
                    seed
                );
            }
        }
    }

    // --- Certificate answers are accepted from any state ---

    fn first_solvable(game: &Tangram) -> TangramState {
        for seed in 0..50u64 {
            let mut rng = ChaChaRng::seed_from_u64(seed);
            if let Some(state) = generate(
                &mut rng,
                game.canvas_size_range.0,
                game.num_pieces_range,
                game.piece_size_range,
                game.solve_budget,
            ) {
                return state;
            }
        }
        panic!("no solvable tangram cert puzzle in 50 seeds");
    }

    fn first_unsolvable(game: &Tangram) -> TangramState {
        for seed in 0..50u64 {
            let mut rng = ChaChaRng::seed_from_u64(seed);
            if let Some(state) = generate_unsolvable(
                &mut rng,
                game.canvas_size_range.0,
                game.num_pieces_range,
                game.piece_size_range,
                game.solve_budget,
            ) {
                return state;
            }
        }
        panic!("no unsolvable tangram cert puzzle in 50 seeds");
    }

    fn apply_first_place(ep: &mut GameEpisode<Tangram>) -> bool {
        let candidates: Vec<String> = ep
            .enumerate_actions()
            .into_iter()
            .filter(|s| s.starts_with("place("))
            .collect();
        for action in candidates {
            if ep.apply(&action) {
                return true;
            }
        }
        false
    }

    #[test]
    fn certificate_solvable_accepts_yes_at_initial_state() {
        let game = Arc::new(make_cert_tangram());
        let state = first_solvable(&game);
        let mut ep = GameEpisode::new(game.clone(), state, game.solve_budget);
        assert!(
            ep.apply("yes"),
            "should accept yes at initial solvable state"
        );
        assert!(ep.is_solved());
    }

    #[test]
    fn certificate_solvable_accepts_yes_after_partial_placement() {
        let game = Arc::new(make_cert_tangram());
        let state = first_solvable(&game);
        let mut ep = GameEpisode::new(game.clone(), state, game.solve_budget);
        assert!(
            apply_first_place(&mut ep),
            "should be able to place at least one piece"
        );
        assert!(
            ep.apply("yes"),
            "should accept yes mid-trajectory on a solvable puzzle"
        );
        assert!(ep.is_solved());
    }

    #[test]
    fn certificate_solvable_rejects_no() {
        let game = Arc::new(make_cert_tangram());
        let state = first_solvable(&game);
        let mut ep = GameEpisode::new(game.clone(), state, game.solve_budget);
        assert!(!ep.apply("no"), "must not accept no on a solvable puzzle");
        assert!(!ep.is_solved());
    }

    #[test]
    fn certificate_unsolvable_accepts_no_at_initial_state() {
        let game = Arc::new(make_cert_tangram());
        let state = first_unsolvable(&game);
        let mut ep = GameEpisode::new(game.clone(), state, game.solve_budget);
        assert!(
            ep.apply("no"),
            "should accept no at initial unsolvable state"
        );
        assert!(ep.is_solved());
    }

    #[test]
    fn certificate_unsolvable_rejects_yes() {
        let game = Arc::new(make_cert_tangram());
        let state = first_unsolvable(&game);
        let mut ep = GameEpisode::new(game.clone(), state, game.solve_budget);
        assert!(
            !ep.apply("yes"),
            "must not accept yes on an unsolvable puzzle"
        );
        assert!(!ep.is_solved());
    }
}
