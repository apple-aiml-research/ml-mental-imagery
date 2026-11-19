//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
use ahash::AHashMap;

use super::state::{BoardConfig, Orientation, Slide};

const MAX_DEPTH: usize = 25;
const BITS_PER_POS: usize = 3;

pub struct IddfsSolution {
    pub actions: Vec<Slide>,
}

impl IddfsSolution {
    pub fn cost(&self) -> usize {
        self.actions.len()
    }

    pub fn first_action(&self) -> &Slide {
        &self.actions[0]
    }
}

pub fn iddfs_solve(config: &BoardConfig, positions: &[u8], budget: usize) -> Option<IddfsSolution> {
    let gs = config.grid_size as usize;
    assert!(
        gs <= (1 << BITS_PER_POS) && positions.len() * BITS_PER_POS <= 64,
        "grid_size={gs} vehicles={} exceeds MemoKey capacity (increase BITS_PER_POS)",
        positions.len()
    );
    if positions[0] + config.vehicles[0].length == config.grid_size {
        return None;
    }
    if is_impossible(config, positions, gs) {
        return None;
    }
    let mut solver = Solver::new(config, positions, budget);
    solver.solve()
}

// ---------------------------------------------------------------------------
// Static analyzer: detect provably unsolvable boards via constraint propagation
// ---------------------------------------------------------------------------

fn is_impossible(config: &BoardConfig, positions: &[u8], gs: usize) -> bool {
    // Any horizontal vehicle on the exit row to A's right is a permanent blocker:
    // same-row horizontal vehicles can never swap order, so A can never pass it.
    for (i, def) in config.vehicles.iter().enumerate().skip(1) {
        if def.orientation == Orientation::Horizontal
            && def.fixed_axis == config.exit_row
            && positions[i] > positions[0]
        {
            return true;
        }
    }

    let n = gs * gs;
    let mut horz_blocked = vec![false; n];
    let mut vert_blocked = vec![false; n];

    loop {
        let mut changed = false;

        // Process rows: find horizontally-blocked squares
        for y in 0..gs {
            let mut row_positions = Vec::new();
            let mut row_sizes = Vec::new();
            for (i, def) in config.vehicles.iter().enumerate() {
                if def.orientation == Orientation::Horizontal && def.fixed_axis as usize == y {
                    row_positions.push(positions[i] as usize);
                    row_sizes.push(def.length as usize);
                }
            }
            if row_positions.is_empty() {
                continue;
            }

            // Blocked squares from perpendicular (horizontal) direction on this row
            let mut blocked = Vec::new();
            let row_off = y * gs;
            for x in 0..gs {
                if horz_blocked[row_off + x] {
                    blocked.push(x);
                }
            }

            let result = find_blocked_squares(gs, &mut row_positions, &mut row_sizes, &blocked);
            for sq in result {
                let idx = row_off + sq;
                if !vert_blocked[idx] {
                    vert_blocked[idx] = true;
                    changed = true;
                }
            }
        }

        // Process columns: find vertically-blocked squares
        for x in 0..gs {
            let mut col_positions = Vec::new();
            let mut col_sizes = Vec::new();
            for (i, def) in config.vehicles.iter().enumerate() {
                if def.orientation == Orientation::Vertical && def.fixed_axis as usize == x {
                    col_positions.push(positions[i] as usize);
                    col_sizes.push(def.length as usize);
                }
            }
            if col_positions.is_empty() {
                continue;
            }

            let mut blocked = Vec::new();
            for y in 0..gs {
                if vert_blocked[y * gs + x] {
                    blocked.push(y);
                }
            }

            let result = find_blocked_squares(gs, &mut col_positions, &mut col_sizes, &blocked);
            for sq in result {
                let idx = sq * gs + x;
                if !horz_blocked[idx] {
                    horz_blocked[idx] = true;
                    changed = true;
                }
            }
        }

        if !changed {
            break;
        }
    }

    // Check if any square between car A and the exit is blocked
    let a_pos = positions[0] as usize;
    let a_len = config.vehicles[0].length as usize;
    let exit_row = config.exit_row as usize;
    let row_off = exit_row * gs;
    for col in (a_pos + a_len)..gs {
        if horz_blocked[row_off + col] || vert_blocked[row_off + col] {
            return true;
        }
    }
    false
}

fn find_blocked_squares(
    width: usize,
    positions: &mut Vec<usize>,
    sizes: &mut Vec<usize>,
    blocked: &[usize],
) -> Vec<usize> {
    let n = positions.len();

    // Sort pieces by position
    for i in 1..n {
        let mut j = i;
        while j > 0 && positions[j] < positions[j - 1] {
            positions.swap(j, j - 1);
            sizes.swap(j, j - 1);
            j -= 1;
        }
    }

    // For each piece, determine valid placement range given blocked squares
    let mut all_placements: Vec<Vec<usize>> = Vec::with_capacity(n);
    for i in 0..n {
        let p = positions[i];
        let s = sizes[i];
        let mut x0 = 0usize;
        let mut x1 = width - s;
        for &b in blocked {
            if b < p {
                x0 = x0.max(b + 1);
            }
            if b > p {
                x1 = x1.min(b.saturating_sub(s));
            }
        }
        let placements: Vec<usize> = (x0..=x1).collect();
        all_placements.push(placements);
    }

    // Enumerate all valid placement combinations (Cartesian product)
    let mut counts = vec![0usize; width];
    let mut valid_count = 0usize;
    let mut idx = vec![0usize; n];
    let lens: Vec<usize> = all_placements.iter().map(|p| p.len()).collect();

    if lens.iter().any(|&l| l == 0) {
        return Vec::new();
    }

    loop {
        // Check no overlap between adjacent pieces
        let mut ok = true;
        for i in 1..n {
            if all_placements[i][idx[i]] < all_placements[i - 1][idx[i - 1]] + sizes[i - 1] {
                ok = false;
                break;
            }
        }
        if ok {
            valid_count += 1;
            for i in 0..n {
                let p = all_placements[i][idx[i]];
                for j in 0..sizes[i] {
                    counts[p + j] += 1;
                }
            }
        }

        // Advance to next combination
        let mut i = n as i32 - 1;
        while i >= 0 && idx[i as usize] == lens[i as usize] - 1 {
            idx[i as usize] = 0;
            i -= 1;
        }
        if i < 0 {
            break;
        }
        idx[i as usize] += 1;
    }

    // Squares occupied in every valid configuration are blocked
    let mut result = Vec::new();
    if valid_count > 0 {
        for (i, &c) in counts.iter().enumerate() {
            if c == valid_count {
                result.push(i);
            }
        }
    }
    result
}

// ---------------------------------------------------------------------------
// IDDFS solver
// ---------------------------------------------------------------------------

type MemoKey = u64;

#[inline]
fn make_key(positions: &[u8]) -> MemoKey {
    let mut key: u64 = 0;
    for (i, &pos) in positions.iter().enumerate() {
        key |= (pos as u64) << (i * BITS_PER_POS);
    }
    key
}

struct Solver<'a> {
    grid_size: usize,
    exit_row: usize,
    target_pos: u8,
    num_vehicles: usize,
    vehicles: &'a [super::state::VehicleDef],

    positions: Vec<u8>,
    occupied: Vec<i8>,

    memo: AHashMap<MemoKey, u16>,
    budget: usize,
    nodes_visited: usize,

    path: Vec<(usize, i8)>,
    move_bufs: Vec<Vec<(usize, i8)>>,
}

impl<'a> Solver<'a> {
    fn new(config: &'a BoardConfig, positions: &[u8], budget: usize) -> Self {
        let gs = config.grid_size as usize;
        let mut occupied = vec![-1i8; gs * gs];

        for (idx, def) in config.vehicles.iter().enumerate() {
            let pos = positions[idx] as usize;
            for offset in 0..def.length as usize {
                let cell = match def.orientation {
                    Orientation::Horizontal => def.fixed_axis as usize * gs + pos + offset,
                    Orientation::Vertical => (pos + offset) * gs + def.fixed_axis as usize,
                };
                occupied[cell] = idx as i8;
            }
        }

        Solver {
            grid_size: gs,
            exit_row: config.exit_row as usize,
            target_pos: config.grid_size - config.vehicles[0].length,
            num_vehicles: config.vehicles.len(),
            vehicles: &config.vehicles,
            positions: positions.to_vec(),
            occupied,
            memo: AHashMap::new(),
            budget,
            nodes_visited: 0,
            path: Vec::new(),
            move_bufs: Vec::new(),
        }
    }

    #[inline]
    fn cell_index(&self, idx: usize, pos: usize, offset: usize) -> usize {
        match self.vehicles[idx].orientation {
            Orientation::Horizontal => {
                self.vehicles[idx].fixed_axis as usize * self.grid_size + pos + offset
            }
            Orientation::Vertical => {
                (pos + offset) * self.grid_size + self.vehicles[idx].fixed_axis as usize
            }
        }
    }

    #[inline]
    fn do_move(&mut self, idx: usize, amount: i8) {
        let old_pos = self.positions[idx] as usize;
        let len = self.vehicles[idx].length as usize;

        for offset in 0..len {
            let cell = self.cell_index(idx, old_pos, offset);
            self.occupied[cell] = -1;
        }

        let new_pos = (old_pos as i32 + amount as i32) as usize;
        self.positions[idx] = new_pos as u8;

        for offset in 0..len {
            let cell = self.cell_index(idx, new_pos, offset);
            self.occupied[cell] = idx as i8;
        }
    }

    #[inline]
    fn undo_move(&mut self, idx: usize, amount: i8) {
        self.do_move(idx, -amount);
    }

    #[inline]
    fn is_solved(&self) -> bool {
        self.positions[0] == self.target_pos
    }

    #[inline]
    fn blocking_lower_bound(&self) -> usize {
        let row_offset = self.exit_row * self.grid_size;
        let a_right = self.positions[0] as usize + self.vehicles[0].length as usize;
        let mut seen = 0u32;
        for col in a_right..self.grid_size {
            let v = self.occupied[row_offset + col];
            if v >= 0 {
                seen |= 1 << v as u32;
            }
        }
        seen.count_ones() as usize
    }

    fn generate_moves_into(&self, buf: &mut Vec<(usize, i8)>, last_vehicle: usize) {
        buf.clear();
        let gs = self.grid_size;

        for idx in 0..self.num_vehicles {
            if idx == last_vehicle {
                continue;
            }
            let def = &self.vehicles[idx];
            let pos = self.positions[idx] as usize;
            let len = def.length as usize;

            match def.orientation {
                Orientation::Horizontal => {
                    let row_off = def.fixed_axis as usize * gs;
                    let mut col = pos;
                    let mut dist = 1i8;
                    while col > 0 {
                        col -= 1;
                        if self.occupied[row_off + col] >= 0 {
                            break;
                        }
                        buf.push((idx, -dist));
                        dist += 1;
                    }
                    let mut col = pos + len;
                    let mut dist = 1i8;
                    while col < gs {
                        if self.occupied[row_off + col] >= 0 {
                            break;
                        }
                        buf.push((idx, dist));
                        dist += 1;
                        col += 1;
                    }
                }
                Orientation::Vertical => {
                    let col = def.fixed_axis as usize;
                    let mut row = pos;
                    let mut dist = 1i8;
                    while row > 0 {
                        row -= 1;
                        if self.occupied[row * gs + col] >= 0 {
                            break;
                        }
                        buf.push((idx, -dist));
                        dist += 1;
                    }
                    let mut row = pos + len;
                    let mut dist = 1i8;
                    while row < gs {
                        if self.occupied[row * gs + col] >= 0 {
                            break;
                        }
                        buf.push((idx, dist));
                        dist += 1;
                        row += 1;
                    }
                }
            }
        }
    }

    fn dfs(&mut self, depth: usize, max_depth: usize, last_vehicle: usize) -> bool {
        if self.is_solved() {
            return true;
        }
        if depth == max_depth {
            return false;
        }

        let remaining = max_depth - depth;
        if self.blocking_lower_bound() >= remaining {
            return false;
        }

        let key = make_key(&self.positions);
        let remaining_u16 = remaining as u16;
        if let Some(&prev) = self.memo.get(&key) {
            if prev >= remaining_u16 {
                return false;
            }
        }
        self.memo.insert(key, remaining_u16);
        self.nodes_visited += 1;
        if self.nodes_visited >= self.budget {
            return false;
        }

        if self.move_bufs.len() <= depth {
            self.move_bufs.resize_with(depth + 1, Vec::new);
        }
        let mut buf = std::mem::take(&mut self.move_bufs[depth]);
        self.generate_moves_into(&mut buf, last_vehicle);

        for i in 0..buf.len() {
            let (vehicle_idx, amount) = buf[i];
            self.do_move(vehicle_idx, amount);
            let solved = self.dfs(depth + 1, max_depth, vehicle_idx);
            self.undo_move(vehicle_idx, amount);
            if solved {
                self.path.push((vehicle_idx, amount));
                self.move_bufs[depth] = buf;
                return true;
            }
        }
        self.move_bufs[depth] = buf;
        false
    }

    fn solve(&mut self) -> Option<IddfsSolution> {
        let mut prev_memo_size = 0;
        let mut no_change = 0;
        let cutoff = self.grid_size - self.vehicles[0].length as usize;

        for max_depth in 1..=MAX_DEPTH {
            self.path.clear();
            if self.dfs(0, max_depth, usize::MAX) {
                self.path.reverse();
                let actions = self
                    .path
                    .iter()
                    .map(|&(idx, amount)| Slide {
                        vehicle_id: self.vehicles[idx].id,
                        amount,
                    })
                    .collect();
                return Some(IddfsSolution { actions });
            }

            let memo_size = self.memo.len();
            if memo_size == prev_memo_size {
                no_change += 1;
            } else {
                no_change = 0;
            }
            if no_change > cutoff {
                return None;
            }
            prev_memo_size = memo_size;

            if self.nodes_visited >= self.budget {
                return None;
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::game::Game;
    use crate::games::rush_hour::state::{from_ascii, RushHourState};
    use crate::games::rush_hour::RushHour;

    fn solve_with_bfs(state: &RushHourState, budget: usize) -> Option<usize> {
        let game = RushHour {
            grid_size_range: (state.config.grid_size, state.config.grid_size),
            num_cars_range: (
                state.config.vehicles.len() as u8,
                state.config.vehicles.len() as u8,
            ),
            solve_budget: budget,
        };
        crate::solver::bfs::bfs_solve(&game, state, budget).map(|r| r.cost())
    }

    fn replay_solution(state: &RushHourState, solution: &IddfsSolution) -> bool {
        let game = RushHour {
            grid_size_range: (state.config.grid_size, state.config.grid_size),
            num_cars_range: (
                state.config.vehicles.len() as u8,
                state.config.vehicles.len() as u8,
            ),
            solve_budget: 100_000,
        };
        let mut current = state.clone();
        for action in &solution.actions {
            match game.apply(&current, action) {
                Some(next) => current = next,
                None => return false,
            }
        }
        game.is_solved(&current)
    }

    // --- Static analyzer tests ---

    #[test]
    fn static_analyzer_rejects_impossible() {
        // Two horizontal vehicles on exit row that can't pass
        let state = from_ascii("......\n......\nAA.BB.\n......\n......\n......");
        assert!(is_impossible(
            &state.config,
            &state.positions,
            state.config.grid_size as usize
        ));
    }

    #[test]
    fn static_analyzer_accepts_solvable() {
        let state = from_ascii("....B.\n....B.\nAA..B.\n......\n......\n......");
        assert!(!is_impossible(
            &state.config,
            &state.positions,
            state.config.grid_size as usize
        ));
    }

    #[test]
    fn static_analyzer_accepts_trivial() {
        let state = from_ascii("......\n......\nAA....\n......\n......\n......");
        assert!(!is_impossible(
            &state.config,
            &state.positions,
            state.config.grid_size as usize
        ));
    }

    // --- Known board tests ---

    #[test]
    fn single_move_solution() {
        let state = from_ascii("......\n......\n...AA.\n......\n......\n......");
        let sol = iddfs_solve(&state.config, &state.positions, 100_000).unwrap();
        assert_eq!(sol.cost(), 1);
        assert_eq!(sol.actions[0].vehicle_id, 'A');
        assert_eq!(sol.actions[0].amount, 1);
        assert!(replay_solution(&state, &sol));
    }

    #[test]
    fn already_solved_returns_none() {
        let state = from_ascii("......\n......\n....AA\n......\n......\n......");
        assert!(iddfs_solve(&state.config, &state.positions, 100_000).is_none());
    }

    #[test]
    fn multi_step_no_blockers() {
        let state = from_ascii("......\n......\nAA....\n......\n......\n......");
        let sol = iddfs_solve(&state.config, &state.positions, 100_000).unwrap();
        assert_eq!(sol.cost(), 1);
        assert_eq!(sol.actions[0].amount, 4);
        assert!(replay_solution(&state, &sol));
    }

    #[test]
    fn one_vertical_blocker() {
        let state = from_ascii("....B.\n....B.\nAA..B.\n......\n......\n......");
        let sol = iddfs_solve(&state.config, &state.positions, 100_000).unwrap();
        assert!(sol.cost() >= 2);
        assert!(replay_solution(&state, &sol));
        let bfs_cost = solve_with_bfs(&state, 100_000).unwrap();
        assert_eq!(sol.cost(), bfs_cost);
    }

    #[test]
    fn classic_6x6_puzzle() {
        let state = from_ascii("..BCCC\n..B...\nAAB..D\n.EE..D\n......\n......");
        let sol = iddfs_solve(&state.config, &state.positions, 100_000).unwrap();
        assert!(replay_solution(&state, &sol));
        let bfs_cost = solve_with_bfs(&state, 100_000).unwrap();
        assert_eq!(sol.cost(), bfs_cost);
    }

    #[test]
    fn horizontal_vehicle_on_exit_row() {
        let state = from_ascii("......\n....C.\nBBAAC.\n......\n......\n......");
        let sol = iddfs_solve(&state.config, &state.positions, 100_000).unwrap();
        assert!(replay_solution(&state, &sol));
        let bfs_cost = solve_with_bfs(&state, 100_000).unwrap();
        assert_eq!(sol.cost(), bfs_cost);
    }

    #[test]
    fn unsolvable_horizontal_blocker() {
        let state = from_ascii("......\n......\nAA.BB.\n......\n......\n......");
        assert!(iddfs_solve(&state.config, &state.positions, 100_000).is_none());
        assert!(solve_with_bfs(&state, 100_000).is_none());
    }

    #[test]
    fn budget_exhaustion_returns_none() {
        let state = from_ascii("..BCCC\n..B...\nAAB..D\n.EE..D\n......\n......");
        let result = iddfs_solve(&state.config, &state.positions, 1);
        assert!(result.is_none());
    }

    // --- solve_path format ---

    #[test]
    fn solve_path_format_matches_bfs() {
        let state = from_ascii("..BCCC\n..B...\nAAB..D\n.EE..D\n......\n......");
        let game = RushHour {
            grid_size_range: (6, 6),
            num_cars_range: (2, 14),
            solve_budget: 100_000,
        };
        let path = game.solve_path(&state, 100_000).unwrap();
        let n = path.len();
        for (i, (_, remaining)) in path.iter().enumerate() {
            assert_eq!(*remaining, n - i);
        }
        let mut current = state.clone();
        for (action, _) in &path {
            current = game
                .apply(&current, action)
                .expect("action should be valid");
        }
        assert!(game.is_solved(&current));
    }
}
