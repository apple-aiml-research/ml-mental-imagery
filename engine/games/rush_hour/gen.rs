//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
use std::sync::Arc;

use rand::Rng;
use rand_chacha::ChaChaRng;

use super::state::{BoardConfig, Orientation, RushHourState, VehicleDef};

/// Collect all valid placements for a vehicle of the given length on the grid.
fn valid_placements(occupied: &[bool], gs: usize, length: u8) -> Vec<(Orientation, u8, u8)> {
    let len = length as usize;
    let mut result = Vec::new();

    for orientation in [Orientation::Horizontal, Orientation::Vertical] {
        let axis_limit = gs;
        let pos_limit = gs - len;
        for fixed_axis in 0..axis_limit {
            for pos in 0..=pos_limit {
                let ok = (0..len).all(|offset| {
                    let (col, row) = match orientation {
                        Orientation::Horizontal => (pos + offset, fixed_axis),
                        Orientation::Vertical => (fixed_axis, pos + offset),
                    };
                    !occupied[row * gs + col]
                });
                if ok {
                    result.push((orientation, fixed_axis as u8, pos as u8));
                }
            }
        }
    }

    result
}

/// Generate a candidate Rush Hour puzzle.
///
/// 1. Place target car 'A' (horizontal, length 2) on exit_row at a random column
/// 2. Place remaining vehicles by enumerating valid positions and picking randomly
/// 3. Reject if target car is already at exit (trivially solved)
fn generate_candidate(
    rng: &mut ChaChaRng,
    grid_size: u8,
    num_cars_range: (u8, u8),
) -> Option<RushHourState> {
    let gs = grid_size as usize;
    // Bias toward more vehicles for denser, harder boards
    let biased = rng
        .gen_range(num_cars_range.0..=num_cars_range.1)
        .max(rng.gen_range(num_cars_range.0..=num_cars_range.1));
    let num_vehicles = biased as usize;

    let mut occupied = vec![false; gs * gs];
    let mut vehicles = Vec::new();
    let mut positions = Vec::new();

    // Target car 'A': horizontal, length 2, on a random exit row
    // Bias toward column 0 (far from exit) for harder puzzles
    let exit_row: u8 = rng.gen_range(0..gs) as u8;
    let target_length: u8 = 2;
    let max_col = gs - target_length as usize;
    let target_col = rng.gen_range(0..=max_col).min(rng.gen_range(0..=max_col)) as u8;

    // Reject if already at exit
    if target_col as usize + target_length as usize == gs {
        return None;
    }

    for offset in 0..target_length as usize {
        occupied[exit_row as usize * gs + target_col as usize + offset] = true;
    }

    vehicles.push(VehicleDef {
        id: 'A',
        orientation: Orientation::Horizontal,
        length: target_length,
        fixed_axis: exit_row,
    });
    positions.push(target_col);

    // Place remaining vehicles by enumerating valid positions
    let mut next_id = b'B';
    for _ in 1..num_vehicles {
        let primary_length: u8 = if rng.gen::<f64>() < 0.3 { 3 } else { 2 };
        let fallback_length: u8 = if primary_length == 3 { 2 } else { 3 };

        let mut length = primary_length;
        let mut candidates = valid_placements(&occupied, gs, length);
        if candidates.is_empty() {
            length = fallback_length;
            candidates = valid_placements(&occupied, gs, length);
        }
        if candidates.is_empty() {
            break;
        }

        let chosen = rng.gen_range(0..candidates.len());
        let (orientation, fixed_axis, pos) = candidates[chosen];

        for offset in 0..length as usize {
            let (col, row) = match orientation {
                Orientation::Horizontal => (pos as usize + offset, fixed_axis as usize),
                Orientation::Vertical => (fixed_axis as usize, pos as usize + offset),
            };
            occupied[row * gs + col] = true;
        }

        vehicles.push(VehicleDef {
            id: next_id as char,
            orientation,
            length,
            fixed_axis,
        });
        positions.push(pos);
        next_id += 1;
    }

    if vehicles.len() < num_cars_range.0 as usize {
        return None;
    }

    let config = Arc::new(BoardConfig {
        grid_size,
        vehicles,
        exit_row,
    });

    Some(RushHourState { config, positions })
}

/// Generate a solvable Rush Hour puzzle.
///
/// Generates multiple candidates and picks the one with the longest
/// optimal solution, creating harder puzzles that require more moves.
pub fn generate(
    rng: &mut ChaChaRng,
    grid_size: u8,
    num_cars_range: (u8, u8),
    solve_budget: usize,
) -> Option<RushHourState> {
    let keep = 10;
    let mut best: Option<(RushHourState, usize)> = None;
    let mut found = 0usize;
    let gen_budget = solve_budget.min(20_000);

    for _ in 0..1_000 {
        let candidate = match generate_candidate(rng, grid_size, num_cars_range) {
            Some(c) => c,
            None => continue,
        };

        if let Some(result) =
            super::solver::iddfs_solve(&candidate.config, &candidate.positions, gen_budget)
        {
            let cost = result.cost();
            if cost < 4 {
                continue;
            }
            if best
                .as_ref()
                .map_or(true, |(_, best_cost)| cost > *best_cost)
            {
                best = Some((candidate, cost));
            }
            found += 1;
            if found >= keep {
                break;
            }
        }
    }
    best.map(|(state, _)| state)
}
