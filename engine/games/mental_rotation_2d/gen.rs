//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
use std::collections::HashSet;
use std::sync::Arc;

use rand::Rng;
use rand_chacha::ChaChaRng;

use crate::base::game::Game;

use super::state::{canonical_form_2d, MentalRotation2dPuzzle, MentalRotation2dState, Polyomino};

/// Generate a random connected polyomino via growth algorithm (4-connected).
pub fn generate_polyomino(rng: &mut ChaChaRng, num_cells: usize) -> Polyomino {
    let mut occupied: HashSet<(i32, i32)> = HashSet::new();
    let mut cells: Vec<(i32, i32)> = Vec::with_capacity(num_cells);

    occupied.insert((0, 0));
    cells.push((0, 0));

    let neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)];

    while cells.len() < num_cells {
        let idx = rng.gen_range(0..cells.len());
        let (x, y) = cells[idx];

        let free: Vec<(i32, i32)> = neighbors
            .iter()
            .map(|&(dx, dy)| (x + dx, y + dy))
            .filter(|p| !occupied.contains(p))
            .collect();

        if free.is_empty() {
            continue;
        }

        let ni = rng.gen_range(0..free.len());
        let new_pos = free[ni];
        occupied.insert(new_pos);
        cells.push(new_pos);
    }

    Polyomino { cells }
}

/// Check that a polyomino has no rotational symmetry and is chiral (no
/// rotation-only orientation produces the same canonical form as any
/// rotation+flip orientation).
fn is_asymmetric(polyomino: &Polyomino, increment: u16) -> bool {
    let steps = 360 / increment as u32;
    let identity = canonical_form_2d(polyomino, 0, false);

    // No rotational symmetry: every non-identity rotation must differ
    for i in 1..steps {
        let rot = (i * increment as u32) as u16;
        if canonical_form_2d(polyomino, rot, false) == identity {
            return false;
        }
    }

    // Chirality: no flipped orientation matches any unflipped orientation
    let unflipped: Vec<Vec<(i32, i32)>> = (0..steps)
        .map(|i| canonical_form_2d(polyomino, (i * increment as u32) as u16, false))
        .collect();
    for i in 0..steps {
        let rot = (i * increment as u32) as u16;
        let flipped = canonical_form_2d(polyomino, rot, true);
        if unflipped.contains(&flipped) {
            return false;
        }
    }

    true
}

/// Generate a solvable 2D mental rotation puzzle (rotation only, no flip).
pub fn generate(
    rng: &mut ChaChaRng,
    num_cells: usize,
    increment: u16,
    max_delta: u16,
    max_step: i32,
    min_steps: usize,
    solve_budget: usize,
) -> Option<MentalRotation2dState> {
    let half_steps = (max_delta / increment) as i32;
    let min_distance = if min_steps > 1 {
        (min_steps as i32 - 1) * max_step + 1
    } else {
        1
    };

    for _ in 0..1000 {
        let polyomino = generate_polyomino(rng, num_cells);

        if !is_asymmetric(&polyomino, increment) {
            continue;
        }

        // Pick random target rotation (no flip)
        let d_rot = rng.gen_range(-half_steps..=half_steps) * increment as i32;
        let target_rotation = d_rot.rem_euclid(360) as u16;

        if target_rotation == 0 {
            continue;
        }

        let distance = super::state::shortest_delta(0, target_rotation).abs();
        if distance < min_distance {
            continue;
        }

        let target_canonical = canonical_form_2d(&polyomino, target_rotation, false);

        let puzzle = Arc::new(MentalRotation2dPuzzle {
            polyomino,
            target_rotation,
            target_flipped: false,
            increment,
            target_canonical,
            start: 0,
            solvable: true,
            shadow_target: None,
            color_seed: rng.gen(),
        });

        let state = MentalRotation2dState {
            puzzle: puzzle.clone(),
            rotation: 0,
        };

        // Verify solvable
        let game = super::MentalRotation2d {
            num_cells,
            increment,
            max_step: 180,
            max_delta,
            solve_budget,
            certificate_game: false,
            min_steps: 1,
        };
        if game.solve(&state, solve_budget).is_some() {
            return Some(state);
        }
    }
    None
}

/// Generate an unsolvable 2D mental rotation puzzle.
///
/// Uses the same polyomino but with a flipped target orientation.
/// Since the player can only rotate (not flip), matching is impossible
/// for chiral shapes.
pub fn generate_unsolvable(
    rng: &mut ChaChaRng,
    num_cells: usize,
    increment: u16,
    max_delta: u16,
    max_step: i32,
    min_steps: usize,
) -> Option<MentalRotation2dState> {
    let half_steps = (max_delta / increment) as i32;
    // Shadow solver reports done when within max_step, so an extra step's
    // worth of distance guarantees min_steps rotation actions.
    let min_distance = if min_steps > 1 {
        min_steps as i32 * max_step + 1
    } else {
        1
    };

    for _ in 0..1000 {
        let polyomino = generate_polyomino(rng, num_cells);

        if !is_asymmetric(&polyomino, increment) {
            continue;
        }

        // Pick random target rotation, but with flip=true
        let d_rot = rng.gen_range(-half_steps..=half_steps) * increment as i32;
        let target_rotation = d_rot.rem_euclid(360) as u16;

        let distance = super::state::shortest_delta(0, target_rotation).abs();
        if distance < min_distance {
            continue;
        }

        let target_canonical = canonical_form_2d(&polyomino, target_rotation, true);

        let puzzle = Arc::new(MentalRotation2dPuzzle {
            polyomino,
            target_rotation,
            target_flipped: true,
            increment,
            target_canonical,
            start: 0,
            solvable: false,
            shadow_target: Some(target_rotation),
            color_seed: rng.gen(),
        });

        return Some(MentalRotation2dState {
            puzzle,
            rotation: 0,
        });
    }
    None
}

/// Generate a certificate-mode puzzle: 50/50 solvable vs unsolvable.
pub fn generate_certificate(
    rng: &mut ChaChaRng,
    num_cells: usize,
    increment: u16,
    max_delta: u16,
    max_step: i32,
    min_steps: usize,
    solve_budget: usize,
) -> Option<MentalRotation2dState> {
    let solvable = generate(
        rng,
        num_cells,
        increment,
        max_delta,
        max_step,
        min_steps,
        solve_budget,
    )?;
    let unsolvable =
        generate_unsolvable(rng, num_cells, increment, max_delta, max_step, min_steps)?;
    if rng.gen_bool(0.5) {
        Some(solvable)
    } else {
        Some(unsolvable)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn polyomino_connected_and_correct_size() {
        let neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)];
        for seed in 0..10u64 {
            let mut rng = ChaChaRng::seed_from_u64(seed);
            let poly = generate_polyomino(&mut rng, 5);
            assert_eq!(poly.cells.len(), 5);

            // Verify all unique
            let set: HashSet<_> = poly.cells.iter().copied().collect();
            assert_eq!(set.len(), 5);

            // Verify connected via BFS
            let mut visited = HashSet::new();
            let mut queue = std::collections::VecDeque::new();
            queue.push_back(poly.cells[0]);
            visited.insert(poly.cells[0]);
            while let Some((x, y)) = queue.pop_front() {
                for &(dx, dy) in &neighbors {
                    let n = (x + dx, y + dy);
                    if set.contains(&n) && visited.insert(n) {
                        queue.push_back(n);
                    }
                }
            }
            assert_eq!(
                visited.len(),
                5,
                "Polyomino not connected for seed {}",
                seed
            );
        }
    }

    #[test]
    fn generated_polyominoes_are_asymmetric() {
        for seed in 0..20u64 {
            let mut rng = ChaChaRng::seed_from_u64(seed);
            let state =
                generate(&mut rng, 5, 90, 180, 180, 1, 100_000).expect("test seed should generate");
            assert!(
                is_asymmetric(&state.puzzle.polyomino, 90),
                "Seed {}: generated polyomino should be asymmetric",
                seed
            );
        }
        for seed in 0..20u64 {
            let mut rng = ChaChaRng::seed_from_u64(seed);
            let state = generate_unsolvable(&mut rng, 5, 90, 180, 180, 1)
                .expect("seed should generate unsolvable");
            assert!(
                is_asymmetric(&state.puzzle.polyomino, 90),
                "Seed {}: unsolvable polyomino should be asymmetric",
                seed
            );
        }
    }

    #[test]
    fn generate_produces_solvable() {
        for seed in 0..10u64 {
            let mut rng = ChaChaRng::seed_from_u64(seed);
            let state =
                generate(&mut rng, 5, 90, 180, 180, 1, 100_000).expect("test seed should generate");
            assert!(state.puzzle.solvable);
            assert!(!state.puzzle.target_flipped);

            let game = super::super::MentalRotation2d {
                num_cells: 5,
                increment: 90,
                max_step: 180,
                max_delta: 180,
                solve_budget: 100_000,
                certificate_game: false,
                min_steps: 1,
            };
            let result = game.solve(&state, 100_000);
            assert!(result.is_some(), "Seed {} produced unsolvable puzzle", seed);
        }
    }

    #[test]
    fn generate_unsolvable_produces_unsolvable() {
        for seed in 0..10u64 {
            let mut rng = ChaChaRng::seed_from_u64(seed);
            let state = generate_unsolvable(&mut rng, 5, 90, 180, 180, 1)
                .expect("seed should generate unsolvable");
            assert!(!state.puzzle.solvable);
            assert!(state.puzzle.target_flipped);
            assert!(state.puzzle.shadow_target.is_some());

            let game = super::super::MentalRotation2d {
                num_cells: 5,
                increment: 90,
                max_step: 180,
                max_delta: 180,
                solve_budget: 100_000,
                certificate_game: false,
                min_steps: 1,
            };
            // No rotation-only orientation should solve it
            for rot in (0..360).step_by(90) {
                let test_state = MentalRotation2dState {
                    puzzle: state.puzzle.clone(),
                    rotation: rot as u16,
                };
                assert!(
                    !game.is_solved(&test_state),
                    "Seed {}: unsolvable puzzle should never be solved at rot={}",
                    seed,
                    rot
                );
            }
        }
    }

    #[test]
    fn generate_certificate_distribution() {
        let mut solvable = 0;
        let mut unsolvable = 0;
        for seed in 0..100u64 {
            let mut rng = ChaChaRng::seed_from_u64(seed);
            if let Some(state) = generate_certificate(&mut rng, 5, 90, 180, 180, 1, 100_000) {
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
}
