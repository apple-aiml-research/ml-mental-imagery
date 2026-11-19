//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
use std::collections::{HashSet, VecDeque};
use std::sync::Arc;

use rand::Rng;
use rand_chacha::ChaChaRng;

use crate::base::game::Game;

use super::state::{
    canonical_form, shortest_delta, MentalRotationPuzzle, MentalRotationState, Polycube,
};

/// Generate a random connected polycube via growth algorithm.
pub fn generate_polycube(rng: &mut ChaChaRng, num_cubes: usize) -> Polycube {
    let mut occupied: HashSet<(i32, i32, i32)> = HashSet::new();
    let mut voxels: Vec<(i32, i32, i32)> = Vec::with_capacity(num_cubes);

    occupied.insert((0, 0, 0));
    voxels.push((0, 0, 0));

    let neighbors = [
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    ];

    while voxels.len() < num_cubes {
        // Pick a random existing voxel
        let idx = rng.gen_range(0..voxels.len());
        let (x, y, z) = voxels[idx];

        // Collect unoccupied neighbors
        let free: Vec<(i32, i32, i32)> = neighbors
            .iter()
            .map(|&(dx, dy, dz)| (x + dx, y + dy, z + dz))
            .filter(|p| !occupied.contains(p))
            .collect();

        if free.is_empty() {
            continue;
        }

        let ni = rng.gen_range(0..free.len());
        let new_pos = free[ni];
        occupied.insert(new_pos);
        voxels.push(new_pos);
    }

    Polycube { voxels }
}

/// Generate a solvable mental rotation puzzle.
pub fn generate(
    rng: &mut ChaChaRng,
    num_cubes: usize,
    increment: u16,
    max_delta: u16,
    max_step: i32,
    min_solve_steps: usize,
    solve_budget: usize,
) -> Option<MentalRotationState> {
    let half_steps = (max_delta / increment) as i32;

    for _ in 0..1000 {
        let polycube = generate_polycube(rng, num_cubes);

        // Pick random target rotation via bounded signed deltas
        let dx = rng.gen_range(-half_steps..=half_steps) * increment as i32;
        let dy = rng.gen_range(-half_steps..=half_steps) * increment as i32;
        let dz = rng.gen_range(-half_steps..=half_steps) * increment as i32;

        let target_rx = dx.rem_euclid(360) as u16;
        let target_ry = dy.rem_euclid(360) as u16;
        let target_rz = dz.rem_euclid(360) as u16;

        if target_rx == 0 && target_ry == 0 && target_rz == 0 {
            continue;
        }

        // Reject if the target canonical form matches identity (object looks the same)
        let identity_canonical = canonical_form(&polycube, 0, 0, 0);
        let target_canonical = canonical_form(&polycube, target_rx, target_ry, target_rz);

        if identity_canonical == target_canonical {
            continue;
        }

        // Reject puzzles requiring fewer than min_solve_steps
        let steps_needed: usize = [target_rx, target_ry, target_rz]
            .iter()
            .map(|&t| {
                let d = shortest_delta(0, t).abs();
                if d == 0 {
                    0
                } else {
                    ((d + max_step - 1) / max_step) as usize
                }
            })
            .sum();
        if steps_needed < min_solve_steps {
            continue;
        }

        let puzzle = Arc::new(MentalRotationPuzzle {
            polycube,
            target_rx,
            target_ry,
            target_rz,
            increment,
            target_canonical,
            start: (0, 0, 0),
            solvable: true,
            target_polycube: None,
            shadow_target: None,
        });

        let state = MentalRotationState {
            puzzle: puzzle.clone(),
            rx: 0,
            ry: 0,
            rz: 0,
        };

        // Verify BFS-solvable
        let game = super::MentalRotation {
            num_cubes_range: (num_cubes, num_cubes),
            increment,
            max_step: 180,
            max_delta,
            solve_budget,
            zoom_pct: 100,
            certificate_game: false,
            min_solve_steps: 1,
        };
        if game.solve(&state, solve_budget).is_some() {
            return Some(state);
        }
    }
    None
}

/// Check if a polycube is connected via BFS.
fn is_connected(voxels: &[(i32, i32, i32)]) -> bool {
    if voxels.is_empty() {
        return true;
    }
    let set: HashSet<(i32, i32, i32)> = voxels.iter().copied().collect();
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    queue.push_back(voxels[0]);
    visited.insert(voxels[0]);
    let neighbors = [
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    ];
    while let Some((x, y, z)) = queue.pop_front() {
        for &(dx, dy, dz) in &neighbors {
            let n = (x + dx, y + dy, z + dz);
            if set.contains(&n) && visited.insert(n) {
                queue.push_back(n);
            }
        }
    }
    visited.len() == voxels.len()
}

/// Mirror a polycube along a random axis to produce a chiral counterpart.
/// Returns None if the polycube is achiral along all tried axes.
fn mirror_polycube(rng: &mut ChaChaRng, polycube: &Polycube) -> Option<Polycube> {
    let orig_canon = canonical_form(polycube, 0, 0, 0);
    for _ in 0..10 {
        let axis = rng.gen_range(0..3u8);
        let mirrored: Vec<(i32, i32, i32)> = polycube
            .voxels
            .iter()
            .map(|&(x, y, z)| match axis {
                0 => (-x, y, z),
                1 => (x, -y, z),
                _ => (x, y, -z),
            })
            .collect();
        let candidate = Polycube { voxels: mirrored };
        if canonical_form(&candidate, 0, 0, 0) != orig_canon {
            return Some(candidate);
        }
    }
    None
}

/// Perturb a polycube by adding or removing a voxel to produce a different shape.
/// Returns None if perturbation fails (e.g. all strategies produce the same canonical form).
fn perturb_polycube(rng: &mut ChaChaRng, polycube: &Polycube) -> Option<Polycube> {
    let orig_canon = canonical_form(polycube, 0, 0, 0);

    for _ in 0..50 {
        let add = rng.gen_bool(0.5);
        if add {
            let occupied: HashSet<(i32, i32, i32)> = polycube.voxels.iter().copied().collect();
            let neighbors = [
                (1, 0, 0),
                (-1, 0, 0),
                (0, 1, 0),
                (0, -1, 0),
                (0, 0, 1),
                (0, 0, -1),
            ];
            let mut candidates: Vec<(i32, i32, i32)> = Vec::new();
            for &(x, y, z) in &polycube.voxels {
                for &(dx, dy, dz) in &neighbors {
                    let n = (x + dx, y + dy, z + dz);
                    if !occupied.contains(&n) {
                        candidates.push(n);
                    }
                }
            }
            candidates.sort();
            candidates.dedup();
            if !candidates.is_empty() {
                let new_voxel = candidates[rng.gen_range(0..candidates.len())];
                let mut voxels = polycube.voxels.clone();
                voxels.push(new_voxel);
                let candidate = Polycube { voxels };
                if canonical_form(&candidate, 0, 0, 0) != orig_canon {
                    return Some(candidate);
                }
            }
        } else {
            if polycube.voxels.len() <= 2 {
                continue;
            }
            let idx = rng.gen_range(0..polycube.voxels.len());
            let mut voxels = polycube.voxels.clone();
            voxels.swap_remove(idx);
            if is_connected(&voxels) {
                let candidate = Polycube { voxels };
                if canonical_form(&candidate, 0, 0, 0) != orig_canon {
                    return Some(candidate);
                }
            }
        }
    }
    None
}

/// Generate an unsolvable mental rotation puzzle.
///
/// Creates a solvable puzzle, then either mirror-flips the polycube (50%) or
/// perturbs it by adding/removing voxels (50%). Mirror flips produce a chiral
/// mismatch that cannot be resolved through rotations alone. The shadow solver
/// targets the original rotation to guide the policy near-completion before
/// recommending "no".
pub fn generate_unsolvable(
    rng: &mut ChaChaRng,
    num_cubes: usize,
    increment: u16,
    max_delta: u16,
    max_step: i32,
    min_solve_steps: usize,
    solve_budget: usize,
) -> Option<MentalRotationState> {
    for _ in 0..100 {
        let base_state = generate(
            rng,
            num_cubes,
            increment,
            max_delta,
            max_step,
            min_solve_steps,
            solve_budget,
        )?;
        let base_puzzle = &base_state.puzzle;

        // 50% mirror flip (chiral mismatch), 50% shape perturbation.
        // Polycubes are asymmetric by construction, so mirrors are always chiral.
        let use_mirror = rng.gen_bool(0.5);
        let altered = if use_mirror {
            mirror_polycube(rng, &base_puzzle.polycube)
                .or_else(|| perturb_polycube(rng, &base_puzzle.polycube))
        } else {
            perturb_polycube(rng, &base_puzzle.polycube)
                .or_else(|| mirror_polycube(rng, &base_puzzle.polycube))
        };

        if let Some(altered) = altered {
            // The target_canonical is computed from the *perturbed* polycube at the target rotation.
            let target_canonical = canonical_form(
                &altered,
                base_puzzle.target_rx,
                base_puzzle.target_ry,
                base_puzzle.target_rz,
            );

            // Verify no rotation of the original polycube matches this canonical form.
            // This ensures is_solved() is always false for the original shape.
            let steps = 360 / increment as u32;
            let mut collision = false;
            'outer: for ix in 0..steps {
                for iy in 0..steps {
                    for iz in 0..steps {
                        let rx = (ix * increment as u32) as u16;
                        let ry = (iy * increment as u32) as u16;
                        let rz = (iz * increment as u32) as u16;
                        if canonical_form(&base_puzzle.polycube, rx, ry, rz) == target_canonical {
                            collision = true;
                            break 'outer;
                        }
                    }
                }
            }
            if collision {
                continue;
            }

            let puzzle = Arc::new(MentalRotationPuzzle {
                polycube: base_puzzle.polycube.clone(),
                target_rx: base_puzzle.target_rx,
                target_ry: base_puzzle.target_ry,
                target_rz: base_puzzle.target_rz,
                increment,
                target_canonical,
                start: (0, 0, 0),
                solvable: false,
                target_polycube: Some(altered),
                shadow_target: Some((
                    base_puzzle.target_rx,
                    base_puzzle.target_ry,
                    base_puzzle.target_rz,
                )),
            });

            return Some(MentalRotationState {
                puzzle,
                rx: 0,
                ry: 0,
                rz: 0,
            });
        }
    }
    None
}

/// Generate a certificate-mode puzzle: 50/50 solvable vs unsolvable.
pub fn generate_certificate(
    rng: &mut ChaChaRng,
    num_cubes: usize,
    increment: u16,
    max_delta: u16,
    max_step: i32,
    min_solve_steps: usize,
    solve_budget: usize,
) -> Option<MentalRotationState> {
    // Generate both variants first, then flip — guarantees exact 50/50.
    let solvable = generate(
        rng,
        num_cubes,
        increment,
        max_delta,
        max_step,
        min_solve_steps,
        solve_budget,
    )?;
    let unsolvable = generate_unsolvable(
        rng,
        num_cubes,
        increment,
        max_delta,
        max_step,
        min_solve_steps,
        solve_budget,
    )?;
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
    fn polycube_connected_and_correct_size() {
        for seed in 0..10u64 {
            let mut rng = ChaChaRng::seed_from_u64(seed);
            let poly = generate_polycube(&mut rng, 9);
            assert_eq!(poly.voxels.len(), 9);

            // Verify all unique
            let set: HashSet<_> = poly.voxels.iter().copied().collect();
            assert_eq!(set.len(), 9);

            // Verify connected: BFS from first voxel should reach all
            let mut visited = HashSet::new();
            let mut queue = vec![poly.voxels[0]];
            visited.insert(poly.voxels[0]);
            let neighbors = [
                (1, 0, 0),
                (-1, 0, 0),
                (0, 1, 0),
                (0, -1, 0),
                (0, 0, 1),
                (0, 0, -1),
            ];
            while let Some((x, y, z)) = queue.pop() {
                for &(dx, dy, dz) in &neighbors {
                    let n = (x + dx, y + dy, z + dz);
                    if set.contains(&n) && visited.insert(n) {
                        queue.push(n);
                    }
                }
            }
            assert_eq!(visited.len(), 9, "Polycube not connected for seed {}", seed);
        }
    }

    #[test]
    fn generate_produces_solvable() {
        for seed in 0..10u64 {
            let mut rng = ChaChaRng::seed_from_u64(seed);
            let state =
                generate(&mut rng, 9, 90, 180, 45, 5, 100_000).expect("test seed should generate");
            // Verify it is solvable
            let game = super::super::MentalRotation {
                num_cubes_range: (9, 9),
                increment: 90,
                max_step: 45,
                max_delta: 180,
                solve_budget: 100_000,
                zoom_pct: 100,
                certificate_game: false,
                min_solve_steps: 5,
            };
            let result = game.solve(&state, 100_000);
            assert!(result.is_some(), "Seed {} produced unsolvable puzzle", seed);
        }
    }

    #[test]
    fn generate_unsolvable_produces_unsolvable() {
        for seed in 0..10u64 {
            let mut rng = ChaChaRng::seed_from_u64(seed);
            let state = generate_unsolvable(&mut rng, 9, 90, 180, 45, 5, 100_000)
                .expect("test seed should generate unsolvable");
            assert!(!state.puzzle.solvable);
            assert!(state.puzzle.target_polycube.is_some());
            assert!(state.puzzle.shadow_target.is_some());

            // is_solved should never be true since canonical forms differ
            let game = super::super::MentalRotation {
                num_cubes_range: (9, 9),
                increment: 90,
                max_step: 45,
                max_delta: 180,
                solve_budget: 100_000,
                zoom_pct: 100,
                certificate_game: false,
                min_solve_steps: 5,
            };
            // Check all 24 orientations at 90° increments
            for rx in (0..360).step_by(90) {
                for ry in (0..360).step_by(90) {
                    for rz in (0..360).step_by(90) {
                        let test_state = MentalRotationState {
                            puzzle: state.puzzle.clone(),
                            rx: rx as u16,
                            ry: ry as u16,
                            rz: rz as u16,
                        };
                        assert!(
                            !game.is_solved(&test_state),
                            "Seed {}: unsolvable puzzle should never be solved at ({},{},{})",
                            seed,
                            rx,
                            ry,
                            rz
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn generate_certificate_distribution() {
        let mut solvable = 0;
        let mut unsolvable = 0;
        for seed in 0..100u64 {
            let mut rng = ChaChaRng::seed_from_u64(seed);
            if let Some(state) = generate_certificate(&mut rng, 9, 90, 180, 45, 5, 100_000) {
                if state.puzzle.solvable {
                    solvable += 1;
                } else {
                    unsolvable += 1;
                }
            }
        }
        // Expect roughly 50/50 (with tolerance for randomness)
        assert!(solvable > 20, "Too few solvable: {}", solvable);
        assert!(unsolvable > 20, "Too few unsolvable: {}", unsolvable);
    }
}
