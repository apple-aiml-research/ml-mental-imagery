//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
pub mod gen;
pub mod object_boxes;
pub mod render;
pub mod state;

use rand_chacha::ChaChaRng;

use crate::base::game::{Game, RgbBuffer};
use crate::base::state_text::StateField;
use crate::base::template::{ActionTemplate, ParamType};

use state::{canonical_form, shortest_delta, transform_error, MentalRotationState, RotationAction};

/// Mental rotation game configuration.
pub struct MentalRotation {
    pub num_cubes_range: (usize, usize),
    pub increment: u16,
    pub max_step: i32,
    pub max_delta: u16,
    pub solve_budget: usize,
    pub zoom_pct: u32,
    pub certificate_game: bool,
    pub min_solve_steps: usize,
}

/// Largest rotation per action, in degrees.
const MAX_STEP: i32 = 30;
/// Largest initial offset per axis, in degrees.
const MAX_DELTA: u16 = 180;
const SOLVE_BUDGET: usize = 100_000;

pub fn build_factory(
    name: &str,
    params: &[super::registry::ConfigParam],
    overrides: &std::collections::HashMap<String, super::registry::ConfigValue>,
) -> Result<Box<dyn crate::stream::loader::GameFactory>, String> {
    let (nc_lo, nc_hi) = super::registry::resolve_int_range(params, overrides, "num_cubes_range")?;
    let increment = super::registry::resolve_int(params, overrides, "increment")? as u16;
    if !matches!(increment, 15 | 30 | 45 | 90) {
        return Err(format!(
            "increment must be 15, 30, 45, or 90, got {}",
            increment
        ));
    }
    let zoom_pct = super::registry::resolve_int(params, overrides, "zoom_pct")? as u32;
    if zoom_pct == 0 {
        return Err("zoom_pct must be > 0".to_string());
    }
    let certificate_game =
        super::registry::resolve_int(params, overrides, "certificate_game")? != 0;
    let min_solve_steps =
        super::registry::resolve_int(params, overrides, "min_solve_steps")? as usize;
    let game = MentalRotation {
        num_cubes_range: (nc_lo as usize, nc_hi as usize),
        increment,
        max_step: MAX_STEP,
        max_delta: MAX_DELTA,
        solve_budget: SOLVE_BUDGET,
        zoom_pct,
        certificate_game,
        min_solve_steps,
    };
    Ok(crate::stream::loader::game_factory(
        game,
        name,
        SOLVE_BUDGET,
    ))
}

inventory::submit! {
    super::registry::GameEntry {
        name: "mental_rotation",
        params: &[
            super::registry::ConfigParam {
                key: "num_cubes_range",
                param_type: super::registry::ConfigParamType::IntRange,
                default: super::registry::ConfigValue::IntRange(5, 5),
            },
            super::registry::ConfigParam {
                key: "increment",
                param_type: super::registry::ConfigParamType::Int,
                default: super::registry::ConfigValue::Int(15),
            },
            super::registry::ConfigParam {
                key: "zoom_pct",
                param_type: super::registry::ConfigParamType::Int,
                default: super::registry::ConfigValue::Int(100),
            },
            super::registry::ConfigParam {
                key: "certificate_game",
                param_type: super::registry::ConfigParamType::Int,
                default: super::registry::ConfigValue::Int(0),
            },
            super::registry::ConfigParam {
                key: "min_solve_steps",
                param_type: super::registry::ConfigParamType::Int,
                default: super::registry::ConfigValue::Int(1),
            },
        ],
        build: build_factory,
    }
}

/// Analytical solver: compute optimal action and total cost.
/// Returns the action that closes the largest angular delta (clamped to
/// `max_step`), and the total angular distance to the goal (in degrees).
fn analytical_solve(
    state: &MentalRotationState,
    game: &MentalRotation,
) -> Option<(RotationAction, usize)> {
    analytical_solve_target(
        state,
        game,
        state.puzzle.target_rx,
        state.puzzle.target_ry,
        state.puzzle.target_rz,
    )
}

/// Shadow analytical solver: targets the original rotation for unsolvable puzzles.
fn shadow_analytical_solve(
    state: &MentalRotationState,
    game: &MentalRotation,
) -> Option<(RotationAction, usize)> {
    if let Some((trx, try_, trz)) = state.puzzle.shadow_target {
        analytical_solve_target(state, game, trx, try_, trz)
    } else {
        None
    }
}

/// Core analytical solver targeting a specific rotation.
fn analytical_solve_target(
    state: &MentalRotationState,
    game: &MentalRotation,
    target_rx: u16,
    target_ry: u16,
    target_rz: u16,
) -> Option<(RotationAction, usize)> {
    let dx = shortest_delta(state.rx, target_rx);
    let dy = shortest_delta(state.ry, target_ry);
    let dz = shortest_delta(state.rz, target_rz);
    let cost = (dx.abs() + dy.abs() + dz.abs()) as usize;
    if cost == 0 {
        return None;
    }

    // Pick the axis with the largest absolute delta
    let ax = dx.abs();
    let ay = dy.abs();
    let az = dz.abs();

    let action = if ax >= ay && ax >= az && dx != 0 {
        RotationAction::RotateX(dx.signum() * dx.abs().min(game.max_step))
    } else if ay >= az && dy != 0 {
        RotationAction::RotateY(dy.signum() * dy.abs().min(game.max_step))
    } else {
        RotationAction::RotateZ(dz.signum() * dz.abs().min(game.max_step))
    };
    Some((action, cost))
}

impl Game for MentalRotation {
    type State = MentalRotationState;
    type Action = RotationAction;

    fn generate(&self, rng: &mut ChaChaRng) -> Option<MentalRotationState> {
        use rand::Rng;
        let num_cubes = rng.gen_range(self.num_cubes_range.0..=self.num_cubes_range.1);
        if self.certificate_game {
            gen::generate_certificate(
                rng,
                num_cubes,
                self.increment,
                self.max_delta,
                self.max_step,
                self.min_solve_steps,
                self.solve_budget,
            )
        } else {
            gen::generate(
                rng,
                num_cubes,
                self.increment,
                self.max_delta,
                self.max_step,
                self.min_solve_steps,
                self.solve_budget,
            )
        }
    }

    fn puzzle_fingerprint(&self, state: &MentalRotationState) -> u64 {
        use std::hash::Hasher;

        // Translate voxels to the bbox-min origin, then sort: an order- and
        // translation-independent encoding that preserves orientation (raw
        // identity — no rotation/reflection canonicalization).
        fn normalize(voxels: &[(i32, i32, i32)]) -> Vec<(i32, i32, i32)> {
            let min_x = voxels.iter().map(|v| v.0).min().unwrap_or(0);
            let min_y = voxels.iter().map(|v| v.1).min().unwrap_or(0);
            let min_z = voxels.iter().map(|v| v.2).min().unwrap_or(0);
            let mut out: Vec<(i32, i32, i32)> = voxels
                .iter()
                .map(|&(x, y, z)| (x - min_x, y - min_y, z - min_z))
                .collect();
            out.sort();
            out
        }

        fn write_voxels(h: &mut crate::base::fingerprint::FnvHasher, voxels: &[(i32, i32, i32)]) {
            use std::hash::Hasher;
            let normalized = normalize(voxels);
            h.write_u64(normalized.len() as u64);
            for (x, y, z) in normalized {
                h.write_u32(x as u32);
                h.write_u32(y as u32);
                h.write_u32(z as u32);
            }
        }

        // Deref the Arc; hash the immutable puzzle definition, never the pointer.
        let puzzle = &*state.puzzle;
        crate::base::fingerprint::fingerprint(|h| {
            // 1. solvable flag (distinguishes solvable vs certificate-unsolvable)
            h.write_u8(if puzzle.solvable { 1 } else { 0 });
            // 2. rotation increment
            h.write_u16(puzzle.increment);
            // 3. left polycube geometry (normalized + sorted)
            write_voxels(h, &puzzle.polycube.voxels);
            // 4. target orientation (exact integer Euler angles)
            h.write_u16(puzzle.target_rx);
            h.write_u16(puzzle.target_ry);
            h.write_u16(puzzle.target_rz);
            // 5. optional distinct right-panel shape (unsolvable puzzles)
            match &puzzle.target_polycube {
                None => h.write_u8(0),
                Some(target_poly) => {
                    h.write_u8(1);
                    write_voxels(h, &target_poly.voxels);
                }
            }
            // Excluded: rx/ry/rz and
            // start (always identity), target_canonical (float-derived), and
            // shadow_target (solver-only, determined by solvable + target).
        })
    }

    fn trivial(&self) -> MentalRotationState {
        // Bar along X, target is 90 deg on Y axis — different canonical form, solvable in 1 move
        let polycube = state::Polycube {
            voxels: vec![(0, 0, 0), (1, 0, 0)],
        };
        let target_ry = 90;
        let target_canonical = canonical_form(&polycube, 0, target_ry, 0);
        let puzzle = std::sync::Arc::new(state::MentalRotationPuzzle {
            polycube,
            target_rx: 0,
            target_ry,
            target_rz: 0,
            increment: self.increment,
            target_canonical,
            start: (0, 0, 0),
            solvable: true,
            target_polycube: None,
            shadow_target: None,
        });
        MentalRotationState {
            puzzle,
            rx: 0,
            ry: 0,
            rz: 0,
        }
    }

    fn actions(&self, _state: &MentalRotationState) -> Vec<RotationAction> {
        let mut actions = Vec::with_capacity(360 * 3);
        for d in -180..=180i32 {
            if d == 0 {
                continue;
            }
            actions.push(RotationAction::RotateX(d));
            actions.push(RotationAction::RotateY(d));
            actions.push(RotationAction::RotateZ(d));
        }
        actions
    }

    fn apply(
        &self,
        state: &MentalRotationState,
        action: &RotationAction,
    ) -> Option<MentalRotationState> {
        Some(state.apply_action(*action))
    }

    fn is_solved(&self, state: &MentalRotationState) -> bool {
        if !state.puzzle.solvable {
            return false;
        }
        transform_error(
            &state.puzzle.polycube,
            state.rx,
            state.ry,
            state.rz,
            state.puzzle.target_rx,
            state.puzzle.target_ry,
            state.puzzle.target_rz,
        ) <= 0.15
    }

    fn render(&self, state: &MentalRotationState, resolution: u32) -> RgbBuffer {
        render::render(state, resolution, self.zoom_pct)
    }

    fn action_to_str(&self, action: &RotationAction) -> String {
        match action {
            RotationAction::RotateX(d) => format!("rotatex({})", d),
            RotationAction::RotateY(d) => format!("rotatey({})", d),
            RotationAction::RotateZ(d) => format!("rotatez({})", d),
        }
    }

    fn parse_action(&self, _state: &MentalRotationState, s: &str) -> Option<RotationAction> {
        fn parse_rotate(s: &str, prefix: &str) -> Option<i32> {
            let inner = s.strip_prefix(prefix)?.strip_suffix(')')?;
            let d: i32 = inner.parse().ok()?;
            if d == 0 || d < -180 || d > 180 {
                return None;
            }
            Some(d)
        }
        if let Some(d) = parse_rotate(s, "rotatex(") {
            Some(RotationAction::RotateX(d))
        } else if let Some(d) = parse_rotate(s, "rotatey(") {
            Some(RotationAction::RotateY(d))
        } else if let Some(d) = parse_rotate(s, "rotatez(") {
            Some(RotationAction::RotateZ(d))
        } else {
            None
        }
    }

    fn action_templates(&self, _state: &MentalRotationState) -> Vec<ActionTemplate> {
        ["rotatex(", "rotatey(", "rotatez("]
            .iter()
            .map(|prefix| ActionTemplate {
                segments: vec![
                    ParamType::Literal(prefix.to_string()),
                    ParamType::Int {
                        min: -(self.max_step),
                        max: self.max_step,
                    },
                    ParamType::Literal(")".to_string()),
                ],
            })
            .collect()
    }

    fn prompt(&self, _state: &MentalRotationState) -> String {
        if self.certificate_game {
            "Determine if the left and right shapes are the same. \
             Rotate to check, then say 'yes' if they match or 'no' if they don't."
                .to_string()
        } else {
            "Rotate the left shape to match the right shape.".to_string()
        }
    }

    fn action_description(&self, _state: &MentalRotationState) -> String {
        if self.certificate_game {
            "rotatex(deg), rotatey(deg), rotatez(deg), yes, no".to_string()
        } else {
            "rotatex(deg), rotatey(deg), rotatez(deg)".to_string()
        }
    }

    fn object_boxes(
        &self,
        state: &MentalRotationState,
    ) -> Vec<crate::base::object_boxes::ObjectBox> {
        object_boxes::object_boxes(state)
    }

    /// Current angles (`rx`, `ry`, `rz`), target angles (`tx`, `ty`, `tz`),
    /// signed shortest delta per axis (`dx`, `dy`, `dz`), rotation increment
    /// (`inc`), and polycube size (`n_voxels`). Unsolvable certificate puzzles
    /// use the solver's shadow target, so the text does not reveal the answer.
    fn state_fields(&self, state: &MentalRotationState) -> Vec<StateField> {
        let puzzle = &state.puzzle;
        let mut target = (puzzle.target_rx, puzzle.target_ry, puzzle.target_rz);
        if self.certificate_game && !puzzle.solvable {
            target = puzzle.shadow_target.unwrap_or(target);
        }
        let (target_x, target_y, target_z) = target;
        vec![
            ("rx", state.rx.to_string()),
            ("ry", state.ry.to_string()),
            ("rz", state.rz.to_string()),
            ("tx", target_x.to_string()),
            ("ty", target_y.to_string()),
            ("tz", target_z.to_string()),
            ("dx", shortest_delta(state.rx, target_x).to_string()),
            ("dy", shortest_delta(state.ry, target_y).to_string()),
            ("dz", shortest_delta(state.rz, target_z).to_string()),
            ("inc", puzzle.increment.to_string()),
            ("n_voxels", puzzle.polycube.voxels.len().to_string()),
        ]
    }

    fn solve(
        &self,
        state: &MentalRotationState,
        _budget: usize,
    ) -> Option<(RotationAction, usize)> {
        if self.certificate_game && !state.puzzle.solvable {
            shadow_analytical_solve(state, self)
        } else {
            analytical_solve(state, self).map(|(action, cost)| (action, cost))
        }
    }

    fn solve_path(
        &self,
        state: &MentalRotationState,
        _budget: usize,
    ) -> Option<Vec<(RotationAction, usize)>> {
        let use_shadow = self.certificate_game && !state.puzzle.solvable;
        let solver_fn = if use_shadow {
            shadow_analytical_solve
        } else {
            analytical_solve
        };
        let (target_rx, target_ry, target_rz) = if use_shadow {
            state.puzzle.shadow_target.unwrap_or((
                state.puzzle.target_rx,
                state.puzzle.target_ry,
                state.puzzle.target_rz,
            ))
        } else {
            (
                state.puzzle.target_rx,
                state.puzzle.target_ry,
                state.puzzle.target_rz,
            )
        };

        let mut actions = Vec::new();
        let mut states = Vec::new();
        let mut current = state.clone();
        let limit = 1080; // safety cap
        for _ in 0..limit {
            if solver_fn(&current, self).is_none() {
                break;
            }
            let (action, _) = solver_fn(&current, self)?;
            states.push(current.clone());
            current = current.apply_action(action);
            actions.push(action);
        }
        if solver_fn(&current, self).is_none() {
            // Cost-to-goal = angular distance from each state along the path
            Some(
                actions
                    .into_iter()
                    .zip(states)
                    .map(|(a, st)| {
                        let dx = shortest_delta(st.rx, target_rx);
                        let dy = shortest_delta(st.ry, target_ry);
                        let dz = shortest_delta(st.rz, target_rz);
                        (a, (dx.abs() + dy.abs() + dz.abs()) as usize)
                    })
                    .collect(),
            )
        } else {
            None
        }
    }

    fn certificate_mode(&self) -> bool {
        self.certificate_game
    }

    fn is_solvable(&self, state: &MentalRotationState) -> bool {
        state.puzzle.solvable
    }

    fn shadow_solve_cost(&self, state: &MentalRotationState) -> Option<usize> {
        if state.puzzle.solvable {
            return None;
        }
        if let Some((trx, try_, trz)) = state.puzzle.shadow_target {
            let dx = shortest_delta(state.rx, trx);
            let dy = shortest_delta(state.ry, try_);
            let dz = shortest_delta(state.rz, trz);
            Some((dx.abs() + dy.abs() + dz.abs()) as usize)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::episode::{Episode, GameEpisode};
    use rand::SeedableRng;
    use std::sync::Arc;

    fn make_game() -> MentalRotation {
        MentalRotation {
            num_cubes_range: (9, 9),
            increment: 90,
            max_step: 45,
            max_delta: 180,
            solve_budget: 100_000,
            zoom_pct: 100,
            certificate_game: false,
            min_solve_steps: 5,
        }
    }

    #[test]
    fn state_text_reports_angles_and_deltas() {
        let game = make_game();
        let mut rng = ChaChaRng::seed_from_u64(42);
        let mut state = game.generate(&mut rng).expect("test seed should generate");
        let mut puzzle = (*state.puzzle).clone();
        (puzzle.target_rx, puzzle.target_ry, puzzle.target_rz) = (90, 180, 0);
        state.puzzle = Arc::new(puzzle);
        (state.rx, state.ry, state.rz) = (0, 270, 0);
        let fields = game.state_fields(&state);
        let text = crate::base::state_text::format_state_json(&fields, 0);
        let expected_prefix =
            r#"{"rx":0,"ry":270,"rz":0,"tx":90,"ty":180,"tz":0,"dx":90,"dy":-90,"dz":0,"inc":"#;
        assert!(text.starts_with(expected_prefix), "{text}");
        assert!(text.ends_with(r#","n_voxels":9,"moves":0}"#), "{text}");
    }

    #[test]
    fn action_roundtrip() {
        let game = make_game();
        let mut rng = ChaChaRng::seed_from_u64(42);
        let state = game.generate(&mut rng).expect("test seed should generate");
        let sample_actions = [
            RotationAction::RotateX(90),
            RotationAction::RotateX(-45),
            RotationAction::RotateY(180),
            RotationAction::RotateZ(-1),
        ];
        for action in &sample_actions {
            let s = game.action_to_str(action);
            let parsed = game.parse_action(&state, &s).unwrap();
            assert_eq!(*action, parsed);
        }
    }

    #[test]
    fn is_solved_positive() {
        let game = make_game();
        let mut rng = ChaChaRng::seed_from_u64(42);
        let state = game.generate(&mut rng).expect("test seed should generate");
        // Construct a state at target rotation
        let target = MentalRotationState {
            puzzle: state.puzzle.clone(),
            rx: state.puzzle.target_rx,
            ry: state.puzzle.target_ry,
            rz: state.puzzle.target_rz,
        };
        assert!(game.is_solved(&target));
    }

    #[test]
    fn is_solved_negative() {
        let game = make_game();
        let mut rng = ChaChaRng::seed_from_u64(42);
        let state = game.generate(&mut rng).expect("test seed should generate");
        // Identity should not be solved (generate rejects this case)
        assert!(!game.is_solved(&state));
    }

    #[test]
    fn registered_in_registry() {
        assert!(crate::games::registry::find_game("mental_rotation").is_some());
    }

    #[test]
    fn test_aspect_ratio_default() {
        let game = make_game();
        assert_eq!(game.aspect_ratio(), (1, 1));
    }

    #[test]
    fn generate_and_solve() {
        let game = make_game();
        for seed in 0..10u64 {
            let mut rng = ChaChaRng::seed_from_u64(seed);
            let state = game.generate(&mut rng).expect("test seed should generate");
            let result = game.solve(&state, 100_000);
            assert!(result.is_some(), "Seed {} unsolvable", seed);
        }
    }

    #[test]
    fn episode_apply_undo_reset() {
        let game = Arc::new(make_game());
        let mut rng = ChaChaRng::seed_from_u64(42);
        let state = game.generate(&mut rng).expect("test seed should generate");
        let initial_rx = state.rx;
        let initial_ry = state.ry;
        let initial_rz = state.rz;

        let mut ep = GameEpisode::new(game, state, 100_000);

        assert!(ep.apply("rotatex(90)"));
        assert_ne!(ep.state.rx, initial_rx);

        ep.undo();
        assert_eq!(ep.state.rx, initial_rx);
        assert_eq!(ep.state.ry, initial_ry);
        assert_eq!(ep.state.rz, initial_rz);

        ep.apply("rotatey(45)");
        ep.apply("rotatez(-30)");
        ep.reset();
        assert_eq!(ep.state.rx, initial_rx);
        assert_eq!(ep.state.ry, initial_ry);
        assert_eq!(ep.state.rz, initial_rz);
    }

    #[test]
    fn episode_expert_action() {
        let game = Arc::new(make_game());
        let mut rng = ChaChaRng::seed_from_u64(42);
        let state = game.generate(&mut rng).expect("test seed should generate");

        let ep = GameEpisode::new(game, state, 100_000);
        let action = ep.expert_action();
        // Should be a rotatex/y/z action, or back/reset (not "done" since not solved at identity)
        assert!(
            action.starts_with("rotate") || action == "back" || action == "reset",
            "Unexpected expert action: {}",
            action
        );
    }

    #[test]
    fn episode_expert_done_when_solved() {
        let game = Arc::new(make_game());
        let mut rng = ChaChaRng::seed_from_u64(42);
        let state = game.generate(&mut rng).expect("test seed should generate");

        // Manually set state to target
        let solved_state = MentalRotationState {
            puzzle: state.puzzle.clone(),
            rx: state.puzzle.target_rx,
            ry: state.puzzle.target_ry,
            rz: state.puzzle.target_rz,
        };
        let ep = GameEpisode::new(game, solved_state, 100_000);
        assert_eq!(ep.expert_action(), "done");
    }

    #[test]
    fn trivial_is_solvable() {
        let game = make_game();
        let state = game.trivial();
        let result = game.solve(&state, 100_000);
        assert!(result.is_some(), "trivial state should be solvable");
    }

    #[test]
    fn symmetric_piece_exact_target_solves() {
        // A bar along X: rotating 90° around Y is the target.
        // With transform_error, only rotations visually close to the target
        // are accepted — 270° Y has different face colors and should NOT solve.
        let game = MentalRotation {
            num_cubes_range: (2, 2),
            increment: 90,
            max_step: 180,
            max_delta: 180,
            solve_budget: 100_000,
            zoom_pct: 100,
            certificate_game: false,
            min_solve_steps: 5,
        };
        let polycube = state::Polycube {
            voxels: vec![(0, 0, 0), (1, 0, 0)],
        };
        let target_ry = 90;
        let target_canonical = canonical_form(&polycube, 0, target_ry, 0);
        let puzzle = Arc::new(state::MentalRotationPuzzle {
            polycube,
            target_rx: 0,
            target_ry,
            target_rz: 0,
            increment: 90,
            target_canonical,
            start: (0, 0, 0),
            solvable: true,
            target_polycube: None,
            shadow_target: None,
        });
        let start = MentalRotationState {
            puzzle: puzzle.clone(),
            rx: 0,
            ry: 0,
            rz: 0,
        };

        let at_90 = MentalRotationState {
            puzzle: puzzle.clone(),
            rx: 0,
            ry: 90,
            rz: 0,
        };
        let at_270 = MentalRotationState {
            puzzle: puzzle.clone(),
            rx: 0,
            ry: 270,
            rz: 0,
        };
        assert!(game.is_solved(&at_90));
        assert!(
            !game.is_solved(&at_270),
            "270° should NOT solve — different face colors"
        );

        let (_action, cost) = game.solve(&start, 100_000).expect("should be solvable");
        assert_eq!(cost, 90, "symmetric piece should have 90° angular distance");
    }

    #[test]
    fn expert_solves_in_optimal_steps() {
        let game = Arc::new(make_game());
        for seed in 0..20u64 {
            let mut rng = ChaChaRng::seed_from_u64(seed);
            let state = game.generate(&mut rng).expect("seed should generate");
            let optimal = game
                .solve_path(&state, 100_000)
                .expect("should have solve path")
                .len();
            assert!(
                optimal >= 5,
                "seed {}: optimal {} < min_solve_steps 5",
                seed,
                optimal
            );

            let mut ep = GameEpisode::new(game.clone(), state, 100_000);
            let mut steps = 0;
            while !ep.is_solved() && steps < 20 {
                let action = ep.expert_action();
                assert!(
                    action.starts_with("rotate"),
                    "seed {}: step {}: expected rotate action, got '{}'",
                    seed,
                    steps,
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
            assert!(
                ep.is_solved(),
                "seed {}: not solved after {} steps",
                seed,
                steps
            );
            assert!(
                steps <= optimal,
                "seed {}: took {} steps (optimal {})",
                seed,
                steps,
                optimal
            );
        }
    }

    #[test]
    fn small_max_step_takes_more_moves() {
        // With max_step=45 and a 90° delta on one axis, solver needs 2 steps.
        let game = Arc::new(MentalRotation {
            num_cubes_range: (2, 2),
            increment: 90,
            max_step: 45,
            max_delta: 180,
            solve_budget: 100_000,
            zoom_pct: 100,
            certificate_game: false,
            min_solve_steps: 5,
        });
        let polycube = state::Polycube {
            voxels: vec![(0, 0, 0), (1, 0, 0)],
        };
        let target_ry = 90;
        let target_canonical = canonical_form(&polycube, 0, target_ry, 0);
        let puzzle = Arc::new(state::MentalRotationPuzzle {
            polycube,
            target_rx: 0,
            target_ry,
            target_rz: 0,
            increment: 90,
            target_canonical,
            start: (0, 0, 0),
            solvable: true,
            target_polycube: None,
            shadow_target: None,
        });
        let start = MentalRotationState {
            puzzle,
            rx: 0,
            ry: 0,
            rz: 0,
        };

        // Cost should be ceil(90/45) = 2
        let (_, cost) = game.solve(&start, 100_000).expect("should be solvable");
        assert_eq!(cost, 90, "90° delta should have 90° angular distance");

        // solve_path should produce exactly 2 actions
        let path = game.solve_path(&start, 100_000).expect("should have path");
        assert_eq!(path.len(), 2, "should take 2 steps with max_step=45");

        // Expert should solve in 2 moves
        let mut ep = GameEpisode::new(game, start, 100_000);
        let mut steps = 0;
        while !ep.is_solved() && steps < 10 {
            let action = ep.expert_action();
            assert!(
                action.starts_with("rotate"),
                "expected rotate, got '{}'",
                action
            );
            assert!(ep.apply(&action), "action '{}' failed", action);
            steps += 1;
        }
        assert!(ep.is_solved(), "not solved after {} steps", steps);
        assert_eq!(steps, 2, "expected exactly 2 steps with max_step=45");
    }

    // --- Certificate mode tests ---

    fn make_cert_game() -> MentalRotation {
        MentalRotation {
            num_cubes_range: (9, 9),
            increment: 90,
            max_step: 45,
            max_delta: 180,
            solve_budget: 100_000,
            zoom_pct: 100,
            certificate_game: true,
            min_solve_steps: 5,
        }
    }

    #[test]
    fn certificate_expert_says_yes_when_solvable() {
        let game = Arc::new(make_cert_game());
        for seed in 0..10u64 {
            let mut rng = ChaChaRng::seed_from_u64(seed);
            // Generate a solvable puzzle
            let state =
                gen::generate(&mut rng, 9, 90, 180, 45, 5, 100_000).expect("seed should generate");
            assert!(state.puzzle.solvable);

            let mut ep = GameEpisode::new(game.clone(), state, 100_000);
            let mut steps = 0;
            while steps < 20 {
                let action = ep.expert_action();
                if action == "yes" {
                    break;
                }
                assert!(
                    action.starts_with("rotate"),
                    "seed {}: expected rotate or yes, got '{}'",
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
                "yes",
                "seed {}: expert should say yes when solved",
                seed
            );
        }
    }

    #[test]
    fn certificate_expert_says_no_when_unsolvable() {
        let game = Arc::new(make_cert_game());
        for seed in 0..10u64 {
            let mut rng = ChaChaRng::seed_from_u64(seed);
            let state = gen::generate_unsolvable(&mut rng, 9, 90, 180, 45, 5, 100_000)
                .expect("seed should generate unsolvable");
            assert!(!state.puzzle.solvable);

            let mut ep = GameEpisode::new(game.clone(), state, 100_000);
            let mut steps = 0;
            while steps < 20 {
                let action = ep.expert_action();
                if action == "no" {
                    break;
                }
                assert!(
                    action.starts_with("rotate"),
                    "seed {}: expected rotate or no, got '{}'",
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
                "seed {}: expert should say no for unsolvable",
                seed
            );
        }
    }

    #[test]
    fn certificate_enumerate_actions_has_yes_no() {
        let game = Arc::new(make_cert_game());
        let mut rng = ChaChaRng::seed_from_u64(42);
        let state = game.generate(&mut rng).expect("seed should generate");
        let ep = GameEpisode::new(game, state, 100_000);
        let actions = ep.enumerate_actions();
        assert!(actions.contains(&"yes".to_string()));
        assert!(actions.contains(&"no".to_string()));
        assert!(!actions.contains(&"done".to_string()));
    }

    #[test]
    fn noncertificate_enumerate_actions_has_done() {
        let game = Arc::new(make_game());
        let mut rng = ChaChaRng::seed_from_u64(42);
        let state = game.generate(&mut rng).expect("seed should generate");
        let ep = GameEpisode::new(game, state, 100_000);
        let actions = ep.enumerate_actions();
        assert!(actions.contains(&"done".to_string()));
        assert!(!actions.contains(&"yes".to_string()));
        assert!(!actions.contains(&"no".to_string()));
    }

    // --- Certificate answers are accepted from any state ---

    fn first_cert_solvable() -> MentalRotationState {
        for seed in 0..20u64 {
            let mut rng = ChaChaRng::seed_from_u64(seed);
            if let Some(state) = gen::generate(&mut rng, 9, 90, 180, 45, 5, 100_000) {
                if state.puzzle.solvable {
                    return state;
                }
            }
        }
        panic!("no solvable mental_rotation cert puzzle in 20 seeds");
    }

    fn first_cert_unsolvable() -> MentalRotationState {
        for seed in 0..20u64 {
            let mut rng = ChaChaRng::seed_from_u64(seed);
            if let Some(state) = gen::generate_unsolvable(&mut rng, 9, 90, 180, 45, 5, 100_000) {
                return state;
            }
        }
        panic!("no unsolvable mental_rotation cert puzzle in 20 seeds");
    }

    #[test]
    fn certificate_solvable_accepts_yes_at_initial_state() {
        let game = Arc::new(make_cert_game());
        let state = first_cert_solvable();
        let mut ep = GameEpisode::new(game, state, 100_000);
        assert!(ep.apply("yes"));
        assert!(ep.is_solved());
    }

    #[test]
    fn certificate_solvable_accepts_yes_after_partial_rotation() {
        let game = Arc::new(make_cert_game());
        let state = first_cert_solvable();
        let mut ep = GameEpisode::new(game, state, 100_000);
        // Apply a non-target rotation (intermediate state).
        assert!(ep.apply("rotatex(45)"));
        assert!(ep.apply("yes"), "should accept yes mid-trajectory");
        assert!(ep.is_solved());
    }

    #[test]
    fn certificate_solvable_rejects_no() {
        let game = Arc::new(make_cert_game());
        let state = first_cert_solvable();
        let mut ep = GameEpisode::new(game, state, 100_000);
        assert!(!ep.apply("no"));
        assert!(!ep.is_solved());
    }

    #[test]
    fn certificate_unsolvable_accepts_no_at_initial_state() {
        let game = Arc::new(make_cert_game());
        let state = first_cert_unsolvable();
        let mut ep = GameEpisode::new(game, state, 100_000);
        assert!(ep.apply("no"));
        assert!(ep.is_solved());
    }

    #[test]
    fn certificate_unsolvable_rejects_yes() {
        let game = Arc::new(make_cert_game());
        let state = first_cert_unsolvable();
        let mut ep = GameEpisode::new(game, state, 100_000);
        assert!(!ep.apply("yes"));
        assert!(!ep.is_solved());
    }
}
