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

use state::{canonical_form_2d, shortest_delta, MentalRotation2dState, Rotation2dAction};

/// 2D mental rotation game configuration.
pub struct MentalRotation2d {
    pub num_cells: usize,
    pub increment: u16,
    pub max_step: i32,
    pub max_delta: u16,
    pub solve_budget: usize,
    pub certificate_game: bool,
    pub min_steps: usize,
}

/// Largest rotation per action, in degrees.
const MAX_STEP: i32 = 30;
/// Largest initial offset, in degrees.
const MAX_DELTA: u16 = 180;
/// Fewest rotation actions in an expert solution.
const MIN_STEPS: usize = 3;
const SOLVE_BUDGET: usize = 100_000;

pub fn build_factory(
    name: &str,
    params: &[super::registry::ConfigParam],
    overrides: &std::collections::HashMap<String, super::registry::ConfigValue>,
) -> Result<Box<dyn crate::stream::loader::GameFactory>, String> {
    let num_cells = super::registry::resolve_int(params, overrides, "num_cells")? as usize;
    let increment = super::registry::resolve_int(params, overrides, "increment")? as u16;
    if !matches!(increment, 15 | 30 | 45 | 90) {
        return Err(format!(
            "increment must be 15, 30, 45, or 90, got {}",
            increment
        ));
    }
    let certificate_game =
        super::registry::resolve_int(params, overrides, "certificate_game")? != 0;
    let game = MentalRotation2d {
        num_cells,
        increment,
        max_step: MAX_STEP,
        max_delta: MAX_DELTA,
        solve_budget: SOLVE_BUDGET,
        certificate_game,
        min_steps: MIN_STEPS,
    };
    Ok(crate::stream::loader::game_factory(
        game,
        name,
        SOLVE_BUDGET,
    ))
}

inventory::submit! {
    super::registry::GameEntry {
        name: "mental_rotation_2d",
        params: &[
            super::registry::ConfigParam {
                key: "num_cells",
                param_type: super::registry::ConfigParamType::Int,
                default: super::registry::ConfigValue::Int(5),
            },
            super::registry::ConfigParam {
                key: "increment",
                param_type: super::registry::ConfigParamType::Int,
                default: super::registry::ConfigValue::Int(15),
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

/// Find all rotation-only orientations that match a target canonical form.
fn matching_orientations(
    polyomino: &state::Polyomino,
    target_canonical: &[(i32, i32)],
    increment: u16,
) -> Vec<u16> {
    let steps = 360 / increment as u32;
    let mut matches = Vec::new();
    for i in 0..steps {
        let rot = (i * increment as u32) as u16;
        if canonical_form_2d(polyomino, rot, false) == target_canonical {
            matches.push(rot);
        }
    }
    matches
}

/// Find the closest rotation (from a set of equivalent ones) to the current state.
fn closest_orientation(cur_rot: u16, targets: &[u16]) -> Option<u16> {
    targets
        .iter()
        .min_by_key(|&&rot| shortest_delta(cur_rot, rot).abs())
        .copied()
}

/// Cost from current rotation to the nearest equivalent orientation.
fn min_cost_to_canonical(cur_rot: u16, targets: &[u16]) -> usize {
    targets
        .iter()
        .map(|&rot| shortest_delta(cur_rot, rot).abs() as usize)
        .min()
        .unwrap_or(0)
}

/// Analytical solver: find the nearest equivalent orientation (accounting for
/// rotational symmetries) and return the best first action toward it.
fn analytical_solve(
    state: &MentalRotation2dState,
    game: &MentalRotation2d,
) -> Option<(Rotation2dAction, usize)> {
    let targets = matching_orientations(
        &state.puzzle.polyomino,
        &state.puzzle.target_canonical,
        game.increment,
    );
    let trot = closest_orientation(state.rotation, &targets)?;
    let delta = shortest_delta(state.rotation, trot);
    let cost = delta.abs() as usize;
    if cost == 0 {
        return None;
    }
    let step = delta.signum() * delta.abs().min(game.max_step);
    Some((Rotation2dAction::Rotate(step), cost))
}

/// Shadow analytical solver: targets the shadow rotation for unsolvable puzzles.
fn shadow_analytical_solve(
    state: &MentalRotation2dState,
    game: &MentalRotation2d,
) -> Option<(Rotation2dAction, usize)> {
    let trot = state.puzzle.shadow_target?;
    let delta = shortest_delta(state.rotation, trot);
    let cost = delta.abs() as usize;
    if cost == 0 {
        return None;
    }
    let step = delta.signum() * delta.abs().min(game.max_step);
    Some((Rotation2dAction::Rotate(step), cost))
}

impl Game for MentalRotation2d {
    type State = MentalRotation2dState;
    type Action = Rotation2dAction;

    fn generate(&self, rng: &mut ChaChaRng) -> Option<MentalRotation2dState> {
        if self.certificate_game {
            gen::generate_certificate(
                rng,
                self.num_cells,
                self.increment,
                self.max_delta,
                self.max_step,
                self.min_steps,
                self.solve_budget,
            )
        } else {
            gen::generate(
                rng,
                self.num_cells,
                self.increment,
                self.max_delta,
                self.max_step,
                self.min_steps,
                self.solve_budget,
            )
        }
    }

    fn puzzle_fingerprint(&self, state: &MentalRotation2dState) -> u64 {
        use std::hash::Hasher;
        // Puzzle identity as presented. Excludes rotation (always 0)
        // and start (const), target_canonical (float-derived from included
        // fields), and shadow_target (derivable from target_rotation + solvable).
        let puzzle = &*state.puzzle;
        crate::base::fingerprint::fingerprint(|h| {
            h.write_u16(puzzle.increment);
            // Normalized polyomino geometry: translate to bbox-min origin, then
            // sort — removes growth-order/offset noise (no symmetry canonicalization).
            let mut cells = puzzle.polyomino.cells.clone();
            let min_x = cells.iter().map(|&(x, _)| x).min().unwrap_or(0);
            let min_y = cells.iter().map(|&(_, y)| y).min().unwrap_or(0);
            for cell in &mut cells {
                cell.0 -= min_x;
                cell.1 -= min_y;
            }
            cells.sort();
            h.write_u64(cells.len() as u64);
            for (x, y) in cells {
                h.write_i32(x);
                h.write_i32(y);
            }
            h.write_u16(puzzle.target_rotation);
            h.write_u8(puzzle.target_flipped as u8);
            h.write_u8(puzzle.solvable as u8);
            // Only the hue `color_seed % 360` is drawn (render.rs `colors_from_seed`);
            // hashing the raw seed would tell visually identical puzzles apart.
            h.write_u32(puzzle.color_seed % 360);
        })
    }

    fn trivial(&self) -> MentalRotation2dState {
        // L-shape, target is 90° rotation — different canonical form, solvable in 1 move
        let polyomino = state::Polyomino {
            cells: vec![(0, 0), (1, 0), (0, 1)],
        };
        let target_rotation = 90;
        let target_canonical = canonical_form_2d(&polyomino, target_rotation, false);
        let puzzle = std::sync::Arc::new(state::MentalRotation2dPuzzle {
            polyomino,
            target_rotation,
            target_flipped: false,
            increment: self.increment,
            target_canonical,
            start: 0,
            solvable: true,
            shadow_target: None,
            color_seed: 0,
        });
        MentalRotation2dState {
            puzzle,
            rotation: 0,
        }
    }

    fn actions(&self, _state: &MentalRotation2dState) -> Vec<Rotation2dAction> {
        let mut actions = Vec::with_capacity(360);
        for d in -180..=180i32 {
            if d == 0 {
                continue;
            }
            actions.push(Rotation2dAction::Rotate(d));
        }
        actions
    }

    fn apply(
        &self,
        state: &MentalRotation2dState,
        action: &Rotation2dAction,
    ) -> Option<MentalRotation2dState> {
        Some(state.apply_action(*action))
    }

    fn is_solved(&self, state: &MentalRotation2dState) -> bool {
        let current_canonical = canonical_form_2d(&state.puzzle.polyomino, state.rotation, false);
        current_canonical == state.puzzle.target_canonical
    }

    fn render(&self, state: &MentalRotation2dState, resolution: u32) -> RgbBuffer {
        render::render(state, resolution)
    }

    fn action_to_str(&self, action: &Rotation2dAction) -> String {
        let Rotation2dAction::Rotate(d) = action;
        format!("rotate({})", d)
    }

    fn parse_action(&self, _state: &MentalRotation2dState, s: &str) -> Option<Rotation2dAction> {
        if let Some(inner) = s.strip_prefix("rotate(").and_then(|r| r.strip_suffix(')')) {
            let d: i32 = inner.parse().ok()?;
            if d == 0 || d < -180 || d > 180 {
                return None;
            }
            return Some(Rotation2dAction::Rotate(d));
        }
        None
    }

    fn action_templates(&self, _state: &MentalRotation2dState) -> Vec<ActionTemplate> {
        vec![ActionTemplate {
            segments: vec![
                ParamType::Literal("rotate(".to_string()),
                ParamType::Int {
                    min: -180,
                    max: 180,
                },
                ParamType::Literal(")".to_string()),
            ],
        }]
    }

    fn prompt(&self, _state: &MentalRotation2dState) -> String {
        if self.certificate_game {
            "Determine if the left and right 2D shapes are the same (possibly rotated). \
             Rotate to check, then say 'yes' if they match or 'no' if they don't."
                .to_string()
        } else {
            "Rotate the left 2D shape to match the right shape.".to_string()
        }
    }

    fn action_description(&self, _state: &MentalRotation2dState) -> String {
        if self.certificate_game {
            "rotate(deg), yes, no".to_string()
        } else {
            "rotate(deg)".to_string()
        }
    }

    fn object_boxes(
        &self,
        state: &MentalRotation2dState,
    ) -> Vec<crate::base::object_boxes::ObjectBox> {
        object_boxes::object_boxes(state)
    }

    /// Current rotation (`r`), target rotation, signed shortest delta,
    /// rotation increment (`inc`), and polyomino size (`n_cells`). Unsolvable
    /// certificate puzzles use the solver's shadow target.
    fn state_fields(&self, state: &MentalRotation2dState) -> Vec<StateField> {
        let puzzle = &state.puzzle;
        let mut target = puzzle.target_rotation;
        if self.certificate_game && !puzzle.solvable {
            target = puzzle.shadow_target.unwrap_or(target);
        }
        vec![
            ("r", state.rotation.to_string()),
            ("target", target.to_string()),
            ("delta", shortest_delta(state.rotation, target).to_string()),
            ("inc", puzzle.increment.to_string()),
            ("n_cells", puzzle.polyomino.cells.len().to_string()),
        ]
    }

    fn solve(
        &self,
        state: &MentalRotation2dState,
        _budget: usize,
    ) -> Option<(Rotation2dAction, usize)> {
        if self.certificate_game && !state.puzzle.solvable {
            shadow_analytical_solve(state, self)
        } else {
            analytical_solve(state, self)
        }
    }

    fn solve_path(
        &self,
        state: &MentalRotation2dState,
        _budget: usize,
    ) -> Option<Vec<(Rotation2dAction, usize)>> {
        let use_shadow = self.certificate_game && !state.puzzle.solvable;
        let solver_fn = if use_shadow {
            shadow_analytical_solve
        } else {
            analytical_solve
        };

        let target_canonical = if use_shadow {
            let trot = state
                .puzzle
                .shadow_target
                .unwrap_or(state.puzzle.target_rotation);
            canonical_form_2d(&state.puzzle.polyomino, trot, false)
        } else {
            state.puzzle.target_canonical.clone()
        };
        let targets =
            matching_orientations(&state.puzzle.polyomino, &target_canonical, self.increment);

        let mut actions = Vec::new();
        let mut states = Vec::new();
        let mut current = state.clone();
        let limit = 1080;
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
            Some(
                actions
                    .into_iter()
                    .zip(states)
                    .map(|(a, st)| {
                        let cost = min_cost_to_canonical(st.rotation, &targets);
                        (a, cost)
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

    fn is_solvable(&self, state: &MentalRotation2dState) -> bool {
        state.puzzle.solvable
    }

    fn shadow_solve_cost(&self, state: &MentalRotation2dState) -> Option<usize> {
        if state.puzzle.solvable {
            return None;
        }
        if let Some(trot) = state.puzzle.shadow_target {
            let dist = shortest_delta(state.rotation, trot).abs() as usize;
            if dist <= self.max_step as usize {
                Some(0)
            } else {
                Some(dist)
            }
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

    fn make_game() -> MentalRotation2d {
        MentalRotation2d {
            num_cells: 5,
            increment: 90,
            max_step: 180,
            max_delta: 180,
            solve_budget: 100_000,
            certificate_game: false,
            min_steps: 1,
        }
    }

    #[test]
    fn state_text_reports_rotation_target_and_delta() {
        let game = make_game();
        let mut rng = ChaChaRng::seed_from_u64(42);
        let mut state = game.generate(&mut rng).expect("test seed should generate");
        let mut puzzle = (*state.puzzle).clone();
        puzzle.target_rotation = 135;
        state.puzzle = Arc::new(puzzle);
        state.rotation = 45;
        let fields = game.state_fields(&state);
        let text = crate::base::state_text::format_state_json(&fields, 3);
        assert!(
            text.starts_with(r#"{"r":45,"target":135,"delta":90,"inc":"#),
            "{text}"
        );
        assert!(text.ends_with(r#","n_cells":5,"moves":3}"#), "{text}");
    }

    #[test]
    fn action_roundtrip() {
        let game = make_game();
        let mut rng = ChaChaRng::seed_from_u64(42);
        let state = game.generate(&mut rng).expect("test seed should generate");
        let sample_actions = [
            Rotation2dAction::Rotate(90),
            Rotation2dAction::Rotate(-45),
            Rotation2dAction::Rotate(180),
            Rotation2dAction::Rotate(-1),
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
        let target = MentalRotation2dState {
            puzzle: state.puzzle.clone(),
            rotation: state.puzzle.target_rotation,
        };
        assert!(game.is_solved(&target));
    }

    #[test]
    fn is_solved_negative() {
        let game = make_game();
        let mut rng = ChaChaRng::seed_from_u64(42);
        let state = game.generate(&mut rng).expect("test seed should generate");
        assert!(!game.is_solved(&state));
    }

    #[test]
    fn registered_in_registry() {
        assert!(crate::games::registry::find_game("mental_rotation_2d").is_some());
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
        let initial_rot = state.rotation;

        let mut ep = GameEpisode::new(game, state, 100_000);

        assert!(ep.apply("rotate(90)"));
        assert_ne!(ep.state.rotation, initial_rot);

        ep.undo();
        assert_eq!(ep.state.rotation, initial_rot);

        ep.apply("rotate(-30)");
        ep.reset();
        assert_eq!(ep.state.rotation, initial_rot);
    }

    #[test]
    fn episode_expert_action() {
        let game = Arc::new(make_game());
        let mut rng = ChaChaRng::seed_from_u64(42);
        let state = game.generate(&mut rng).expect("test seed should generate");

        let ep = GameEpisode::new(game, state, 100_000);
        let action = ep.expert_action();
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

        let solved_state = MentalRotation2dState {
            puzzle: state.puzzle.clone(),
            rotation: state.puzzle.target_rotation,
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
    fn expert_solves_within_budget() {
        let game = Arc::new(make_game());
        for seed in 0..20u64 {
            let mut rng = ChaChaRng::seed_from_u64(seed);
            let state = game.generate(&mut rng).expect("seed should generate");
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
        }
    }

    // --- Certificate mode tests ---

    fn make_cert_game() -> MentalRotation2d {
        MentalRotation2d {
            num_cells: 5,
            increment: 90,
            max_step: 180,
            max_delta: 180,
            solve_budget: 100_000,
            certificate_game: true,
            min_steps: 1,
        }
    }

    #[test]
    fn certificate_expert_says_yes_when_solvable() {
        let game = Arc::new(make_cert_game());
        for seed in 0..10u64 {
            let mut rng = ChaChaRng::seed_from_u64(seed);
            let state =
                gen::generate(&mut rng, 5, 90, 180, 180, 1, 100_000).expect("seed should generate");
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
            let state = gen::generate_unsolvable(&mut rng, 5, 90, 180, 180, 1)
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

    #[test]
    fn min_steps_enforced() {
        let game = Arc::new(MentalRotation2d {
            num_cells: 5,
            increment: 15,
            max_step: 30,
            max_delta: 180,
            solve_budget: 100_000,
            certificate_game: false,
            min_steps: 3,
        });
        for seed in 0..20u64 {
            let mut rng = ChaChaRng::seed_from_u64(seed);
            let state = game.generate(&mut rng).expect("seed should generate");
            let mut ep = GameEpisode::new(game.clone(), state, 100_000);
            let mut steps = 0;
            while !ep.is_solved() && steps < 100 {
                let action = ep.expert_action();
                assert!(ep.apply(&action), "action '{}' failed", action);
                steps += 1;
            }
            assert!(
                ep.is_solved(),
                "seed {}: not solved after {} steps",
                seed,
                steps
            );
            assert!(
                steps >= 3,
                "seed {}: solved in {} steps, expected >= 3",
                seed,
                steps
            );
        }
    }

    #[test]
    fn certificate_min_steps_enforced() {
        let game = Arc::new(MentalRotation2d {
            num_cells: 5,
            increment: 15,
            max_step: 30,
            max_delta: 180,
            solve_budget: 100_000,
            certificate_game: true,
            min_steps: 3,
        });
        for seed in 0..20u64 {
            let mut rng = ChaChaRng::seed_from_u64(seed);
            let state = game.generate(&mut rng).expect("seed should generate");
            let mut ep = GameEpisode::new(game.clone(), state, 100_000);
            let mut steps = 0;
            while steps < 100 {
                let action = ep.expert_action();
                if action == "yes" || action == "no" {
                    break;
                }
                assert!(ep.apply(&action), "action '{}' failed", action);
                steps += 1;
            }
            assert!(
                steps >= 3,
                "seed {}: reached answer in {} rotation steps, expected >= 3",
                seed,
                steps
            );
        }
    }

    // --- Certificate answers are accepted from any state ---

    fn make_cert_game_2d() -> MentalRotation2d {
        MentalRotation2d {
            num_cells: 7,
            increment: 15,
            max_step: 30,
            max_delta: 180,
            solve_budget: 100_000,
            certificate_game: true,
            min_steps: 3,
        }
    }

    fn first_cert_solvable_2d() -> MentalRotation2dState {
        for seed in 0..50u64 {
            let mut rng = ChaChaRng::seed_from_u64(seed);
            if let Some(state) = gen::generate(&mut rng, 7, 15, 180, 30, 3, 100_000) {
                if state.puzzle.solvable {
                    return state;
                }
            }
        }
        panic!("no solvable mental_rotation_2d cert puzzle in 50 seeds");
    }

    fn first_cert_unsolvable_2d() -> MentalRotation2dState {
        for seed in 0..50u64 {
            let mut rng = ChaChaRng::seed_from_u64(seed);
            if let Some(state) = gen::generate_unsolvable(&mut rng, 7, 15, 180, 30, 3) {
                return state;
            }
        }
        panic!("no unsolvable mental_rotation_2d cert puzzle in 50 seeds");
    }

    #[test]
    fn certificate_solvable_accepts_yes_at_initial_state_2d() {
        let game = Arc::new(make_cert_game_2d());
        let state = first_cert_solvable_2d();
        let mut ep = GameEpisode::new(game, state, 100_000);
        assert!(ep.apply("yes"));
        assert!(ep.is_solved());
    }

    #[test]
    fn certificate_solvable_accepts_yes_after_partial_rotation_2d() {
        let game = Arc::new(make_cert_game_2d());
        let state = first_cert_solvable_2d();
        let mut ep = GameEpisode::new(game, state, 100_000);
        assert!(ep.apply("rotate(90)"));
        assert!(ep.apply("yes"), "should accept yes mid-trajectory");
        assert!(ep.is_solved());
    }

    #[test]
    fn certificate_solvable_rejects_no_2d() {
        let game = Arc::new(make_cert_game_2d());
        let state = first_cert_solvable_2d();
        let mut ep = GameEpisode::new(game, state, 100_000);
        assert!(!ep.apply("no"));
        assert!(!ep.is_solved());
    }

    #[test]
    fn certificate_unsolvable_accepts_no_at_initial_state_2d() {
        let game = Arc::new(make_cert_game_2d());
        let state = first_cert_unsolvable_2d();
        let mut ep = GameEpisode::new(game, state, 100_000);
        assert!(ep.apply("no"));
        assert!(ep.is_solved());
    }

    #[test]
    fn certificate_unsolvable_rejects_yes_2d() {
        let game = Arc::new(make_cert_game_2d());
        let state = first_cert_unsolvable_2d();
        let mut ep = GameEpisode::new(game, state, 100_000);
        assert!(!ep.apply("yes"));
        assert!(!ep.is_solved());
    }
}
