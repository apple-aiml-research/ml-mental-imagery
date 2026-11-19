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
use crate::base::state_text::{json_string, StateField};
use crate::base::template::{ActionTemplate, ParamType};

use state::{Cell, Direction, SokobanState, ALL_DIRECTIONS};

/// Sokoban game configuration.
pub struct Sokoban {
    pub grid_size_range: (usize, usize),
    pub num_boxes_range: (usize, usize),
    pub solve_budget: usize,
    pub wall_density: u32,
}

#[cfg(test)]
impl Default for Sokoban {
    fn default() -> Self {
        Sokoban {
            grid_size_range: (6, 6),
            num_boxes_range: (1, 2),
            solve_budget: 100_000,
            wall_density: 20,
        }
    }
}

const SOLVE_BUDGET: usize = 100_000;
/// Percentage of interior cells turned into walls.
const WALL_DENSITY: u32 = 20;

pub fn build_factory(
    name: &str,
    params: &[super::registry::ConfigParam],
    overrides: &std::collections::HashMap<String, super::registry::ConfigValue>,
) -> Result<Box<dyn crate::stream::loader::GameFactory>, String> {
    let (gs_lo, gs_hi) = super::registry::resolve_int_range(params, overrides, "grid_size_range")?;
    let (lo, hi) = super::registry::resolve_int_range(params, overrides, "num_boxes_range")?;
    let game = Sokoban {
        grid_size_range: (gs_lo as usize, gs_hi as usize),
        num_boxes_range: (lo as usize, hi as usize),
        solve_budget: SOLVE_BUDGET,
        wall_density: WALL_DENSITY,
    };
    Ok(crate::stream::loader::game_factory(
        game,
        name,
        SOLVE_BUDGET,
    ))
}

inventory::submit! {
    super::registry::GameEntry {
        name: "sokoban",
        params: &[
            super::registry::ConfigParam {
                key: "grid_size_range",
                param_type: super::registry::ConfigParamType::IntRange,
                default: super::registry::ConfigValue::IntRange(6, 6),
            },
            super::registry::ConfigParam {
                key: "num_boxes_range",
                param_type: super::registry::ConfigParamType::IntRange,
                default: super::registry::ConfigValue::IntRange(1, 2),
            },
        ],
        build: build_factory,
    }
}

impl Game for Sokoban {
    type State = SokobanState;
    type Action = Direction;

    fn generate(&self, rng: &mut ChaChaRng) -> Option<SokobanState> {
        use rand::Rng;
        let grid_size = rng.gen_range(self.grid_size_range.0..=self.grid_size_range.1);
        gen::generate(
            rng,
            grid_size,
            self.num_boxes_range,
            self.solve_budget,
            self.wall_density,
        )
    }

    fn puzzle_fingerprint(&self, state: &SokobanState) -> u64 {
        use std::hash::Hasher;
        let layout = &*state.layout;
        crate::base::fingerprint::fingerprint(|h| {
            h.write_u64(layout.width as u64);
            h.write_u64(layout.height as u64);
            h.write_u64(state.box_positions.len() as u64);
            // Row-major grid: encodes walls and goals together (Cell is repr(u8)).
            for &cell in &layout.grid {
                h.write_u8(cell as u8);
            }
            // Initial box positions, sorted (boxes are interchangeable).
            let mut boxes = state.box_positions.clone();
            boxes.sort_unstable();
            for (col, row) in boxes {
                h.write_u64(col as u64);
                h.write_u64(row as u64);
            }
            // Initial player cell.
            h.write_u64(state.player_col as u64);
            h.write_u64(state.player_row as u64);
        })
    }

    fn trivial(&self) -> SokobanState {
        // Border walls, all interior floor, player at (2,2), box at (2,3), goal at (2,4).
        // 1-push solve: push box down.
        let gs = self.grid_size_range.0.max(6);
        let mut grid = vec![Cell::Floor; gs * gs];
        for row in 0..gs {
            for col in 0..gs {
                if row == 0 || row == gs - 1 || col == 0 || col == gs - 1 {
                    grid[row * gs + col] = Cell::Wall;
                }
            }
        }
        let goal_pos = (2, 4);
        grid[goal_pos.1 * gs + goal_pos.0] = Cell::Goal;
        let layout = std::sync::Arc::new(state::BoardLayout {
            width: gs,
            height: gs,
            grid,
            goal_positions: vec![goal_pos],
        });
        SokobanState::new(layout, 2, 2, vec![(2, 3)])
    }

    fn actions(&self, _state: &SokobanState) -> Vec<Direction> {
        ALL_DIRECTIONS.to_vec()
    }

    fn apply(&self, state: &SokobanState, action: &Direction) -> Option<SokobanState> {
        let new_player_col = state.player_col as i32 + action.dc();
        let new_player_row = state.player_row as i32 + action.dr();

        // Out of bounds
        if !state.layout.in_bounds(new_player_col, new_player_row) {
            return None;
        }

        let npc = new_player_col as usize;
        let npr = new_player_row as usize;

        // Wall collision
        if state.layout.cell_at(npc, npr) == Cell::Wall {
            return None;
        }

        // Pushing a box?
        if state.has_box(npc, npr) {
            let new_box_col = npc as i32 + action.dc();
            let new_box_row = npr as i32 + action.dr();

            if !state.layout.in_bounds(new_box_col, new_box_row) {
                return None;
            }

            let nbc = new_box_col as usize;
            let nbr = new_box_row as usize;

            if state.layout.cell_at(nbc, nbr) == Cell::Wall {
                return None;
            }

            if state.has_box(nbc, nbr) {
                return None;
            }

            let mut new_boxes: Vec<(usize, usize)> = state
                .box_positions
                .iter()
                .copied()
                .filter(|&pos| pos != (npc, npr))
                .collect();
            new_boxes.push((nbc, nbr));

            return Some(SokobanState::new(state.layout.clone(), npc, npr, new_boxes));
        }

        // Simple move
        Some(SokobanState::new(
            state.layout.clone(),
            npc,
            npr,
            state.box_positions.clone(),
        ))
    }

    fn is_solved(&self, state: &SokobanState) -> bool {
        state
            .layout
            .goal_positions
            .iter()
            .all(|g| state.has_box(g.0, g.1))
    }

    fn render(&self, state: &SokobanState, resolution: u32) -> RgbBuffer {
        render::render(state, resolution)
    }

    fn action_to_str(&self, action: &Direction) -> String {
        action.name().to_string()
    }

    fn parse_action(&self, _state: &SokobanState, s: &str) -> Option<Direction> {
        match s {
            "up" => Some(Direction::Up),
            "down" => Some(Direction::Down),
            "left" => Some(Direction::Left),
            "right" => Some(Direction::Right),
            _ => None,
        }
    }

    fn action_templates(&self, _state: &SokobanState) -> Vec<ActionTemplate> {
        vec![
            ActionTemplate {
                segments: vec![ParamType::Literal("up".into())],
            },
            ActionTemplate {
                segments: vec![ParamType::Literal("down".into())],
            },
            ActionTemplate {
                segments: vec![ParamType::Literal("left".into())],
            },
            ActionTemplate {
                segments: vec![ParamType::Literal("right".into())],
            },
        ]
    }

    fn prompt(&self, _state: &SokobanState) -> String {
        "Solve this Sokoban puzzle. Push the brown boxes onto the red diamonds.".to_string()
    }

    fn action_description(&self, _state: &SokobanState) -> String {
        "up, down, left, right".to_string()
    }

    fn object_boxes(&self, state: &SokobanState) -> Vec<crate::base::object_boxes::ObjectBox> {
        object_boxes::object_boxes(state)
    }

    /// `g`: the board as one string without row breaks, using `#` wall,
    /// `.` floor, `@` player, `O` box, `X` goal, `*` box on goal, `+` player on goal.
    fn state_fields(&self, state: &SokobanState) -> Vec<StateField> {
        vec![("g", json_string(&state::to_ascii(state).replace('\n', "")))]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::episode::{Episode, GameEpisode};
    use crate::games::sokoban::state::from_ascii;
    use rand::SeedableRng;
    use std::sync::Arc;

    #[test]
    fn state_text_strips_row_breaks() {
        let state = from_ascii("#####\n#@OX#\n#####");
        let episode = GameEpisode::new(Arc::new(Sokoban::default()), state, 100_000);
        assert_eq!(
            episode.state_text(3),
            "{\"g\":\"######@OX######\",\"moves\":3}"
        );
    }

    #[test]
    fn apply_simple_move() {
        let state = from_ascii("####\n#.@#\n#OX#\n####");
        let game = Sokoban::default();
        // Move down: player at (2,1) moves to (2,2) which is a goal, no box
        let new_state = game.apply(&state, &Direction::Down).unwrap();
        assert_eq!(new_state.player_col, 2);
        assert_eq!(new_state.player_row, 2);
        // Boxes unchanged
        assert_eq!(new_state.box_positions, state.box_positions);
    }

    #[test]
    fn apply_push_box() {
        // Player at (1,1), box at (2,1), push right
        let state = from_ascii("#####\n#@O.#\n#...#\n#####");
        let game = Sokoban::default();
        let new_state = game.apply(&state, &Direction::Right).unwrap();
        assert_eq!(new_state.player_col, 2);
        assert_eq!(new_state.player_row, 1);
        assert!(new_state.has_box(3, 1));
        assert!(!new_state.has_box(2, 1));
    }

    #[test]
    fn apply_wall_collision() {
        let state = from_ascii("####\n#@.#\n#..#\n####");
        let game = Sokoban::default();
        // Move up: wall at row 0
        assert!(game.apply(&state, &Direction::Up).is_none());
    }

    #[test]
    fn apply_box_on_box() {
        // Player pushes box into another box
        let state = from_ascii("######\n#@OO.#\n######");
        let game = Sokoban::default();
        // Push right: box at (2,1) would push into box at (3,1)
        assert!(game.apply(&state, &Direction::Right).is_none());
    }

    #[test]
    fn is_solved_positive() {
        // Box on goal
        let state = from_ascii("####\n#@*#\n####");
        let game = Sokoban::default();
        assert!(game.is_solved(&state));
    }

    #[test]
    fn is_solved_negative() {
        // Box not on goal
        let state = from_ascii("####\n#@.#\n#OX#\n####");
        let game = Sokoban::default();
        assert!(!game.is_solved(&state));
    }

    #[test]
    fn action_to_str_parse_roundtrip() {
        let game = Sokoban::default();
        let state = from_ascii("####\n#@.#\n####");
        for dir in &ALL_DIRECTIONS {
            let s = game.action_to_str(dir);
            let parsed = game.parse_action(&state, &s).unwrap();
            assert_eq!(*dir, parsed);
        }
    }

    #[test]
    fn action_templates_no_meta() {
        let game = Sokoban::default();
        let state = from_ascii("####\n#@.#\n####");
        let templates = game.action_templates(&state);
        let mut actions: Vec<String> = templates.iter().flat_map(|t| t.expand()).collect();
        actions.sort();
        assert_eq!(actions, vec!["down", "left", "right", "up"]);
    }

    #[test]
    fn solve_trivial() {
        // Box right next to goal, player can push it
        let state = from_ascii("#####\n#@OX#\n#####");
        let game = Sokoban::default();
        let result = game.solve(&state, 10_000);
        assert!(result.is_some());
        let (first_action, cost) = result.unwrap();
        assert_eq!(first_action, Direction::Right);
        assert_eq!(cost, 1);
    }

    #[test]
    fn registered_in_registry() {
        assert!(crate::games::registry::find_game("sokoban").is_some());
    }

    #[test]
    fn test_aspect_ratio_default() {
        let game = Sokoban::default();
        assert_eq!(game.aspect_ratio(), (1, 1));
    }

    #[test]
    fn generate_produces_solvable() {
        let game = Sokoban {
            grid_size_range: (6, 6),
            num_boxes_range: (1, 2),
            solve_budget: 100_000,
            wall_density: 20,
        };
        for seed in 0..10u64 {
            let mut rng = ChaChaRng::seed_from_u64(seed);
            let state = game.generate(&mut rng).expect("test seed should generate");
            let result = game.solve(&state, 100_000);
            assert!(result.is_some(), "Seed {} produced unsolvable puzzle", seed);
        }
    }

    #[test]
    fn trivial_is_solvable() {
        let game = Sokoban::default();
        let state = game.trivial();
        let result = game.solve(&state, 100_000);
        assert!(result.is_some(), "trivial state should be solvable");
    }

    // --- GameEpisode tests (exercise the Episode bridge with Sokoban) ---

    fn make_episode(ascii: &str) -> GameEpisode<Sokoban> {
        let state = from_ascii(ascii);
        let game = Arc::new(Sokoban::default());
        GameEpisode::new(game, state, 100_000)
    }

    #[test]
    fn episode_apply_undo_roundtrip() {
        let mut ep = make_episode("####\n#@.#\n#..#\n####");
        let original_col = ep.state.player_col;
        let original_row = ep.state.player_row;

        assert!(ep.apply("right"));
        assert_ne!(ep.state.player_col, original_col);

        ep.undo();
        assert_eq!(ep.state.player_col, original_col);
        assert_eq!(ep.state.player_row, original_row);
    }

    #[test]
    fn episode_apply_invalid_returns_false() {
        let mut ep = make_episode("####\n#@.#\n####");
        let orig_col = ep.state.player_col;
        let orig_row = ep.state.player_row;

        // "up" hits a wall
        assert!(!ep.apply("up"));
        assert_eq!(ep.state.player_col, orig_col);
        assert_eq!(ep.state.player_row, orig_row);

        // Nonsense action
        assert!(!ep.apply("fly"));
    }

    #[test]
    fn episode_reset_restores_initial() {
        let mut ep = make_episode("####\n#@.#\n#..#\n####");
        let initial_col = ep.state.player_col;
        let initial_row = ep.state.player_row;

        ep.apply("right");
        ep.apply("down");
        assert_ne!(ep.state.player_col, initial_col);

        ep.reset();
        assert_eq!(ep.state.player_col, initial_col);
        assert_eq!(ep.state.player_row, initial_row);
        assert!(ep.history.is_empty());
    }

    #[test]
    fn episode_expert_action_returns_optimal() {
        // Player must push box right onto goal: @OX → push right
        let ep = make_episode("#####\n#@OX#\n#####");
        let action = ep.expert_action();
        assert_eq!(action, "right");
    }

    #[test]
    fn episode_expert_action_done_when_solved() {
        let ep = make_episode("####\n#@*#\n####");
        assert_eq!(ep.expert_action(), "done");
    }

    #[test]
    fn episode_back_reset_from_dead_state() {
        // Board:
        //   #####
        //   #..X#    Goal at (3,1)
        //   #.O.#    Box at (2,2)
        //   #.@.#    Player at (2,3)
        //   #####
        //
        // Optimal solution: push up (box to (2,1)), left, up, right
        // (push box to (3,1) = goal). 4 moves.
        //
        // Deadlock path below: push box up to (2,1), navigate player
        // around to (3,1), then push box left into the top-left interior
        // corner (1,1). Wall above (row 0) + wall left (col 0) = permanent
        // corner deadlock. Box can never leave: pushing right requires
        // player at (0,1) = wall, pushing down requires player at (1,0) = wall.
        let mut ep = make_episode("#####\n#..X#\n#.O.#\n#.@.#\n#####");

        // Verify the initial state is solvable via forward actions
        let initial_action = ep.expert_action();
        assert!(
            initial_action != "back" && initial_action != "reset" && initial_action != "done",
            "Initial state should be solvable via forward actions, got: {initial_action}"
        );

        // Step 1: push box up — player (2,3)→(2,2), box (2,2)→(2,1)
        assert!(ep.apply("up"));
        assert_eq!(ep.state.player_col, 2);
        assert_eq!(ep.state.player_row, 2);
        assert!(ep.state.has_box(2, 1));

        // Step 2: player moves right to (3,2)
        assert!(ep.apply("right"));

        // Step 3: player moves up to (3,1)
        assert!(ep.apply("up"));

        // Step 4: push box left — player (3,1)→(2,1), box (2,1)→(1,1). DEADLOCK.
        assert!(ep.apply("left"));
        assert!(ep.state.has_box(1, 1));
        assert!(!ep.is_solved()); // (1,1) is not the goal (3,1)

        // Box at (1,1) is permanently stuck in a corner.
        // Current state is unsolvable. Here reset (cost 5) is strictly
        // cheaper than back (cost 6), so reset is the correct choice.
        let action = ep.expert_action();
        assert_eq!(
            action, "reset",
            "Dead state should recommend 'reset' when it's strictly cheaper than back"
        );
    }

    #[test]
    fn episode_cache_consistency() {
        // Check that cached costs match fresh BFS solves
        let state = from_ascii("#####\n#@OX#\n#####");
        let game = Arc::new(Sokoban::default());
        let ep = GameEpisode::new(game.clone(), state.clone(), 100_000);

        let cached_cost = ep.cached_solve(&state);
        let fresh = game.solve(&state, 100_000).map(|(_, c)| c);
        assert_eq!(cached_cost, fresh);
    }

    #[test]
    fn episode_enumerate_includes_meta() {
        let ep = make_episode("#####\n#@OX#\n#####");
        let actions = ep.enumerate_actions();
        assert!(actions.contains(&"up".to_string()));
        assert!(actions.contains(&"down".to_string()));
        assert!(actions.contains(&"left".to_string()));
        assert!(actions.contains(&"right".to_string()));
        assert!(actions.contains(&"back".to_string()));
        assert!(actions.contains(&"reset".to_string()));
        assert!(actions.contains(&"done".to_string()));
        assert_eq!(actions.len(), 7); // 4 game + 3 meta
    }

    #[test]
    fn expert_prefers_back_in_unsolvable_state_when_costs_equal() {
        // One push from initial creates a corner deadlock. Since only one
        // move was taken, "back" and "reset" both restore the initial state
        // and have identical cost. Expert should prefer "back".
        //
        //   ######
        //   #....#
        //   #.O@X#   Box(2,2), Player(3,2), Goal(4,2)
        //   ######
        //
        // Push left: box (2,2)→(1,2). Corner deadlock (wall left + wall below).
        let mut ep = make_episode("######\n#....#\n#.O@X#\n######");
        assert!(
            ep.cached_solve(&ep.initial).is_some(),
            "initial should be solvable"
        );

        assert!(ep.apply("left")); // push box into corner deadlock

        let back_cost = ep.cached_solve(ep.history.last().unwrap());
        let reset_cost = ep.cached_solve(&ep.initial);
        assert_eq!(
            back_cost, reset_cost,
            "precondition: back and reset should have equal cost (both restore initial)"
        );

        assert_eq!(
            ep.expert_action(),
            "back",
            "unsolvable state: should prefer back over reset when costs are equal"
        );
    }

    #[test]
    fn expert_prefers_back_over_reset_when_costs_equal() {
        // After one non-push move from initial, "back" and "reset" both
        // restore the same initial state, so they have identical cost.
        // Expert should prefer "back" over "reset".
        //
        //   #####
        //   #@OX#   Player(1,1), Box(2,1), Goal(3,1)
        //   #...#
        //   #####
        let mut ep = make_episode("#####\n#@OX#\n#...#\n#####");
        assert!(ep.apply("down")); // wasteful move to (1,2)

        let back_cost = ep.cached_solve(ep.history.last().unwrap());
        let reset_cost = ep.cached_solve(&ep.initial);
        assert_eq!(
            back_cost, reset_cost,
            "precondition: back and reset should have equal cost"
        );

        assert_ne!(
            ep.expert_action(),
            "reset",
            "expert should prefer back over reset when they have equal cost"
        );
    }

    #[test]
    fn expert_picks_reset_over_back_when_cheaper() {
        // Push box away from goal twice. "reset" restores the closer
        // initial state; "back" restores the worse intermediate state.
        //
        //   #######
        //   #X@O..#   Player(2,1), Box(3,1), Goal(1,1)
        //   #.....#
        //   #######
        let mut ep = make_episode("#######\n#X@O..#\n#.....#\n#######");
        assert!(ep.apply("right")); // push box right: (3,1)→(4,1)
        assert!(ep.apply("right")); // push box right: (4,1)→(5,1)

        let back_cost = ep.cached_solve(ep.history.last().unwrap()).unwrap();
        let reset_cost = ep.cached_solve(&ep.initial).unwrap();
        assert!(
            reset_cost < back_cost,
            "precondition: reset ({reset_cost}) should be cheaper than back ({back_cost})"
        );

        assert_ne!(
            ep.expert_action(),
            "back",
            "expert should not pick back when reset is strictly cheaper"
        );
    }

    #[test]
    fn expert_picks_back_over_reset_when_cheaper() {
        // Push box toward goal, then take a wasteful non-push move.
        // "back" restores the progress state; "reset" loses all progress.
        //
        //   ########
        //   #X...O@#   Player(6,1), Box(5,1), Goal(1,1)
        //   #......#
        //   ########
        let mut ep = make_episode("########\n#X...O@#\n#......#\n########");
        assert!(ep.apply("left")); // push box left: (5,1)→(4,1)
        assert!(ep.apply("left")); // push box left: (4,1)→(3,1)
        assert!(ep.apply("down")); // wasteful move

        let back_cost = ep.cached_solve(ep.history.last().unwrap()).unwrap();
        let reset_cost = ep.cached_solve(&ep.initial).unwrap();
        assert!(
            back_cost < reset_cost,
            "precondition: back ({back_cost}) should be cheaper than reset ({reset_cost})"
        );

        assert_ne!(
            ep.expert_action(),
            "reset",
            "expert should not pick reset when back is strictly cheaper"
        );
    }
}
