//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use crate::solver::bfs::state_hash;

use super::game::{Game, RgbBuffer};
use super::object_boxes::format_object_boxes;
use super::state_text::format_state_json;
use super::template::ActionTemplate;

/// Type-erased episode interface used by the episode streams (`engine/stream/`).
///
/// Different games coexist in the same batch via `Vec<Box<dyn Episode>>`.
pub trait Episode: Send {
    fn is_solved(&self) -> bool;
    fn apply(&mut self, action_str: &str) -> bool;
    fn undo(&mut self);
    fn reset(&mut self);

    fn expert_action(&self) -> String;
    fn enumerate_actions(&self) -> Vec<String>;

    fn render(&self, resolution: u32) -> RgbBuffer;
    fn prompt(&self) -> String;
    fn action_description(&self) -> String;

    /// Compact JSON description of the current state for State CoT, with
    /// `moves` = number of actions taken so far.
    fn state_text(&self, moves: usize) -> String;

    /// Compact JSON of the current state's object boxes for Boxes CoT,
    /// e.g. `{"player":[6,4],"box_0":[2,6]}`.
    fn object_boxes_text(&self) -> String;

    /// Returns the aspect ratio as (width, height). Default: (1, 1).
    fn aspect_ratio(&self) -> (u32, u32) {
        (1, 1)
    }

    /// Returns the optimal solution length (number of actions in the shortest
    /// solution). Returns None if unknown (e.g. solver budget exhausted).
    /// Used to compute per-episode action limits.
    fn optimal_len(&self) -> Option<usize> {
        None
    }

    /// Returns the number of expert actions needed to solve from the current state.
    /// Walks the solver cache without modifying the episode's actual state.
    fn solution_length(&self) -> Option<usize> {
        None
    }
}

/// Generic adapter that implements `Episode` for any `Game`.
///
/// Handles undo history, solver cache, and delegates to the game's methods.
pub struct GameEpisode<G: Game> {
    pub game: Arc<G>,
    pub state: G::State,
    pub initial: G::State,
    pub history: Vec<G::State>,
    /// Cache: state_hash -> (cost_to_goal, best_first_action)
    /// Uses RefCell for interior mutability since expert_action is &self.
    pub cache: RefCell<HashMap<u64, (usize, G::Action)>>,
    pub templates: Vec<ActionTemplate>,
    pub solve_budget: usize,
    /// Optimal solution length from solve_path. None if unsolvable within budget.
    pub optimal_len: Option<usize>,
    /// Whether a certificate action ("yes" or "no") has been successfully applied.
    /// In certificate mode, `is_solved()` returns this instead of checking board coverage.
    pub certificate_affirmed: bool,
}

impl<G: Game> GameEpisode<G> {
    /// Create a new GameEpisode, warming the cache along the optimal path.
    pub fn new(game: Arc<G>, initial: G::State, solve_budget: usize) -> Self {
        let templates = game.action_templates(&initial);
        let mut cache = HashMap::new();

        // Warm cache along the optimal path.
        // solve_path returns the full path in one call (BFS games get this from
        // a single bfs_solve; DLX games iterate internally).
        let optimal_len = if let Some(path) = game.solve_path(&initial, solve_budget) {
            let len = path.len();
            let mut current = initial.clone();
            for (action, cost) in &path {
                let h = state_hash(&current);
                cache.insert(h, (*cost, action.clone()));
                if let Some(next) = game.apply(&current, action) {
                    current = next;
                } else {
                    break;
                }
            }
            Some(len)
        } else {
            None
        };

        GameEpisode {
            game,
            state: initial.clone(),
            initial,
            history: Vec::new(),
            cache: RefCell::new(cache),
            templates,
            solve_budget,
            optimal_len,
            certificate_affirmed: false,
        }
    }

    /// Look up or compute the solver's recommended action and cost for a state.
    /// Returns None if solved (or shadow-exhausted for unsolvable certificate puzzles).
    ///
    /// This is the single source of truth for cache lookups and population.
    /// `cached_solve()` delegates here.
    pub fn cached_solve_action(&self, state: &G::State) -> Option<(G::Action, usize)> {
        if self.game.is_solved(state) {
            return None;
        }

        // Shadow solver exhausted — no more game actions, just "no"
        if self.game.certificate_mode()
            && !self.game.is_solvable(state)
            && self.game.shadow_solve_cost(state) == Some(0)
        {
            return None;
        }

        let h = state_hash(state);

        if let Some((cost, action)) = self.cache.borrow().get(&h) {
            return Some((action.clone(), *cost));
        }

        let result = self.game.solve(state, self.solve_budget);

        if let Some((action, cost)) = result {
            self.cache.borrow_mut().insert(h, (cost, action.clone()));
            Some((action, cost))
        } else {
            None
        }
    }

    /// Look up or compute the cost-to-goal for a state.
    /// Returns Some(0) for solved states (or shadow-exhausted unsolvable states).
    /// Returns None if unsolvable within budget.
    pub fn cached_solve(&self, state: &G::State) -> Option<usize> {
        if self.game.is_solved(state) {
            return Some(0);
        }
        // Shadow solver exhausted — cost is 0 (just say "no")
        if self.game.certificate_mode()
            && !self.game.is_solvable(state)
            && self.game.shadow_solve_cost(state) == Some(0)
        {
            return Some(0);
        }
        self.cached_solve_action(state).map(|(_, cost)| cost)
    }
}

#[cfg(test)]
impl<G: Game> GameEpisode<G> {
    /// Cost-to-goal after taking a game action from the current state.
    pub fn action_cost(&self, action_str: &str) -> Option<usize> {
        let action = self.game.parse_action(&self.state, action_str)?;
        let next = self.game.apply(&self.state, &action)?;
        self.cached_solve(&next).map(|cost| 1 + cost)
    }
}

impl<G: Game> Episode for GameEpisode<G> {
    fn is_solved(&self) -> bool {
        if self.game.certificate_mode() {
            return self.certificate_affirmed;
        }
        self.game.is_solved(&self.state)
    }

    fn apply(&mut self, action_str: &str) -> bool {
        match action_str {
            "back" => {
                if let Some(prev) = self.history.pop() {
                    self.state = prev;
                    true
                } else {
                    false
                }
            }
            "reset" => {
                self.state = self.initial.clone();
                self.history.clear();
                self.certificate_affirmed = false;
                true
            }
            // Certificate answers are accepted from any state: "yes" iff the
            // puzzle is solvable, "no" iff it is not.
            "yes" => {
                if self.game.certificate_mode() && self.game.is_solvable(&self.state) {
                    self.certificate_affirmed = true;
                    true
                } else {
                    false
                }
            }
            "no" => {
                if self.game.certificate_mode() && !self.game.is_solvable(&self.state) {
                    self.certificate_affirmed = true;
                    true
                } else {
                    false
                }
            }
            "done" => self.game.is_solved(&self.state),
            _ => {
                if let Some(action) = self.game.parse_action(&self.state, action_str) {
                    if let Some(next) = self.game.apply(&self.state, &action) {
                        self.history.push(self.state.clone());
                        self.state = next;
                        true
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
        }
    }

    fn undo(&mut self) {
        if let Some(prev) = self.history.pop() {
            self.state = prev;
        }
    }

    fn reset(&mut self) {
        self.state = self.initial.clone();
        self.history.clear();
    }

    fn expert_action(&self) -> String {
        // Certificate mode: check yes/no before regular actions
        if self.game.certificate_mode() {
            if self.game.is_solvable(&self.state) {
                if self.game.is_solved(&self.state) {
                    return "yes".to_string();
                }
            } else if self.game.shadow_solve_cost(&self.state) == Some(0) {
                return "no".to_string();
            }
        }

        if self.game.is_solved(&self.state) {
            return "done".to_string();
        }

        // Use the game's solver (via cache) — respects domain heuristics
        // (e.g. tangram's DLX largest-piece-first ordering).
        // `cost` from cached_solve_action is the total steps from current state
        // (including the first action), so it already represents the true forward cost.
        let (mut best_cost, mut best_action) = match self.cached_solve_action(&self.state) {
            Some((action, cost)) => (cost, self.game.action_to_str(&action)),
            None => (usize::MAX, "reset".to_string()),
        };

        // For unsolvable certificate puzzles, never go back — always move forward
        // to demonstrate effort before saying "no".
        let skip_back = self.game.certificate_mode() && !self.game.is_solvable(&self.state);

        // Check if back is strictly cheaper (prefer forward in a tie to avoid cycles)
        if !skip_back {
            if let Some(prev) = self.history.last() {
                if let Some(cost) = self.cached_solve(prev) {
                    let total = 1 + cost;
                    if total < best_cost {
                        best_cost = total;
                        best_action = "back".to_string();
                    }
                }
            }
        }

        // Check if reset is strictly cheaper
        if !skip_back {
            if let Some(cost) = self.cached_solve(&self.initial) {
                let total = 1 + cost;
                if total < best_cost {
                    best_action = "reset".to_string();
                }
            }
        }

        best_action
    }

    fn enumerate_actions(&self) -> Vec<String> {
        let mut actions: Vec<String> = self.templates.iter().flat_map(|t| t.expand()).collect();
        actions.push("back".to_string());
        actions.push("reset".to_string());
        if self.game.certificate_mode() {
            actions.push("yes".to_string());
            actions.push("no".to_string());
        } else {
            actions.push("done".to_string());
        }
        actions
    }

    fn render(&self, resolution: u32) -> RgbBuffer {
        self.game.render(&self.state, resolution)
    }

    fn prompt(&self) -> String {
        self.game.prompt(&self.state)
    }

    fn action_description(&self) -> String {
        self.game.action_description(&self.state)
    }

    fn state_text(&self, moves: usize) -> String {
        format_state_json(&self.game.state_fields(&self.state), moves)
    }

    fn object_boxes_text(&self) -> String {
        format_object_boxes(&self.game.object_boxes(&self.state))
    }

    fn aspect_ratio(&self) -> (u32, u32) {
        self.game.aspect_ratio()
    }

    fn optimal_len(&self) -> Option<usize> {
        self.optimal_len
    }

    fn solution_length(&self) -> Option<usize> {
        if self.game.is_solved(&self.state) {
            return Some(0);
        }
        let mut count = 0;
        let mut current = self.state.clone();
        loop {
            match self.cached_solve_action(&current) {
                Some((action, _cost)) => {
                    if let Some(next) = self.game.apply(&current, &action) {
                        current = next;
                        count += 1;
                    } else {
                        break;
                    }
                }
                None => break,
            }
            if count > 1000 {
                break;
            }
        }
        Some(count)
    }
}
