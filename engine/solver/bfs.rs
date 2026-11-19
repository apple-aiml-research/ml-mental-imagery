//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};

use crate::base::game::Game;

/// Hash a state to a u64 using the default hasher. Public utility for
/// downstream caching (e.g. the `GameEpisode` solver cache).
pub fn state_hash<S: Hash>(state: &S) -> u64 {
    let mut hasher = std::hash::DefaultHasher::new();
    state.hash(&mut hasher);
    hasher.finish()
}

/// Result of a BFS solve: the full optimal action sequence from the
/// initial state to the goal.
#[derive(Debug, Clone)]
pub struct SolveResult<A> {
    pub actions: Vec<A>,
}

impl<A> SolveResult<A> {
    /// The first action on the optimal path.
    pub fn first_action(&self) -> &A {
        &self.actions[0]
    }

    /// Cost (number of steps) to reach the goal.
    pub fn cost(&self) -> usize {
        self.actions.len()
    }
}

/// Generic BFS solver for any `Game`.
///
/// Returns the full optimal action path, or `None` if the goal is
/// unreachable within `max_states` explored states.
pub fn bfs_solve<G: Game>(
    game: &G,
    state: &G::State,
    max_states: usize,
) -> Option<SolveResult<G::Action>> {
    if game.is_solved(state) {
        return None;
    }

    // visited: state -> Option<(parent_state, action_to_reach_state)>
    // Initial state has None (no parent).
    let mut visited: HashMap<G::State, Option<(G::State, G::Action)>> = HashMap::new();
    let mut queue: VecDeque<G::State> = VecDeque::new();

    visited.insert(state.clone(), None);

    // Seed phase: apply each action from the initial state.
    for action in game.actions(state) {
        let next_opt = game.apply(state, &action);

        if let Some(next) = next_opt {
            if visited.contains_key(&next) {
                continue;
            }

            let solved = game.is_solved(&next);

            if solved {
                return Some(SolveResult {
                    actions: vec![action],
                });
            }
            visited.insert(next.clone(), Some((state.clone(), action)));
            queue.push_back(next);
        }
    }

    bfs_loop(game, &mut visited, &mut queue, max_states)
}

/// Reconstruct the action path from the visited map by tracing parent
/// pointers back to the initial state (which has `None` as its entry).
fn reconstruct_path<S: Clone + Eq + Hash, A: Clone>(
    visited: &HashMap<S, Option<(S, A)>>,
    terminal_parent: &S,
    terminal_action: A,
) -> Vec<A> {
    let mut path = vec![terminal_action];
    let mut current = terminal_parent.clone();
    while let Some(Some((parent, action))) = visited.get(&current) {
        path.push(action.clone());
        current = parent.clone();
    }
    path.reverse();
    path
}

/// BFS from the seeded queue.
fn bfs_loop<G: Game>(
    game: &G,
    visited: &mut HashMap<G::State, Option<(G::State, G::Action)>>,
    queue: &mut VecDeque<G::State>,
    max_states: usize,
) -> Option<SolveResult<G::Action>> {
    let mut states_explored: usize = 0;

    while let Some(current) = queue.pop_front() {
        states_explored += 1;
        if states_explored >= max_states {
            return None;
        }

        let actions = game.actions(&current);

        for action in actions {
            let next_opt = game.apply(&current, &action);

            if let Some(next) = next_opt {
                if visited.contains_key(&next) {
                    continue;
                }

                let solved = game.is_solved(&next);

                if solved {
                    let actions = reconstruct_path(visited, &current, action);
                    return Some(SolveResult { actions });
                }
                visited.insert(next.clone(), Some((current.clone(), action)));
                queue.push_back(next);
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::game::RgbBuffer;
    use crate::base::template::ActionTemplate;
    use rand_chacha::ChaChaRng;

    /// Tiny game with a controlled state graph:
    ///
    /// ```text
    ///   0 --[0]--> 1 --[0]--> 3 (goal)
    ///   0 --[1]--> 2 --[0]--> 4 --[0]--> 3 (goal)
    ///                  --[1]--> 5 (dead end)
    /// ```
    ///
    /// Optimal from 0: action 0, cost 2 (path: 0→1→3).
    struct TinyGame;

    impl Game for TinyGame {
        type State = u8;
        type Action = u8;

        fn generate(&self, _rng: &mut ChaChaRng) -> Option<u8> {
            Some(0)
        }

        fn trivial(&self) -> u8 {
            0
        }

        fn actions(&self, state: &u8) -> Vec<u8> {
            match *state {
                0 => vec![0, 1],
                1 => vec![0],
                2 => vec![0, 1],
                4 => vec![0],
                _ => vec![],
            }
        }

        fn apply(&self, state: &u8, action: &u8) -> Option<u8> {
            match (*state, *action) {
                (0, 0) => Some(1),
                (0, 1) => Some(2),
                (1, 0) => Some(3),
                (2, 0) => Some(4),
                (2, 1) => Some(5),
                (4, 0) => Some(3),
                _ => None,
            }
        }

        fn is_solved(&self, state: &u8) -> bool {
            *state == 3
        }

        fn render(&self, _state: &u8, _resolution: u32) -> RgbBuffer {
            RgbBuffer {
                width: 1,
                height: 1,
                data: vec![0, 0, 0],
            }
        }

        fn action_to_str(&self, action: &u8) -> String {
            action.to_string()
        }

        fn parse_action(&self, _state: &u8, s: &str) -> Option<u8> {
            s.parse().ok()
        }

        fn action_templates(&self, _state: &u8) -> Vec<ActionTemplate> {
            vec![]
        }

        fn prompt(&self, _state: &u8) -> String {
            String::new()
        }

        fn action_description(&self, _state: &u8) -> String {
            String::new()
        }

        fn state_fields(&self, state: &u8) -> Vec<crate::base::state_text::StateField> {
            vec![("s", state.to_string())]
        }

        fn object_boxes(&self, state: &u8) -> Vec<crate::base::object_boxes::ObjectBox> {
            vec![("s".to_string(), vec![*state as i32])]
        }
    }

    #[test]
    fn standard_bfs_finds_optimal() {
        let r = bfs_solve(&TinyGame, &0, 1000).unwrap();
        assert_eq!(*r.first_action(), 0);
        assert_eq!(r.cost(), 2);
        assert_eq!(r.actions, vec![0, 0]); // 0→1→3
    }

    #[test]
    fn already_solved_returns_none() {
        assert!(bfs_solve(&TinyGame, &3, 1000).is_none());
    }

    #[test]
    fn budget_exceeded_returns_none() {
        assert!(bfs_solve(&TinyGame, &0, 1).is_none());
    }

    #[test]
    fn dead_end_returns_none() {
        assert!(bfs_solve(&TinyGame, &5, 1000).is_none());
    }

    #[test]
    fn state_hash_deterministic() {
        assert_eq!(state_hash(&42u64), state_hash(&42u64));
        assert_ne!(state_hash(&42u64), state_hash(&43u64));
    }

    #[test]
    fn full_path_is_valid() {
        let r = bfs_solve(&TinyGame, &0, 1000).unwrap();
        // Replay the path and verify it reaches the goal.
        let mut state: u8 = 0;
        for action in &r.actions {
            state = TinyGame.apply(&state, action).unwrap();
        }
        assert!(TinyGame.is_solved(&state));
    }
}
