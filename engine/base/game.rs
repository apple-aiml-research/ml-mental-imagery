//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
use std::hash::Hash;

use rand_chacha::ChaChaRng;

use super::object_boxes::ObjectBox;
use super::state_text::StateField;
use super::template::ActionTemplate;

/// Raw RGB image buffer.
#[derive(Clone)]
pub struct RgbBuffer {
    pub width: u32,
    pub height: u32,
    /// Row-major RGB pixels, length = width * height * 3.
    pub data: Vec<u8>,
}

impl RgbBuffer {
    /// Convert to an `image::RgbImage`, consuming self (zero-copy).
    pub fn into_image(self) -> image::RgbImage {
        image::RgbImage::from_raw(self.width, self.height, self.data)
            .expect("RgbBuffer dimensions must match data length")
    }

    /// Convert from an `image::RgbImage`, consuming it (zero-copy).
    pub fn from_image(img: image::RgbImage) -> Self {
        let (width, height) = img.dimensions();
        RgbBuffer {
            width,
            height,
            data: img.into_raw(),
        }
    }
}

/// Core game trait. Each game implements this with concrete State and Action types.
///
/// The required methods are genuinely game-specific. The defaulted solver
/// methods (`solve`, `solve_path`) use BFS and can be overridden for
/// DLX/analytical solvers.
pub trait Game: Send + Sync + 'static {
    type State: Clone + Hash + Eq + Send;
    type Action: Clone + Send;

    // --- rules ---

    /// Generate a random solvable starting state.
    /// Implementations MUST NOT panic. If quality constraints cannot be met,
    /// return `None` so the caller can fall back to `self.trivial()`.
    fn generate(&self, rng: &mut ChaChaRng) -> Option<Self::State>;

    /// Deterministic fingerprint of the *puzzle identity* of a generated state.
    ///
    /// This is distinct from the `Hash`/`Eq` impl on `Self::State` (and thus
    /// from `solver::bfs::state_hash`), which is intentionally scoped to the
    /// BFS solver cache within a single solve and omits the immutable puzzle
    /// definition (board, goal, target shape, colors, ...). `puzzle_fingerprint`
    /// captures the full generated puzzle *as presented*, so distinct puzzles
    /// can be deduplicated/counted for diversity analysis.
    ///
    /// Contract: deterministic, order-sensitive, and free of floats, pointers,
    /// and nondeterministic container iteration (sort/normalize first). Build it
    /// with `base::fingerprint::fingerprint(|h| { ... })`. See the per-game field
    /// specs in each game's implementation.
    ///
    /// Default panics: games used in diversity measurement MUST override this.
    fn puzzle_fingerprint(&self, _state: &Self::State) -> u64 {
        unimplemented!(
            "puzzle_fingerprint is not implemented for this game; \
             add a dedicated impl"
        )
    }

    /// Return a trivially solvable state. Used as a last-resort fallback
    /// when generation cannot produce a quality puzzle. Must never fail.
    fn trivial(&self) -> Self::State;

    /// List all structurally valid actions from this state.
    fn actions(&self, state: &Self::State) -> Vec<Self::Action>;

    /// Apply an action, returning the new state or None if invalid.
    fn apply(&self, state: &Self::State, action: &Self::Action) -> Option<Self::State>;

    /// Check if the state is a goal state.
    fn is_solved(&self, state: &Self::State) -> bool;

    // --- presentation ---

    /// Render the state to an RGB image at the given resolution.
    fn render(&self, state: &Self::State, resolution: u32) -> RgbBuffer;

    /// Convert an action to its canonical string representation.
    fn action_to_str(&self, action: &Self::Action) -> String;

    /// Parse an action string in the context of a state.
    fn parse_action(&self, state: &Self::State, s: &str) -> Option<Self::Action>;

    /// Return templates describing the parametric action space for this state.
    fn action_templates(&self, state: &Self::State) -> Vec<ActionTemplate>;

    /// Natural language prompt describing the task.
    fn prompt(&self, state: &Self::State) -> String;

    /// Human-readable description of the action space.
    fn action_description(&self, state: &Self::State) -> String;

    /// Compact state description for State CoT:
    /// ordered JSON fields with JSON-encoded values. The episode appends the
    /// `moves` field and builds the object (see `base::state_text`).
    fn state_fields(&self, state: &Self::State) -> Vec<StateField>;

    /// Task-relevant objects and where they are drawn in `render`, for Boxes
    /// CoT: names in a fixed order with normalized coordinates (see
    /// `base::object_boxes`).
    fn object_boxes(&self, state: &Self::State) -> Vec<ObjectBox>;

    // --- solver (default = BFS, override for DLX/analytical) ---

    /// Find the optimal first action and cost-to-goal. Default uses BFS.
    fn solve(&self, state: &Self::State, budget: usize) -> Option<(Self::Action, usize)>
    where
        Self: Sized,
    {
        crate::solver::bfs::bfs_solve(self, state, budget)
            .map(|r| (r.first_action().clone(), r.cost()))
    }

    // --- certificate mode (default: disabled) ---

    /// Whether this game instance is in certificate mode (yes/no instead of done).
    fn certificate_mode(&self) -> bool {
        false
    }

    /// Whether the current puzzle is solvable. Only meaningful in certificate mode.
    fn is_solvable(&self, _state: &Self::State) -> bool {
        true
    }

    /// Shadow solve cost for unsolvable puzzles. Returns Some(0) when the shadow
    /// solver has guided the policy close enough that "no" is the expert action.
    fn shadow_solve_cost(&self, _state: &Self::State) -> Option<usize> {
        None
    }

    /// Returns the aspect ratio as (width, height).
    /// Default: (1, 1) for square. Override for non-square games.
    fn aspect_ratio(&self) -> (u32, u32) {
        (1, 1)
    }

    /// Return the full optimal path for warm-caching. Default calls bfs_solve
    /// once. DLX/analytical games override to build the path via their solver.
    fn solve_path(&self, state: &Self::State, budget: usize) -> Option<Vec<(Self::Action, usize)>>
    where
        Self: Sized,
    {
        crate::solver::bfs::bfs_solve(self, state, budget).map(|r| {
            let n = r.actions.len();
            r.actions
                .into_iter()
                .enumerate()
                .map(|(i, a)| (a, n - i))
                .collect()
        })
    }
}

/// Compute (width, height) from aspect ratio and short edge resolution.
pub fn compute_render_size(aspect_ratio: (u32, u32), short_edge_res: u32) -> (u32, u32) {
    let (ratio_w, ratio_h) = aspect_ratio;
    if ratio_w >= ratio_h {
        // Landscape or square: height is the short edge
        (short_edge_res * ratio_w / ratio_h, short_edge_res)
    } else {
        // Portrait: width is the short edge
        (short_edge_res, short_edge_res * ratio_h / ratio_w)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_render_size_square() {
        assert_eq!(compute_render_size((1, 1), 256), (256, 256));
    }

    #[test]
    fn test_compute_render_size_landscape_2_1() {
        assert_eq!(compute_render_size((2, 1), 256), (512, 256));
    }

    #[test]
    fn test_compute_render_size_portrait_1_2() {
        assert_eq!(compute_render_size((1, 2), 256), (256, 512));
    }

    #[test]
    fn test_compute_render_size_landscape_3_2() {
        assert_eq!(compute_render_size((3, 2), 128), (192, 128));
    }

    #[test]
    fn test_compute_render_size_small_res() {
        assert_eq!(compute_render_size((2, 1), 64), (128, 64));
    }

    #[test]
    fn test_compute_render_size_all_square_resolutions() {
        for res in [64, 128, 256, 512] {
            assert_eq!(
                compute_render_size((1, 1), res),
                (res, res),
                "failed for resolution {res}"
            );
        }
    }
}
