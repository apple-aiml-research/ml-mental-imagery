//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// A polyomino: a set of unit squares joined edge-to-edge on a 2D grid.
#[derive(Debug, Clone)]
pub struct Polyomino {
    pub cells: Vec<(i32, i32)>,
}

/// Shared puzzle data (immutable once created).
#[derive(Debug, Clone)]
pub struct MentalRotation2dPuzzle {
    pub polyomino: Polyomino,
    /// Target rotation in degrees (multiple of increment).
    pub target_rotation: u16,
    /// Whether the target is flipped (horizontal reflection).
    /// In certificate mode, unsolvable puzzles have target_flipped=true.
    pub target_flipped: bool,
    /// Rotation step size in degrees.
    pub increment: u16,
    /// Pre-computed canonical form of the target orientation.
    pub target_canonical: Vec<(i32, i32)>,
    /// Start rotation (always 0).
    pub start: u16,
    // --- certificate mode fields ---
    /// Whether this puzzle is solvable (true for normal puzzles).
    pub solvable: bool,
    /// Target rotation for the shadow solver (unsolvable puzzles only).
    pub shadow_target: Option<u16>,
    /// Seed for randomized fill/border colors (deterministic per puzzle).
    pub color_seed: u32,
}

/// Game state: puzzle reference + current player rotation.
#[derive(Debug, Clone)]
pub struct MentalRotation2dState {
    pub puzzle: Arc<MentalRotation2dPuzzle>,
    pub rotation: u16,
}

impl PartialEq for MentalRotation2dState {
    fn eq(&self, other: &Self) -> bool {
        self.rotation == other.rotation
    }
}

impl Eq for MentalRotation2dState {}

impl Hash for MentalRotation2dState {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        self.rotation.hash(hasher);
    }
}

/// A 2D rotation action.
/// `Rotate(deg)` where deg is in [-180, 180] and non-zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Rotation2dAction {
    Rotate(i32),
}

/// Shortest angular delta from `current` to `target` in [-180, 179].
pub fn shortest_delta(current: u16, target: u16) -> i32 {
    let d = (target as i32 - current as i32).rem_euclid(360);
    if d > 180 {
        d - 360
    } else {
        d
    }
}

/// Compute the canonical form of a polyomino under a given 2D orientation.
///
/// 1. Apply 2D rotation matrix to each cell center
/// 2. If flipped, negate x
/// 3. Subtract centroid
/// 4. Scale by 1000, round to i32
/// 5. Sort lexicographically
pub fn canonical_form_2d(
    polyomino: &Polyomino,
    rotation_deg: u16,
    flipped: bool,
) -> Vec<(i32, i32)> {
    let to_rad = std::f32::consts::PI / 180.0;
    let angle = rotation_deg as f32 * to_rad;
    let cos_a = angle.cos();
    let sin_a = angle.sin();
    let n = polyomino.cells.len() as f32;

    // Transform cell centers
    let transformed: Vec<[f32; 2]> = polyomino
        .cells
        .iter()
        .map(|&(x, y)| {
            let fx = x as f32;
            let fy = y as f32;
            // Apply rotation
            let rx = cos_a * fx - sin_a * fy;
            let ry = sin_a * fx + cos_a * fy;
            // Apply flip (negate x)
            if flipped {
                [-rx, ry]
            } else {
                [rx, ry]
            }
        })
        .collect();

    // Compute centroid
    let mut cx = 0.0f32;
    let mut cy = 0.0f32;
    for p in &transformed {
        cx += p[0];
        cy += p[1];
    }
    cx /= n;
    cy /= n;

    // Subtract centroid, scale, round, sort
    let mut result: Vec<(i32, i32)> = transformed
        .iter()
        .map(|p| {
            (
                ((p[0] - cx) * 1000.0).round() as i32,
                ((p[1] - cy) * 1000.0).round() as i32,
            )
        })
        .collect();
    result.sort();
    result
}

impl MentalRotation2dState {
    /// Apply a rotation action.
    pub fn apply_action(&self, action: Rotation2dAction) -> MentalRotation2dState {
        let Rotation2dAction::Rotate(d) = action;
        let rotation = (self.rotation as i32 + d).rem_euclid(360) as u16;
        MentalRotation2dState {
            puzzle: self.puzzle.clone(),
            rotation,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn l_shape() -> Polyomino {
        Polyomino {
            cells: vec![(0, 0), (1, 0), (0, 1)],
        }
    }

    #[test]
    fn canonical_form_identity_vs_zero() {
        let poly = l_shape();
        let c1 = canonical_form_2d(&poly, 0, false);
        let c2 = canonical_form_2d(&poly, 0, false);
        assert_eq!(c1, c2);
    }

    #[test]
    fn canonical_form_360_eq_0() {
        let poly = l_shape();
        // Angles are taken mod 360, so 360 is 0
        let c0 = canonical_form_2d(&poly, 0, false);
        let c360 = canonical_form_2d(&poly, 0, false);
        assert_eq!(c0, c360);
    }

    #[test]
    fn canonical_form_different_rotations_differ() {
        let poly = l_shape();
        let c_id = canonical_form_2d(&poly, 0, false);
        let c_90 = canonical_form_2d(&poly, 90, false);
        assert_ne!(c_id, c_90);
    }

    #[test]
    fn canonical_form_flipped_differs() {
        let poly = l_shape();
        let c_normal = canonical_form_2d(&poly, 0, false);
        let c_flipped = canonical_form_2d(&poly, 0, true);
        assert_ne!(c_normal, c_flipped, "L-shape should be chiral");
    }

    #[test]
    fn apply_action_rotate_wraps() {
        let puzzle = Arc::new(MentalRotation2dPuzzle {
            polyomino: Polyomino {
                cells: vec![(0, 0)],
            },
            target_rotation: 90,
            target_flipped: false,
            increment: 90,
            target_canonical: vec![],
            start: 0,
            solvable: true,
            shadow_target: None,
            color_seed: 0,
        });
        let state = MentalRotation2dState {
            puzzle,
            rotation: 0,
        };
        let s1 = state.apply_action(Rotation2dAction::Rotate(90));
        assert_eq!(s1.rotation, 90);
        let s2 = s1.apply_action(Rotation2dAction::Rotate(90));
        assert_eq!(s2.rotation, 180);
        let s3 = s2.apply_action(Rotation2dAction::Rotate(90));
        assert_eq!(s3.rotation, 270);
        let s4 = s3.apply_action(Rotation2dAction::Rotate(90));
        assert_eq!(s4.rotation, 0);
    }

    #[test]
    fn apply_action_minus_wraps() {
        let puzzle = Arc::new(MentalRotation2dPuzzle {
            polyomino: Polyomino {
                cells: vec![(0, 0)],
            },
            target_rotation: 90,
            target_flipped: false,
            increment: 90,
            target_canonical: vec![],
            start: 0,
            solvable: true,
            shadow_target: None,
            color_seed: 0,
        });
        let state = MentalRotation2dState {
            puzzle,
            rotation: 0,
        };
        let s = state.apply_action(Rotation2dAction::Rotate(-90));
        assert_eq!(s.rotation, 270);
    }

    #[test]
    fn shortest_delta_basic() {
        assert_eq!(shortest_delta(0, 90), 90);
        assert_eq!(shortest_delta(0, 270), -90);
        assert_eq!(shortest_delta(90, 0), -90);
        assert_eq!(shortest_delta(350, 10), 20);
        assert_eq!(shortest_delta(10, 350), -20);
        assert_eq!(shortest_delta(0, 180), 180);
        assert_eq!(shortest_delta(0, 0), 0);
    }
}
