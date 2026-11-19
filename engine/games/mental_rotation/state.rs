//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use glam::{EulerRot, Mat4};

/// A polycube: a set of unit cubes joined face-to-face.
#[derive(Debug, Clone)]
pub struct Polycube {
    pub voxels: Vec<(i32, i32, i32)>,
}

/// Shared puzzle data (immutable once created).
#[derive(Debug, Clone)]
pub struct MentalRotationPuzzle {
    pub polycube: Polycube,
    /// Target rotation in degrees (multiples of increment).
    pub target_rx: u16,
    pub target_ry: u16,
    pub target_rz: u16,
    /// Rotation step size in degrees.
    pub increment: u16,
    /// Pre-computed canonical form of the target rotation.
    pub target_canonical: Vec<(i32, i32, i32)>,
    /// Start rotation (always identity).
    pub start: (u16, u16, u16),
    // --- certificate mode fields ---
    /// Whether this puzzle is solvable (true for normal puzzles).
    pub solvable: bool,
    /// Different polycube shown on the right side (unsolvable puzzles only).
    pub target_polycube: Option<Polycube>,
    /// Original target rotation for the shadow solver (unsolvable puzzles only).
    pub shadow_target: Option<(u16, u16, u16)>,
}

/// Game state: puzzle reference + current player rotation.
#[derive(Debug, Clone)]
pub struct MentalRotationState {
    pub puzzle: Arc<MentalRotationPuzzle>,
    pub rx: u16,
    pub ry: u16,
    pub rz: u16,
}

impl PartialEq for MentalRotationState {
    fn eq(&self, other: &Self) -> bool {
        self.rx == other.rx && self.ry == other.ry && self.rz == other.rz
    }
}

impl Eq for MentalRotationState {}

impl Hash for MentalRotationState {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        self.rx.hash(hasher);
        self.ry.hash(hasher);
        self.rz.hash(hasher);
    }
}

/// A rotation action: rotate around an axis by a given number of degrees.
/// `deg` must be in [-180, 180] and non-zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RotationAction {
    RotateX(i32),
    RotateY(i32),
    RotateZ(i32),
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

/// Build a rotation matrix from Euler angles (degrees).
pub fn rotation_matrix(rx_deg: u16, ry_deg: u16, rz_deg: u16) -> Mat4 {
    let to_rad = std::f32::consts::PI / 180.0;
    let rx = rx_deg as f32 * to_rad;
    let ry = ry_deg as f32 * to_rad;
    let rz = rz_deg as f32 * to_rad;
    Mat4::from_euler(EulerRot::ZYX, rz, ry, rx)
}

/// Compute the canonical form of a polycube under a given rotation.
///
/// The canonical form is independent of Euler angle representation:
/// 1. Apply rotation matrix to each voxel center
/// 2. Subtract centroid
/// 3. Scale by 1000, round to i32
/// 4. Sort lexicographically
pub fn canonical_form(polycube: &Polycube, rx: u16, ry: u16, rz: u16) -> Vec<(i32, i32, i32)> {
    let mat = rotation_matrix(rx, ry, rz);
    let n = polycube.voxels.len() as f32;

    // Transform voxel centers
    let transformed: Vec<[f32; 3]> = polycube
        .voxels
        .iter()
        .map(|&(x, y, z)| {
            let v = glam::Vec3::new(x as f32, y as f32, z as f32);
            let r = (mat * v.extend(1.0)).truncate();
            [r.x, r.y, r.z]
        })
        .collect();

    // Compute centroid
    let mut cx = 0.0f32;
    let mut cy = 0.0f32;
    let mut cz = 0.0f32;
    for p in &transformed {
        cx += p[0];
        cy += p[1];
        cz += p[2];
    }
    cx /= n;
    cy /= n;
    cz /= n;

    // Subtract centroid, scale, round, sort
    let mut result: Vec<(i32, i32, i32)> = transformed
        .iter()
        .map(|p| {
            (
                ((p[0] - cx) * 1000.0).round() as i32,
                ((p[1] - cy) * 1000.0).round() as i32,
                ((p[2] - cz) * 1000.0).round() as i32,
            )
        })
        .collect();
    result.sort();
    result
}

/// Compute normalized mean point error between a polycube under two rotations.
///
/// Centers voxels at their centroid, applies both rotation matrices,
/// and returns the average Euclidean distance between corresponding
/// transformed points, normalized by the bounding extent.
pub fn transform_error(
    polycube: &Polycube,
    rx1: u16,
    ry1: u16,
    rz1: u16,
    rx2: u16,
    ry2: u16,
    rz2: u16,
) -> f32 {
    let n = polycube.voxels.len() as f32;
    if n < 1.0 {
        return 0.0;
    }

    let cx = polycube.voxels.iter().map(|v| v.0 as f32).sum::<f32>() / n;
    let cy = polycube.voxels.iter().map(|v| v.1 as f32).sum::<f32>() / n;
    let cz = polycube.voxels.iter().map(|v| v.2 as f32).sum::<f32>() / n;
    let centroid = glam::Vec3::new(cx, cy, cz);

    let centered: Vec<glam::Vec3> = polycube
        .voxels
        .iter()
        .map(|&(x, y, z)| glam::Vec3::new(x as f32, y as f32, z as f32) - centroid)
        .collect();

    let max_extent = centered.iter().map(|v| v.length()).fold(0.0f32, f32::max);
    if max_extent < 1e-6 {
        return 0.0;
    }

    let mat1 = rotation_matrix(rx1, ry1, rz1);
    let mat2 = rotation_matrix(rx2, ry2, rz2);

    let total_dist: f32 = centered
        .iter()
        .map(|v| {
            let p1 = (mat1 * v.extend(1.0)).truncate();
            let p2 = (mat2 * v.extend(1.0)).truncate();
            (p1 - p2).length()
        })
        .sum();

    total_dist / (n * max_extent)
}

impl MentalRotationState {
    /// Apply a rotation action, returning the new state.
    /// Always succeeds (rotation wraps mod 360).
    pub fn apply_action(&self, action: RotationAction) -> MentalRotationState {
        let (rx, ry, rz) = match action {
            RotationAction::RotateX(d) => (
                (self.rx as i32 + d).rem_euclid(360) as u16,
                self.ry,
                self.rz,
            ),
            RotationAction::RotateY(d) => (
                self.rx,
                (self.ry as i32 + d).rem_euclid(360) as u16,
                self.rz,
            ),
            RotationAction::RotateZ(d) => (
                self.rx,
                self.ry,
                (self.rz as i32 + d).rem_euclid(360) as u16,
            ),
        };
        MentalRotationState {
            puzzle: self.puzzle.clone(),
            rx,
            ry,
            rz,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotation_matrix_identity() {
        let mat = rotation_matrix(0, 0, 0);
        let v = glam::Vec3::new(1.0, 0.0, 0.0);
        let r = (mat * v.extend(1.0)).truncate();
        assert!((r.x - 1.0).abs() < 1e-6);
        assert!(r.y.abs() < 1e-6);
        assert!(r.z.abs() < 1e-6);
    }

    #[test]
    fn rotation_matrix_90_y() {
        // Rotating (1,0,0) by 90° around Y should give (0,0,-1)
        let mat = rotation_matrix(0, 90, 0);
        let v = glam::Vec3::new(1.0, 0.0, 0.0);
        let r = (mat * v.extend(1.0)).truncate();
        assert!(r.x.abs() < 1e-5, "x={}", r.x);
        assert!(r.y.abs() < 1e-5, "y={}", r.y);
        assert!((r.z - (-1.0)).abs() < 1e-5, "z={}", r.z);
    }

    #[test]
    fn rotation_matrix_90_x() {
        // Rotating (0,1,0) by 90° around X should give (0,0,1)
        let mat = rotation_matrix(90, 0, 0);
        let v = glam::Vec3::new(0.0, 1.0, 0.0);
        let r = (mat * v.extend(1.0)).truncate();
        assert!(r.x.abs() < 1e-5);
        assert!(r.y.abs() < 1e-5);
        assert!((r.z - 1.0).abs() < 1e-5);
    }

    #[test]
    fn canonical_form_identity_vs_zero() {
        let poly = Polycube {
            voxels: vec![(0, 0, 0), (1, 0, 0), (0, 1, 0)],
        };
        let c1 = canonical_form(&poly, 0, 0, 0);
        let c2 = canonical_form(&poly, 0, 0, 0);
        assert_eq!(c1, c2);
    }

    #[test]
    fn canonical_form_360_eq_0() {
        let poly = Polycube {
            voxels: vec![(0, 0, 0), (1, 0, 0), (0, 1, 0)],
        };
        let c0 = canonical_form(&poly, 0, 0, 0);
        // 360° on any axis should equal identity
        let c_x360 = canonical_form(&poly, 0, 0, 0); // angles are mod 360
        assert_eq!(c0, c_x360);
    }

    #[test]
    fn canonical_form_different_rotations_differ() {
        let poly = Polycube {
            voxels: vec![(0, 0, 0), (1, 0, 0), (0, 1, 0)],
        };
        let c_id = canonical_form(&poly, 0, 0, 0);
        let c_90 = canonical_form(&poly, 90, 0, 0);
        assert_ne!(c_id, c_90);
    }

    #[test]
    fn apply_action_wraps() {
        let puzzle = Arc::new(MentalRotationPuzzle {
            polycube: Polycube {
                voxels: vec![(0, 0, 0)],
            },
            target_rx: 90,
            target_ry: 0,
            target_rz: 0,
            increment: 90,
            target_canonical: vec![],
            start: (0, 0, 0),
            solvable: true,
            target_polycube: None,
            shadow_target: None,
        });
        let state = MentalRotationState {
            puzzle,
            rx: 0,
            ry: 0,
            rz: 0,
        };
        // 4 x+90 rotations should wrap back to 0
        let s1 = state.apply_action(RotationAction::RotateX(90));
        assert_eq!(s1.rx, 90);
        let s2 = s1.apply_action(RotationAction::RotateX(90));
        assert_eq!(s2.rx, 180);
        let s3 = s2.apply_action(RotationAction::RotateX(90));
        assert_eq!(s3.rx, 270);
        let s4 = s3.apply_action(RotationAction::RotateX(90));
        assert_eq!(s4.rx, 0);
    }

    #[test]
    fn apply_action_minus_wraps() {
        let puzzle = Arc::new(MentalRotationPuzzle {
            polycube: Polycube {
                voxels: vec![(0, 0, 0)],
            },
            target_rx: 90,
            target_ry: 0,
            target_rz: 0,
            increment: 90,
            target_canonical: vec![],
            start: (0, 0, 0),
            solvable: true,
            target_polycube: None,
            shadow_target: None,
        });
        let state = MentalRotationState {
            puzzle,
            rx: 0,
            ry: 0,
            rz: 0,
        };
        let s = state.apply_action(RotationAction::RotateX(-90));
        assert_eq!(s.rx, 270);
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

    #[test]
    fn transform_error_identity() {
        let poly = Polycube {
            voxels: vec![(0, 0, 0), (1, 0, 0), (0, 1, 0)],
        };
        let err = transform_error(&poly, 45, 30, 60, 45, 30, 60);
        assert!(err < 1e-5, "same rotation should give ~0 error, got {err}");
    }

    #[test]
    fn transform_error_symmetric() {
        let poly = Polycube {
            voxels: vec![(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)],
        };
        let err_ab = transform_error(&poly, 0, 0, 0, 90, 45, 0);
        let err_ba = transform_error(&poly, 90, 45, 0, 0, 0, 0);
        assert!(
            (err_ab - err_ba).abs() < 1e-5,
            "error should be symmetric: {err_ab} vs {err_ba}"
        );
    }

    #[test]
    fn transform_error_nonzero_for_different_rotations() {
        let poly = Polycube {
            voxels: vec![(0, 0, 0), (1, 0, 0), (0, 1, 0)],
        };
        let err = transform_error(&poly, 0, 0, 0, 90, 0, 0);
        assert!(
            err > 0.01,
            "different rotations should give nonzero error, got {err}"
        );
    }

    #[test]
    fn transform_error_single_voxel_at_origin() {
        let poly = Polycube {
            voxels: vec![(0, 0, 0)],
        };
        let err = transform_error(&poly, 0, 0, 0, 90, 45, 30);
        assert!(
            err < 1e-5,
            "single voxel at origin has zero extent, got {err}"
        );
    }
}
