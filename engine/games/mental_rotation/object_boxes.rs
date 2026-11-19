//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
//! 3D Mental Rotation object boxes (Boxes CoT): the `current` shape at its
//! current rotation and the `target` shape (a different polycube in unsolvable
//! yes/no puzzles), each as the world-space extent `[x0,y0,z0,x1,y1,z1]` of its
//! rotated voxels around their centroid (as in render.rs), normalized so the
//! shape's bounding radius maps to 0..=100. The box changes as the shape turns.

use glam::Vec3;

use crate::base::object_boxes::{normalize_world_box, ObjectBox};

use super::state::{rotation_matrix, MentalRotationState};

/// Half the size of a drawn voxel (render.rs `build_polycube_mesh`).
const VOXEL_HALF_SIZE: f32 = 0.46;

fn centroid(voxels: &[(i32, i32, i32)]) -> Vec3 {
    let count = voxels.len() as f32;
    Vec3::new(
        voxels.iter().map(|voxel| voxel.0 as f32).sum::<f32>() / count,
        voxels.iter().map(|voxel| voxel.1 as f32).sum::<f32>() / count,
        voxels.iter().map(|voxel| voxel.2 as f32).sum::<f32>() / count,
    )
}

/// Bounding-sphere radius of the polycube (render.rs `bounding_radius`).
fn bounding_radius(voxels: &[(i32, i32, i32)]) -> f32 {
    let center = centroid(voxels);
    let mut max_squared_radius = 0.0f32;
    for &(x, y, z) in voxels {
        let offset_x = (x as f32 - center.x).abs() + 0.5;
        let offset_y = (y as f32 - center.y).abs() + 0.5;
        let offset_z = (z as f32 - center.z).abs() + 0.5;
        max_squared_radius =
            max_squared_radius.max(offset_x * offset_x + offset_y * offset_y + offset_z * offset_z);
    }
    max_squared_radius.sqrt()
}

fn shape_box(
    voxels: &[(i32, i32, i32)],
    rotation_x: u16,
    rotation_y: u16,
    rotation_z: u16,
) -> Vec<i32> {
    let center = centroid(voxels);
    let rotation = rotation_matrix(rotation_x, rotation_y, rotation_z);
    let mut minimum = [f32::MAX; 3];
    let mut maximum = [f32::MIN; 3];
    for &(x, y, z) in voxels {
        let voxel_center = Vec3::new(x as f32, y as f32, z as f32) - center;
        for offset_x in [-VOXEL_HALF_SIZE, VOXEL_HALF_SIZE] {
            for offset_y in [-VOXEL_HALF_SIZE, VOXEL_HALF_SIZE] {
                for offset_z in [-VOXEL_HALF_SIZE, VOXEL_HALF_SIZE] {
                    let corner = (rotation
                        * (voxel_center + Vec3::new(offset_x, offset_y, offset_z)).extend(1.0))
                    .truncate();
                    for (axis, value) in [corner.x, corner.y, corner.z].into_iter().enumerate() {
                        minimum[axis] = minimum[axis].min(value);
                        maximum[axis] = maximum[axis].max(value);
                    }
                }
            }
        }
    }
    normalize_world_box(minimum, maximum, bounding_radius(voxels)).to_vec()
}

pub fn object_boxes(state: &MentalRotationState) -> Vec<ObjectBox> {
    let puzzle = &state.puzzle;
    let target_polycube = puzzle.target_polycube.as_ref().unwrap_or(&puzzle.polycube);
    vec![
        (
            "current".to_string(),
            shape_box(&puzzle.polycube.voxels, state.rx, state.ry, state.rz),
        ),
        (
            "target".to_string(),
            shape_box(
                &target_polycube.voxels,
                puzzle.target_rx,
                puzzle.target_ry,
                puzzle.target_rz,
            ),
        ),
    ]
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::games::mental_rotation::state::{MentalRotationPuzzle, Polycube, RotationAction};

    /// A bar of two voxels along x.
    fn bar_state(
        rotation: (u16, u16, u16),
        target_rotation: (u16, u16, u16),
    ) -> MentalRotationState {
        let puzzle = Arc::new(MentalRotationPuzzle {
            polycube: Polycube {
                voxels: vec![(0, 0, 0), (1, 0, 0)],
            },
            target_rx: target_rotation.0,
            target_ry: target_rotation.1,
            target_rz: target_rotation.2,
            increment: 90,
            target_canonical: vec![],
            start: (0, 0, 0),
            solvable: true,
            target_polycube: None,
            shadow_target: None,
        });
        MentalRotationState {
            puzzle,
            rx: rotation.0,
            ry: rotation.1,
            rz: rotation.2,
        }
    }

    fn spans(coordinates: &[i32]) -> [i32; 3] {
        [
            coordinates[3] - coordinates[0],
            coordinates[4] - coordinates[1],
            coordinates[5] - coordinates[2],
        ]
    }

    #[test]
    fn boxes_are_centered_and_follow_the_orientation() {
        let object_boxes = object_boxes(&bar_state((0, 0, 0), (0, 0, 90)));
        assert_eq!(object_boxes[0].0, "current");
        assert_eq!(object_boxes[1].0, "target");
        for (_, coordinates) in &object_boxes {
            for axis in 0..3 {
                assert!(coordinates[axis] < coordinates[axis + 3], "{coordinates:?}");
                assert!(
                    ((coordinates[axis] + coordinates[axis + 3]) / 2 - 50).abs() <= 2,
                    "{coordinates:?}"
                );
            }
        }
        // The current bar lies along x; the target, turned 90 degrees about z, along y.
        let [current_x, current_y, current_z] = spans(&object_boxes[0].1);
        assert!(current_x > current_y && current_x > current_z);
        let [target_x, target_y, target_z] = spans(&object_boxes[1].1);
        assert!(target_y > target_x && target_y > target_z);
    }

    #[test]
    fn rotating_changes_the_current_box() {
        let state = bar_state((0, 0, 0), (0, 0, 0));
        let rotated = state.apply_action(RotationAction::RotateZ(90));
        assert_ne!(object_boxes(&state)[0].1, object_boxes(&rotated)[0].1);
    }
}
