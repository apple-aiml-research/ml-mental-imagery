//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
//! Rush Hour object boxes (Boxes CoT): each vehicle's centre cell
//! `[column,row]`, the rounded centre of its cells, which always lies on the
//! vehicle. Vehicles are named by their colour in render.rs (the target car is
//! `red`); past 12 vehicles colours repeat and get a `_1`, `_2`, ... suffix.

use std::collections::HashMap;

use crate::base::object_boxes::ObjectBox;

use super::state::{Orientation, RushHourState};

/// Colour name of a vehicle, in the order of `render.rs::VEHICLE_COLORS`.
fn color_name(vehicle_id: char) -> &'static str {
    const COLOR_NAMES: [&str; 12] = [
        "red", "blue", "green", "orange", "purple", "teal", "yellow", "pink", "brown", "lime",
        "lavender", "peach",
    ];
    COLOR_NAMES[(vehicle_id as u8).wrapping_sub(b'A') as usize % COLOR_NAMES.len()]
}

pub fn object_boxes(state: &RushHourState) -> Vec<ObjectBox> {
    let mut repeats_by_color: HashMap<&str, usize> = HashMap::new();
    let mut object_boxes = Vec::with_capacity(state.config.vehicles.len());
    for (vehicle_index, vehicle) in state.config.vehicles.iter().enumerate() {
        let center_along =
            state.positions[vehicle_index] as f64 + (vehicle.length as f64 - 1.0) / 2.0;
        let (center_column, center_row) = match vehicle.orientation {
            Orientation::Horizontal => (center_along, vehicle.fixed_axis as f64),
            Orientation::Vertical => (vehicle.fixed_axis as f64, center_along),
        };
        let color = color_name(vehicle.id);
        let name = match repeats_by_color.get_mut(color) {
            Some(repeats) => {
                *repeats += 1;
                format!("{color}_{repeats}")
            }
            None => {
                repeats_by_color.insert(color, 0);
                color.to_string()
            }
        };
        object_boxes.push((
            name,
            vec![center_column.round() as i32, center_row.round() as i32],
        ));
    }
    object_boxes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::games::rush_hour::state::from_ascii;

    const BOARD: &str = "..BCCC\n..B...\nAAB..D\n.EE..D\n......\n......";

    #[test]
    fn names_vehicles_by_color_with_their_center_cells() {
        // Centres round half away from zero: A covers columns 0-1 -> 0.5 -> 1.
        assert_eq!(
            object_boxes(&from_ascii(BOARD)),
            vec![
                ("red".to_string(), vec![1, 2]),
                ("blue".to_string(), vec![2, 1]),
                ("green".to_string(), vec![4, 0]),
                ("orange".to_string(), vec![5, 3]),
                ("purple".to_string(), vec![2, 3]),
            ]
        );
    }

    #[test]
    fn center_cell_lies_on_its_vehicle() {
        let state = from_ascii(BOARD);
        let object_boxes = object_boxes(&state);
        for (vehicle_index, vehicle) in state.config.vehicles.iter().enumerate() {
            let position = state.positions[vehicle_index] as i32;
            let cells: Vec<(i32, i32)> = (0..vehicle.length as i32)
                .map(|offset| match vehicle.orientation {
                    Orientation::Horizontal => (position + offset, vehicle.fixed_axis as i32),
                    Orientation::Vertical => (vehicle.fixed_axis as i32, position + offset),
                })
                .collect();
            let center = &object_boxes[vehicle_index].1;
            assert!(
                cells.contains(&(center[0], center[1])),
                "centre {center:?} is not on vehicle {}",
                vehicle.id
            );
        }
    }
}
