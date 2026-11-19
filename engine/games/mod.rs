//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
// Puzzle implementations. Each game self-registers with `registry` via
// `inventory::submit!` in its own `mod.rs`.

pub mod anagram;
pub mod bloxorz;
pub mod character_recognition;
pub mod jigsaw;
pub mod mental_rotation;
pub mod mental_rotation_2d;
pub mod registry;
pub mod rush_hour;
pub mod sokoban;
pub mod tangram;
