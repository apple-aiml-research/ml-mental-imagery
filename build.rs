//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
//! The puzzle engine embeds fonts (`include_bytes!` in engine/base/render.rs and
//! engine/games/character_recognition/gen.rs) that are not distributed with this repository.
//! Stop with instructions when any of them has not been downloaded yet.

use std::fs;
use std::path::{Path, PathBuf};

fn rust_sources(directory: &Path, files: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(directory)
        .expect("engine/ is readable")
        .flatten()
    {
        let path = entry.path();
        if path.is_dir() {
            rust_sources(&path, files);
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            files.push(path);
        }
    }
}

fn main() {
    let root = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("set by cargo"));
    println!("cargo:rerun-if-changed=assets/fonts");
    let mut sources = Vec::new();
    rust_sources(&root.join("engine"), &mut sources);
    let mut missing = Vec::new();
    for source in &sources {
        let text = fs::read_to_string(source).expect("engine sources are UTF-8");
        for (start, _) in text.match_indices("assets/fonts/") {
            let rest = &text[start..];
            if let Some(end) = rest.find(".ttf") {
                let font = &rest[..end + 4];
                if !root.join(font).exists() && !missing.contains(&font.to_string()) {
                    missing.push(font.to_string());
                }
            }
        }
    }
    if !missing.is_empty() {
        panic!(
            "\n\n{} fonts the engine embeds are missing (e.g. {}). They are not distributed with this \
             repository; download them first:\n\n    uv run --no-project --with fonttools==4.62.1 python \
             scripts/download_fonts.py\n\n",
            missing.len(),
            missing[0]
        );
    }
}
