//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
//! Stable, deterministic hashing for puzzle-identity fingerprints.
//!
//! This is deliberately NOT `std::hash::DefaultHasher`: `DefaultHasher`
//! (SipHash) is only guaranteed stable within a single Rust build, and its
//! integer `write_*` methods use native-endian bytes. `FnvHasher` is FNV-1a
//! (64-bit) with fixed little-endian integer encodings, so a given puzzle
//! yields the same fingerprint across platforms, Rust versions, and runs —
//! which is what puzzle-diversity counting needs.
//!
//! Games build a fingerprint by feeding puzzle-identity fields into this
//! hasher in a fixed order via `Game::puzzle_fingerprint`.

use std::hash::Hasher;

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// FNV-1a 64-bit hasher with platform-independent integer encodings.
pub struct FnvHasher(u64);

impl Default for FnvHasher {
    fn default() -> Self {
        FnvHasher(FNV_OFFSET)
    }
}

impl Hasher for FnvHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 ^= b as u64;
            self.0 = self.0.wrapping_mul(FNV_PRIME);
        }
    }

    // Override integer writes to use fixed little-endian bytes so fingerprints
    // are identical on 32- and 64-bit hosts (the std defaults use to_ne_bytes).
    fn write_u8(&mut self, i: u8) {
        self.write(&[i]);
    }
    fn write_u16(&mut self, i: u16) {
        self.write(&i.to_le_bytes());
    }
    fn write_u32(&mut self, i: u32) {
        self.write(&i.to_le_bytes());
    }
    fn write_u64(&mut self, i: u64) {
        self.write(&i.to_le_bytes());
    }
    fn write_usize(&mut self, i: usize) {
        self.write(&(i as u64).to_le_bytes());
    }
}

/// Version byte mixed into every fingerprint. Bump if any game's fingerprint
/// scheme changes, so previously-persisted fingerprints are not silently
/// compared against a new scheme.
pub const FINGERPRINT_VERSION: u8 = 1;

/// Convenience: run `build` against a fresh `FnvHasher` (pre-seeded with the
/// version byte) and return the finished 64-bit fingerprint.
pub fn fingerprint(build: impl FnOnce(&mut FnvHasher)) -> u64 {
    let mut h = FnvHasher::default();
    h.write_u8(FINGERPRINT_VERSION);
    build(&mut h);
    h.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::hash::Hasher;

    #[test]
    fn deterministic() {
        let a = fingerprint(|h| {
            h.write_u64(42);
            h.write_u8(7);
        });
        let b = fingerprint(|h| {
            h.write_u64(42);
            h.write_u8(7);
        });
        assert_eq!(a, b);
    }

    #[test]
    fn order_sensitive() {
        let a = fingerprint(|h| {
            h.write_u8(1);
            h.write_u8(2);
        });
        let b = fingerprint(|h| {
            h.write_u8(2);
            h.write_u8(1);
        });
        assert_ne!(a, b);
    }

    #[test]
    fn nonzero_and_distinct() {
        let a = fingerprint(|h| h.write_u64(0));
        let b = fingerprint(|h| h.write_u64(1));
        assert_ne!(a, b);
    }
}
