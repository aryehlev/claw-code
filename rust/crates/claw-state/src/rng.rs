//! Tiny deterministic PRNG used by the scoreboard's epsilon-greedy
//! exploration. A full-fat `rand` dep is overkill for one `f64` and one
//! `usize` per turn and would make the crate harder to audit.

/// Source of pseudo-random values for selection. Trait-based so tests
/// can inject a deterministic sequence without touching real entropy.
pub trait RandomSource {
    /// Returns a value uniformly in `[0.0, 1.0)`.
    fn next_unit_float(&mut self) -> f64;

    /// Returns a value uniformly in `[0, upper)`. `upper` is guaranteed
    /// to be non-zero by the caller.
    fn next_bounded(&mut self, upper: usize) -> usize;
}

/// Splitmix64-style PRNG seeded from the system clock. Good enough for
/// exploration decisions; not suitable for anything security-sensitive.
pub struct DefaultRng {
    state: u64,
}

impl DefaultRng {
    #[must_use]
    pub fn from_seed(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    #[must_use]
    pub fn from_system_time() -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0x9E37_79B9_7F4A_7C15, |duration| {
                // Truncation is intentional: we only need entropy bits,
                // not a faithful nanosecond count.
                #[allow(clippy::cast_possible_truncation)]
                let truncated = duration.as_nanos() as u64;
                truncated
            });
        Self::from_seed(nanos ^ u64::from(std::process::id()).rotate_left(32))
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

impl RandomSource for DefaultRng {
    fn next_unit_float(&mut self) -> f64 {
        // Top 53 bits fit f64's mantissa losslessly.
        let value = self.next_u64() >> 11;
        #[allow(clippy::cast_precision_loss)]
        let as_float = value as f64 / ((1u64 << 53) as f64);
        as_float
    }

    fn next_bounded(&mut self, upper: usize) -> usize {
        debug_assert!(upper > 0, "upper must be non-zero");
        // Modulo bounds the value even on 32-bit targets where the
        // `as usize` cast narrows.
        #[allow(clippy::cast_possible_truncation)]
        let raw = self.next_u64() as usize;
        raw % upper
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_rng_returns_values_in_unit_range() {
        let mut rng = DefaultRng::from_seed(42);
        for _ in 0..500 {
            let value = rng.next_unit_float();
            assert!(
                (0.0..1.0).contains(&value),
                "unit float out of range: {value}"
            );
        }
    }

    #[test]
    fn default_rng_bounded_respects_upper_bound() {
        let mut rng = DefaultRng::from_seed(1);
        for _ in 0..500 {
            let value = rng.next_bounded(7);
            assert!(value < 7);
        }
    }

    #[test]
    fn default_rng_is_deterministic_for_fixed_seed() {
        let mut a = DefaultRng::from_seed(99);
        let mut b = DefaultRng::from_seed(99);
        for _ in 0..20 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }
}
