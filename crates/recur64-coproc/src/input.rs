//! The fixed coprocessor input buffer: canonical observation + canonical legal
//! candidates + the bounded-tactics search depth.

use crate::{
    CoprocError, HEADER_BYTES, INPUT_LEN, INPUT_VERSION, MAX_LEGAL, OBS_BYTES, OBS_OFFSET,
};

/// One canonical legal candidate as stored in the input buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoprocMove {
    pub from: u8,
    pub to: u8,
    /// Promotion code: `0` = none, `1..=4` = knight/bishop/rook/queen.
    pub promo: u8,
}

/// Read-only view over a valid input buffer.
#[derive(Debug, Clone, Copy)]
pub struct CoprocInput<'a> {
    bytes: &'a [u8],
}

impl<'a> CoprocInput<'a> {
    /// Validate the fixed layout and wrap the buffer.
    pub fn new(bytes: &'a [u8]) -> Result<Self, CoprocError> {
        if bytes.len() != INPUT_LEN {
            return Err(CoprocError::BadInputLength(bytes.len()));
        }
        let version = bytes[0];
        if version != INPUT_VERSION {
            return Err(CoprocError::BadVersion(version));
        }
        let this = Self { bytes };
        let n = this.n_legal();
        if n > MAX_LEGAL {
            return Err(CoprocError::TooManyLegal(n));
        }
        for (i, mv) in this.moves().enumerate() {
            if mv.from >= 64 || mv.to >= 64 || mv.promo > 4 {
                return Err(CoprocError::BadMove {
                    index: i,
                    from: mv.from,
                    to: mv.to,
                    promo: mv.promo,
                });
            }
        }
        Ok(this)
    }

    /// Bounded exact-tactics search depth requested for this position
    /// (`0` none, `1` mate-in-1, `2` mate-in-2).
    pub fn mate_search_depth(&self) -> u8 {
        self.bytes[1]
    }

    /// Number of legal candidates stored.
    pub fn n_legal(&self) -> usize {
        u32::from_le_bytes([self.bytes[4], self.bytes[5], self.bytes[6], self.bytes[7]]) as usize
    }

    /// The stored legal candidates.
    pub fn moves(&self) -> impl Iterator<Item = CoprocMove> + '_ {
        let n = self.n_legal().min(MAX_LEGAL);
        (0..n).map(move |i| {
            let o = HEADER_BYTES + i * 4;
            CoprocMove {
                from: self.bytes[o],
                to: self.bytes[o + 1],
                promo: self.bytes[o + 2],
            }
        })
    }

    /// Raw observation bytes (`OBS_LEN` little-endian f32).
    pub fn observation_bytes(&self) -> &'a [u8] {
        &self.bytes[OBS_OFFSET..OBS_OFFSET + OBS_BYTES]
    }

    /// One observation feature for one canonical square.
    pub fn obs_feature(&self, square: usize, feature: usize) -> f32 {
        let o = OBS_OFFSET + (square * crate::FEATURES_PER_SQUARE + feature) * 4;
        f32::from_le_bytes([
            self.bytes[o],
            self.bytes[o + 1],
            self.bytes[o + 2],
            self.bytes[o + 3],
        ])
    }
}

/// Build an input buffer. `observation` is Observation V1 (`[square][feature]`),
/// `moves` the canonical legal candidates, `mate_search_depth` the bounded
/// exact-tactics request.
pub fn write_input(
    observation: &[f32],
    moves: &[CoprocMove],
    mate_search_depth: u8,
) -> Result<Vec<u8>, CoprocError> {
    if observation.len() != crate::OBS_LEN {
        return Err(CoprocError::BadInputLength(observation.len()));
    }
    if moves.len() > MAX_LEGAL {
        return Err(CoprocError::TooManyLegal(moves.len()));
    }
    let mut out = vec![0u8; INPUT_LEN];
    out[0] = INPUT_VERSION;
    out[1] = mate_search_depth.min(2);
    out[2] = 0;
    out[3] = 0;
    out[4..8].copy_from_slice(&(moves.len() as u32).to_le_bytes());
    for (i, mv) in moves.iter().enumerate() {
        if mv.from >= 64 || mv.to >= 64 || mv.promo > 4 {
            return Err(CoprocError::BadMove {
                index: i,
                from: mv.from,
                to: mv.to,
                promo: mv.promo,
            });
        }
        let o = HEADER_BYTES + i * 4;
        out[o] = mv.from;
        out[o + 1] = mv.to;
        out[o + 2] = mv.promo;
        out[o + 3] = 0;
    }
    for (i, v) in observation.iter().enumerate() {
        out[OBS_OFFSET + i * 4..OBS_OFFSET + i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_constants_are_consistent() {
        assert_eq!(HEADER_BYTES + MAX_LEGAL * 4, OBS_OFFSET);
        assert_eq!(OBS_OFFSET + OBS_BYTES, INPUT_LEN);
        // 8 + 1024 + 30464
        assert_eq!(INPUT_LEN, 31496);
    }

    #[test]
    fn round_trip() {
        let mut obs = vec![0.0f32; crate::OBS_LEN];
        obs[0] = 1.0;
        obs[crate::OBS_LEN - 1] = 0.5;
        let moves = vec![
            CoprocMove {
                from: 12,
                to: 28,
                promo: 0,
            },
            CoprocMove {
                from: 11,
                to: 27,
                promo: 4,
            },
        ];
        let bytes = write_input(&obs, &moves, 2).unwrap();
        let inp = CoprocInput::new(&bytes).unwrap();
        assert_eq!(inp.n_legal(), 2);
        assert_eq!(inp.mate_search_depth(), 2);
        assert_eq!(inp.obs_feature(0, 0), 1.0);
        assert_eq!(inp.obs_feature(63, crate::FEATURES_PER_SQUARE - 1), 0.5);
        let got: Vec<_> = inp.moves().collect();
        assert_eq!(got, moves);
    }

    #[test]
    fn rejects_bad_inputs() {
        assert!(matches!(
            CoprocInput::new(&[0u8; 4]),
            Err(CoprocError::BadInputLength(4))
        ));
        let mut bytes = vec![0u8; INPUT_LEN];
        bytes[0] = 9;
        assert!(matches!(
            CoprocInput::new(&bytes),
            Err(CoprocError::BadVersion(9))
        ));
        let bytes = write_input(&vec![0.0; crate::OBS_LEN], &[], 0).unwrap();
        let mut bad = bytes.clone();
        bad[4..8].copy_from_slice(&(MAX_LEGAL as u32 + 1).to_le_bytes());
        assert!(matches!(
            CoprocInput::new(&bad),
            Err(CoprocError::TooManyLegal(_))
        ));
        let moves = vec![CoprocMove {
            from: 0,
            to: 64,
            promo: 0,
        }];
        assert!(write_input(&vec![0.0; crate::OBS_LEN], &moves, 0).is_err());
    }
}
