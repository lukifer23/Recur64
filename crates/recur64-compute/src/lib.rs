//! Deterministic chess coprocessor providers (X15 / Chimera).
//!
//! Three interchangeable providers:
//!
//! * [`ComputeProviderKind::None`] — the pathway is switched off. No host work
//!   is done at all; the model sees an all-zero bank.
//! * [`ComputeProviderKind::NativeV1`] — calls [`recur64_coproc::compute`]
//!   directly on the host.
//! * [`ComputeProviderKind::WasmV1`] — calls the same code compiled to
//!   `wasm32-unknown-unknown` and embedded in `assets/compute_bank_v1.wasm`,
//!   run by the pure-Rust `wasmi` interpreter.
//!
//! `NativeV1` and `WasmV1` run literally the same source, so they must produce
//! identical bytes; `tests::native_and_wasm_agree_byte_exactly` asserts it over
//! random legal positions including every special-move case. The provider label
//! is therefore the only difference between them, and the X15 identity records
//! the *semantic* version (`compute_bank_v1`) separately from the execution
//! label.

use std::sync::Mutex;

use recur64_coproc::{
    COMPUTE_BANK_VERSION, ComputeProviderKind, CoprocError, INPUT_LEN, OUTPUT_LEN,
    compute::compute_bank, input::CoprocMove,
};
use recur64_core::{ActionId, ObservationV1};

pub mod artifact;

pub use recur64_coproc::ComputeProviderKind as ProviderKind;
pub use recur64_coproc::world::{WorldHorizon, WorldStats};

/// A provider failure. Never a silent fallback to zeros.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ComputeError {
    /// The caller passed mismatched batch/output shapes.
    BufferShape(String),
    /// The coprocessor refused the position.
    Coproc(CoprocError),
    /// The WASM guest could not be instantiated or called.
    Wasm(String),
}

impl std::fmt::Display for ComputeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ComputeError::BufferShape(m) => write!(f, "compute buffer shape: {m}"),
            ComputeError::Coproc(e) => write!(f, "coprocessor: {e}"),
            ComputeError::Wasm(m) => write!(f, "wasm provider: {m}"),
        }
    }
}

impl std::error::Error for ComputeError {}

impl From<CoprocError> for ComputeError {
    fn from(e: CoprocError) -> Self {
        ComputeError::Coproc(e)
    }
}

/// Build one fixed-size coprocessor input buffer from the inference contract:
/// the canonical observation plus the canonical legal candidate list.
pub fn encode_input(
    observation: &ObservationV1,
    legal: &[ActionId],
    mate_search_depth: u8,
) -> anyhow::Result<Vec<u8>> {
    let moves: Vec<CoprocMove> = legal
        .iter()
        .map(|a| {
            let (from, to, promo) = a.decode();
            CoprocMove {
                from: from as usize as u8,
                to: to as usize as u8,
                promo: promo.code(),
            }
        })
        .collect();
    recur64_coproc::input::write_input(observation.as_slice(), &moves, mate_search_depth)
        .map_err(|e| anyhow::anyhow!("{e}"))
}

/// A deterministic compute provider.
pub trait ComputeProvider: Send + Sync {
    /// Which provider this is.
    fn kind(&self) -> ComputeProviderKind;
    /// The semantic feature-layout version (independent of the execution
    /// backend).
    fn semantic_version(&self) -> &'static str {
        COMPUTE_BANK_VERSION
    }
    /// Compute one bank per input into the caller-owned `out` buffers, which
    /// must be exactly `inputs.len()` fixed-size buffers.
    fn compute_batch(&self, inputs: &[Vec<u8>], out: &mut [Vec<u8>]) -> Result<(), ComputeError>;
}

/// The "pathway off" provider: no work, empty results.
pub struct NoneProvider;

impl ComputeProvider for NoneProvider {
    fn kind(&self) -> ComputeProviderKind {
        ComputeProviderKind::None
    }
    fn compute_batch(&self, _inputs: &[Vec<u8>], _out: &mut [Vec<u8>]) -> Result<(), ComputeError> {
        Ok(())
    }
}

/// The native provider.
pub struct NativeProvider;

impl ComputeProvider for NativeProvider {
    fn kind(&self) -> ComputeProviderKind {
        ComputeProviderKind::NativeV1
    }
    fn compute_batch(&self, inputs: &[Vec<u8>], out: &mut [Vec<u8>]) -> Result<(), ComputeError> {
        if inputs.len() != out.len() {
            return Err(ComputeError::BufferShape(format!(
                "{} inputs but {} outputs",
                inputs.len(),
                out.len()
            )));
        }
        for (i, (input, slot)) in inputs.iter().zip(out.iter_mut()).enumerate() {
            if slot.len() != OUTPUT_LEN {
                return Err(ComputeError::BufferShape(format!(
                    "output {i} is {} bytes, expected {OUTPUT_LEN}",
                    slot.len()
                )));
            }
            compute_bank(input, slot)?;
        }
        Ok(())
    }
}

/// The WebAssembly provider.
pub struct WasmProvider {
    inner: Mutex<WasmInstance>,
}

struct WasmInstance {
    store: wasmi::Store<()>,
    memory: wasmi::Memory,
    compute: wasmi::TypedFunc<(i32, i32, i32, i32), i32>,
    /// Guest scratch buffers, allocated once and reused for every batch.
    in_ptr: i32,
    out_ptr: i32,
}

impl WasmProvider {
    /// Instantiate the embedded module.
    pub fn new() -> anyhow::Result<Self> {
        let bytes = artifact::WASM_ARTIFACT;
        let engine = wasmi::Engine::default();
        let module = wasmi::Module::new(&engine, bytes)
            .map_err(|e| anyhow::anyhow!("instantiate {}: {e}", artifact::WASM_SHA256))?;
        let mut store = wasmi::Store::new(&engine, ());
        let linker = wasmi::Linker::new(&engine);
        let instance = linker
            .instantiate(&mut store, &module)
            .map_err(|e| anyhow::anyhow!("link: {e}"))?
            .start(&mut store)
            .map_err(|e| anyhow::anyhow!("start: {e}"))?;
        let memory = instance
            .get_memory(&store, "memory")
            .ok_or_else(|| anyhow::anyhow!("the guest exports no `memory`"))?;
        let alloc = instance
            .get_typed_func::<i32, i32>(&store, "coproc_alloc")
            .map_err(|e| anyhow::anyhow!("coproc_alloc: {e}"))?;
        let compute = instance
            .get_typed_func::<(i32, i32, i32, i32), i32>(&store, "coproc_compute")
            .map_err(|e| anyhow::anyhow!("coproc_compute: {e}"))?;
        let in_ptr = alloc
            .call(&mut store, INPUT_LEN as i32)
            .map_err(|e| anyhow::anyhow!("alloc input: {e}"))?;
        let out_ptr = alloc
            .call(&mut store, OUTPUT_LEN as i32)
            .map_err(|e| anyhow::anyhow!("alloc output: {e}"))?;
        Ok(Self {
            inner: Mutex::new(WasmInstance {
                store,
                memory,
                compute,
                in_ptr,
                out_ptr,
            }),
        })
    }
}

impl ComputeProvider for WasmProvider {
    fn kind(&self) -> ComputeProviderKind {
        ComputeProviderKind::WasmV1
    }
    fn compute_batch(&self, inputs: &[Vec<u8>], out: &mut [Vec<u8>]) -> Result<(), ComputeError> {
        if inputs.len() != out.len() {
            return Err(ComputeError::BufferShape(format!(
                "{} inputs but {} outputs",
                inputs.len(),
                out.len()
            )));
        }
        let mut guard = self.inner.lock().map_err(|_| {
            ComputeError::Wasm("the wasm store mutex was poisoned by an earlier panic".into())
        })?;
        let inst = &mut *guard;
        for (i, (input, slot)) in inputs.iter().zip(out.iter_mut()).enumerate() {
            if input.len() != INPUT_LEN {
                return Err(ComputeError::BufferShape(format!(
                    "input {i} is {} bytes, expected {INPUT_LEN}",
                    input.len()
                )));
            }
            if slot.len() != OUTPUT_LEN {
                return Err(ComputeError::BufferShape(format!(
                    "output {i} is {} bytes, expected {OUTPUT_LEN}",
                    slot.len()
                )));
            }
            inst.memory
                .write(&mut inst.store, inst.in_ptr as usize, input)
                .map_err(|e| ComputeError::Wasm(format!("write input: {e}")))?;
            let status = inst
                .compute
                .call(
                    &mut inst.store,
                    (
                        inst.in_ptr,
                        INPUT_LEN as i32,
                        inst.out_ptr,
                        OUTPUT_LEN as i32,
                    ),
                )
                .map_err(|e| ComputeError::Wasm(format!("call: {e}")))?;
            if status != 0 {
                return Err(ComputeError::Wasm(format!(
                    "guest returned status {status} for position {i}"
                )));
            }
            inst.memory
                .read(&inst.store, inst.out_ptr as usize, slot)
                .map_err(|e| ComputeError::Wasm(format!("read output: {e}")))?;
        }
        Ok(())
    }
}

/// Build the provider for a config label. `None` is cheap; the WASM provider
/// instantiates the module eagerly so a broken artifact fails at startup
/// rather than mid-run.
pub fn provider_for(kind: ComputeProviderKind) -> anyhow::Result<Box<dyn ComputeProvider>> {
    Ok(match kind {
        ComputeProviderKind::None => Box::new(NoneProvider),
        ComputeProviderKind::NativeV1 => Box::new(NativeProvider),
        ComputeProviderKind::WasmV1 => Box::new(WasmProvider::new()?),
    })
}

/// The raw native `ComputeBankV1` for one input (parity tests, probes).
pub fn native_bank(input: &[u8]) -> Result<Vec<u8>, ComputeError> {
    let mut out = recur64_coproc::empty_output();
    compute_bank(input, &mut out)?;
    Ok(out)
}

// --- WorldModelV2 providers -------------------------------------------------------------

/// A deterministic `WorldModelV2` provider (native or WebAssembly).
pub trait WorldModelProvider: Send + Sync {
    /// The execution backend label.
    fn kind(&self) -> ComputeProviderKind;
    /// The semantic version of the layout and rules (independent of the backend).
    fn semantic_version(&self) -> &'static str {
        recur64_coproc::world::WORLD_MODEL_VERSION
    }
    /// Compute the packed world model for each input into a fresh buffer of
    /// `world_output_len(w_cap, r_cap)` bytes.
    fn world_batch(
        &self,
        inputs: &[Vec<u8>],
        w_cap: usize,
        r_cap: usize,
        horizon: WorldHorizon,
    ) -> Result<Vec<Vec<u8>>, ComputeError>;
}

/// Native `WorldModelV2`.
pub struct NativeWorldModel;

impl WorldModelProvider for NativeWorldModel {
    fn kind(&self) -> ComputeProviderKind {
        ComputeProviderKind::NativeV1
    }
    fn world_batch(
        &self,
        inputs: &[Vec<u8>],
        w_cap: usize,
        r_cap: usize,
        horizon: WorldHorizon,
    ) -> Result<Vec<Vec<u8>>, ComputeError> {
        let len = recur64_coproc::world::world_output_len(w_cap, r_cap);
        inputs
            .iter()
            .map(|input| {
                let mut out = vec![0u8; len];
                recur64_coproc::world::world_model(input, w_cap, r_cap, horizon, &mut out)?;
                Ok(out)
            })
            .collect()
    }
}

/// (in_ptr, in_len, w_cap, r_cap, horizon, out_ptr, out_len)
type WorldArgs = (i32, i32, i32, i32, i32, i32, i32);

struct WasmWorldInstance {
    store: wasmi::Store<()>,
    memory: wasmi::Memory,
    alloc: wasmi::TypedFunc<i32, i32>,
    dealloc: wasmi::TypedFunc<(i32, i32), ()>,
    world: wasmi::TypedFunc<WorldArgs, i32>,
    in_ptr: i32,
}

/// WebAssembly `WorldModelV2`: the same source, run by the pure-Rust interpreter.
pub struct WasmWorldModel {
    inner: Mutex<WasmWorldInstance>,
}

impl WasmWorldModel {
    pub fn new() -> anyhow::Result<Self> {
        let engine = wasmi::Engine::default();
        let module = wasmi::Module::new(&engine, artifact::WASM_ARTIFACT)
            .map_err(|e| anyhow::anyhow!("instantiate {}: {e}", artifact::WASM_SHA256))?;
        let mut store = wasmi::Store::new(&engine, ());
        let linker = wasmi::Linker::new(&engine);
        let instance = linker
            .instantiate(&mut store, &module)
            .map_err(|e| anyhow::anyhow!("link: {e}"))?
            .start(&mut store)
            .map_err(|e| anyhow::anyhow!("start: {e}"))?;
        let memory = instance
            .get_memory(&store, "memory")
            .ok_or_else(|| anyhow::anyhow!("the guest exports no `memory`"))?;
        let alloc = instance
            .get_typed_func::<i32, i32>(&store, "coproc_alloc")
            .map_err(|e| anyhow::anyhow!("coproc_alloc: {e}"))?;
        let dealloc = instance
            .get_typed_func::<(i32, i32), ()>(&store, "coproc_dealloc")
            .map_err(|e| anyhow::anyhow!("coproc_dealloc: {e}"))?;
        let world = instance
            .get_typed_func::<(i32, i32, i32, i32, i32, i32, i32), i32>(
                &store,
                "coproc_world_model",
            )
            .map_err(|e| anyhow::anyhow!("coproc_world_model: {e}"))?;
        let in_ptr = alloc
            .call(&mut store, INPUT_LEN as i32)
            .map_err(|e| anyhow::anyhow!("alloc input: {e}"))?;
        Ok(Self {
            inner: Mutex::new(WasmWorldInstance {
                store,
                memory,
                alloc,
                dealloc,
                world,
                in_ptr,
            }),
        })
    }
}

impl WorldModelProvider for WasmWorldModel {
    fn kind(&self) -> ComputeProviderKind {
        ComputeProviderKind::WasmV1
    }
    fn world_batch(
        &self,
        inputs: &[Vec<u8>],
        w_cap: usize,
        r_cap: usize,
        horizon: WorldHorizon,
    ) -> Result<Vec<Vec<u8>>, ComputeError> {
        let len = recur64_coproc::world::world_output_len(w_cap, r_cap);
        let mut guard = self.inner.lock().map_err(|_| {
            ComputeError::Wasm("the wasm store mutex was poisoned by an earlier panic".into())
        })?;
        let inst = &mut *guard;
        let out_ptr = inst
            .alloc
            .call(&mut inst.store, len as i32)
            .map_err(|e| ComputeError::Wasm(format!("alloc output: {e}")))?;
        let mut results = Vec::with_capacity(inputs.len());
        let mut failure: Option<ComputeError> = None;
        for (i, input) in inputs.iter().enumerate() {
            if input.len() != INPUT_LEN {
                failure = Some(ComputeError::BufferShape(format!(
                    "input {i} is {} bytes, expected {INPUT_LEN}",
                    input.len()
                )));
                break;
            }
            if let Err(e) = inst
                .memory
                .write(&mut inst.store, inst.in_ptr as usize, input)
            {
                failure = Some(ComputeError::Wasm(format!("write input: {e}")));
                break;
            }
            let status = match inst.world.call(
                &mut inst.store,
                (
                    inst.in_ptr,
                    INPUT_LEN as i32,
                    w_cap as i32,
                    r_cap as i32,
                    horizon.code() as i32,
                    out_ptr,
                    len as i32,
                ),
            ) {
                Ok(s) => s,
                Err(e) => {
                    failure = Some(ComputeError::Wasm(format!("call: {e}")));
                    break;
                }
            };
            if status != 0 {
                failure = Some(ComputeError::Wasm(format!(
                    "guest returned status {status} for position {i}"
                )));
                break;
            }
            let mut buf = vec![0u8; len];
            if let Err(e) = inst.memory.read(&inst.store, out_ptr as usize, &mut buf) {
                failure = Some(ComputeError::Wasm(format!("read output: {e}")));
                break;
            }
            results.push(buf);
        }
        let _ = inst.dealloc.call(&mut inst.store, (out_ptr, len as i32));
        match failure {
            Some(e) => Err(e),
            None => Ok(results),
        }
    }
}

/// Build the `WorldModelV2` provider for a label. `None` is refused: a world
/// model that is off must not be constructed.
pub fn world_model_provider_for(
    kind: ComputeProviderKind,
) -> anyhow::Result<Box<dyn WorldModelProvider>> {
    Ok(match kind {
        ComputeProviderKind::None => anyhow::bail!("the world model provider cannot be `none`"),
        ComputeProviderKind::NativeV1 => Box::new(NativeWorldModel),
        ComputeProviderKind::WasmV1 => Box::new(WasmWorldModel::new()?),
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    use recur64_core::{GameState, StandardMove, encode_observation_v1};
    use recur64_search::Rng;
    use std::collections::HashSet;

    fn rand_positions(n: usize) -> Vec<GameState> {
        let mut rng = Rng::new(20260929);
        let mut out = Vec::with_capacity(n);
        let mut seen = HashSet::new();
        while out.len() < n {
            let mut s = GameState::startpos();
            let plies = (rng.next_u64() % 90) as usize;
            for _ in 0..plies {
                if s.termination().is_some() {
                    break;
                }
                let legal = s.legal_actions();
                let a = legal[(rng.next_u64() % legal.len() as u64) as usize];
                let (from, to, promo) = a.to_physical(s.perspective());
                s.apply(StandardMove::new(
                    from,
                    to,
                    (!promo.is_none()).then_some(promo),
                ))
                .expect("generated move is legal");
            }
            if s.termination().is_none() && seen.insert(s.to_fen()) {
                out.push(s);
            }
        }
        out
    }

    /// Hand-picked special-move and terminal states.
    fn edge_positions() -> Vec<(String, u8)> {
        vec![
            (
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1".into(),
                2,
            ),
            // Castling available.
            ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1".into(), 2),
            // En passant available.
            (
                "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3".into(),
                2,
            ),
            // Promotion, including capture promotions.
            ("1n5k/P7/8/8/8/8/8/K7 w - - 0 1".into(), 2),
            ("2r4k/1P6/8/8/8/8/8/K7 w - - 0 1".into(), 2),
            // Check and double check.
            ("4Q2k/8/8/8/8/8/8/K7 b - - 0 1".into(), 2),
            ("3qk3/8/8/8/8/8/8/K6R w - - 0 1".into(), 2),
            // Pins and rays.
            ("8/8/8/8/8/2k5/4R3/4K3 b - - 0 1".into(), 2),
            // Mate in 1.
            ("6k1/5ppp/8/8/8/8/8/R6K w - - 0 1".into(), 2),
            // Mate in 2 (queen + king).
            ("7k/8/8/8/8/8/6Q1/6K1 w - - 0 1".into(), 2),
            // Stalemate.
            ("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1".into(), 2),
            // Checkmate.
            ("6k1/5ppp/8/8/8/8/8/R6K w - - 0 1".into(), 2),
            // Insufficient material.
            ("8/8/8/4k3/8/8/4B3/4K3 w - - 0 1".into(), 2),
            // Halfmove clock and repetition features.
            ("8/8/8/4k3/8/8/3Q4/4K3 w - - 60 40".into(), 2),
        ]
    }

    fn input_for(state: &GameState, depth: u8) -> Vec<u8> {
        let obs = encode_observation_v1(state);
        let legal = state.legal_actions();
        encode_input(&obs, &legal, depth).expect("input encodes")
    }

    #[test]
    fn embedded_artifact_matches_pinned_digest() {
        use sha2::{Digest, Sha256};
        let actual = format!("{:x}", Sha256::digest(artifact::WASM_ARTIFACT));
        assert_eq!(
            actual,
            artifact::WASM_SHA256,
            "the embedded .wasm does not match its pinned digest; \
             re-run scripts/build-compute-wasm.ps1"
        );
        assert_eq!(artifact::WASM_ARTIFACT.len(), artifact::WASM_ARTIFACT_BYTES);
    }

    #[test]
    fn native_and_wasm_agree_byte_exactly() {
        let native = NativeProvider;
        let wasm = WasmProvider::new().expect("wasm provider instantiates");

        let mut inputs: Vec<Vec<u8>> = Vec::new();
        for (fen, depth) in edge_positions() {
            let state = GameState::from_fen(&fen).expect(&fen);
            // Terminal positions are included on purpose: the bank must be
            // well defined (legal count 0) rather than refused.
            inputs.push(input_for(&state, depth));
        }
        for state in rand_positions(400) {
            inputs.push(input_for(&state, 1));
        }
        // A large random subset also exercises the mate-in-2 path.
        for state in rand_positions(60) {
            inputs.push(input_for(&state, 2));
        }

        let mut a: Vec<Vec<u8>> = inputs
            .iter()
            .map(|_| recur64_coproc::empty_output())
            .collect();
        let mut b: Vec<Vec<u8>> = inputs
            .iter()
            .map(|_| recur64_coproc::empty_output())
            .collect();
        native.compute_batch(&inputs, &mut a).expect("native");
        wasm.compute_batch(&inputs, &mut b).expect("wasm");
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(x, y, "native and wasm differ on position {i}");
        }
    }

    #[test]
    fn world_model_native_and_wasm_agree_byte_exactly() {
        let native = NativeWorldModel;
        let wasm = WasmWorldModel::new().expect("wasm world model instantiates");
        let mut inputs: Vec<Vec<u8>> = Vec::new();
        for (fen, _) in edge_positions() {
            let state = GameState::from_fen(&fen).expect(&fen);
            inputs.push(input_for(&state, 0));
        }
        for state in rand_positions(24) {
            inputs.push(input_for(&state, 0));
        }
        let (w_cap, r_cap) = (128, 64);
        for horizon in WorldHorizon::ALL {
            let a = native
                .world_batch(&inputs, w_cap, r_cap, horizon)
                .expect("native");
            let b = wasm
                .world_batch(&inputs, w_cap, r_cap, horizon)
                .expect("wasm");
            assert_eq!(a.len(), b.len());
            for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
                assert_eq!(
                    x,
                    y,
                    "native and wasm world models differ on position {i} at horizon {}",
                    horizon.label()
                );
            }
        }
        // A capacity error is an error in both providers, never a truncation.
        let startpos = vec![input_for(&GameState::startpos(), 0)];
        assert!(
            native
                .world_batch(&startpos, 8, 8, WorldHorizon::Replies)
                .is_err()
        );
        assert!(
            wasm.world_batch(&startpos, 8, 8, WorldHorizon::Replies)
                .is_err()
        );
    }

    /// The pinned source digest must equal the digest of the sources in the tree. A
    /// change to the coprocessor sources without re-running
    /// `scripts/build-compute-wasm.ps1` fails here. Algorithm is documented in that
    /// script: SHA-256 over, per file in ordinal path order, path + 0x00 +
    /// CRLF-normalised bytes + 0x00.
    #[test]
    fn artifact_source_digest_matches_the_sources() {
        use sha2::{Digest, Sha256};
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..");
        let mut files: Vec<String> = std::fs::read_dir(root.join("crates/recur64-coproc/src"))
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().to_string())
            .filter(|n| n.ends_with(".rs"))
            .map(|n| format!("crates/recur64-coproc/src/{n}"))
            .collect();
        files.push("crates/recur64-coproc-guest/src/lib.rs".into());
        files.push("crates/recur64-coproc/Cargo.toml".into());
        files.push("crates/recur64-coproc-guest/Cargo.toml".into());
        files.sort();
        let mut hasher = Sha256::new();
        for rel in &files {
            hasher.update(rel.as_bytes());
            hasher.update([0u8]);
            let raw = std::fs::read(root.join(rel)).unwrap();
            let mut norm = Vec::with_capacity(raw.len());
            let mut i = 0;
            while i < raw.len() {
                if raw[i] == b'\r' && raw.get(i + 1) == Some(&b'\n') {
                    i += 1;
                    continue;
                }
                norm.push(raw[i]);
                i += 1;
            }
            hasher.update(&norm);
            hasher.update([0u8]);
        }
        let digest = format!("{:x}", hasher.finalize());
        assert_eq!(
            digest,
            artifact::WASM_SOURCE_DIGEST,
            "the coprocessor sources changed since the WASM artifact was built: run \
             scripts/build-compute-wasm.ps1 and commit the result"
        );
    }

    #[test]
    fn provider_labels_round_trip_and_unknown_is_refused() {
        for k in [
            ComputeProviderKind::None,
            ComputeProviderKind::NativeV1,
            ComputeProviderKind::WasmV1,
        ] {
            assert_eq!(ComputeProviderKind::parse(k.label()).unwrap(), k);
        }
        assert!(ComputeProviderKind::parse("stockfish").is_err());
    }

    #[test]
    fn none_provider_does_no_work() {
        let p = provider_for(ComputeProviderKind::None).unwrap();
        let inputs = vec![vec![0u8; 4]];
        let mut out = vec![vec![0u8; 4]];
        // With the pathway off there is nothing to validate and nothing written.
        p.compute_batch(&inputs, &mut out).unwrap();
        assert_eq!(out[0], vec![0u8; 4]);
    }

    #[test]
    fn mismatched_buffers_are_refused() {
        let p = NativeProvider;
        let inputs = vec![input_for(&GameState::startpos(), 0)];
        let mut out = vec![
            recur64_coproc::empty_output(),
            recur64_coproc::empty_output(),
        ];
        assert!(matches!(
            p.compute_batch(&inputs, &mut out),
            Err(ComputeError::BufferShape(_))
        ));
    }

    #[test]
    fn bank_version_is_reported() {
        let p = NativeProvider;
        assert_eq!(p.semantic_version(), COMPUTE_BANK_VERSION);
        assert_eq!(p.semantic_version(), "compute_bank_v1");
    }
}
