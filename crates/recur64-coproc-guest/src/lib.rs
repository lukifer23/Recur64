//! WebAssembly guest for the deterministic chess coprocessor.
//!
//! This crate contains no algorithm of its own: it is the thin wasm ABI over
//! [`recur64_coproc::compute`], the *same* source the native `NativeV1`
//! provider calls. That is why the two providers are byte-identical by
//! construction rather than by coincidence.
//!
//! ABI (all pointers are guest linear-memory offsets):
//!
//! * `coproc_alloc(len) -> ptr` — reserve `len` bytes.
//! * `coproc_dealloc(ptr, len)` — release a buffer from `coproc_alloc`.
//! * `coproc_compute(in_ptr, in_len, out_ptr, out_len) -> status`
//!   `0` = ok, `1` = the input was structurally invalid, `2` = wrong buffer
//!   length, `3` = the position was rejected by the coprocessor.

use recur64_coproc::{INPUT_LEN, OUTPUT_LEN, compute};

/// Allocate `len` bytes in guest memory.
///
/// # Safety
/// The returned pointer must be released with [`coproc_dealloc`] using the same
/// length.
#[unsafe(no_mangle)]
pub extern "C" fn coproc_alloc(len: usize) -> *mut u8 {
    let mut buffer = Vec::<u8>::with_capacity(len);
    let ptr = buffer.as_mut_ptr();
    // The guest hands out the raw buffer; the host owns it until dealloc.
    std::mem::forget(buffer);
    ptr
}

/// Release a buffer from [`coproc_alloc`].
///
/// # Safety
/// `ptr` must come from [`coproc_alloc`] with the same `len`, and must not be
/// used afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn coproc_dealloc(ptr: *mut u8, len: usize) {
    if ptr.is_null() {
        return;
    }
    // SAFETY: the caller guarantees `ptr`/`len` came from `coproc_alloc`.
    unsafe {
        let _ = Vec::from_raw_parts(ptr, 0, len);
    }
}

/// Compute `ComputeBankV1` into the caller's output buffer.
///
/// # Safety
/// `in_ptr..in_ptr + in_len` and `out_ptr..out_ptr + out_len` must be valid,
/// non-overlapping guest-memory regions.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn coproc_compute(
    in_ptr: *const u8,
    in_len: usize,
    out_ptr: *mut u8,
    out_len: usize,
) -> i32 {
    if in_len != INPUT_LEN || out_len != OUTPUT_LEN {
        return 2;
    }
    if in_ptr.is_null() || out_ptr.is_null() {
        return 2;
    }
    // SAFETY: the caller guarantees the regions and their lengths.
    let input = unsafe { std::slice::from_raw_parts(in_ptr, in_len) };
    // SAFETY: as above; the provider never aliases the input and output.
    let out = unsafe { std::slice::from_raw_parts_mut(out_ptr, out_len) };
    match compute::compute_bank(input, out) {
        Ok(()) => 0,
        Err(recur64_coproc::CoprocError::InvalidObservation(_)) => 3,
        Err(recur64_coproc::CoprocError::InvalidBoard(_)) => 3,
        Err(recur64_coproc::CoprocError::BadInputLength(_))
        | Err(recur64_coproc::CoprocError::BadOutputLength(_)) => 2,
        Err(_) => 1,
    }
}

/// Compute `WorldModelV2` into the caller's output buffer.
///
/// Status: `0` ok, `1` structurally invalid input, `2` wrong buffer length,
/// `3` the position was rejected, `4` a capacity (`w_cap` / `r_cap`) was exceeded.
/// `horizon` is the [`recur64_coproc::world::WorldHorizon`] code (1 root, 2 successor, 3 replies).
///
/// # Safety
/// `in_ptr..in_ptr + in_len` and `out_ptr..out_ptr + out_len` must be valid,
/// non-overlapping guest-memory regions.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn coproc_world_model(
    in_ptr: *const u8,
    in_len: usize,
    w_cap: usize,
    r_cap: usize,
    horizon: usize,
    out_ptr: *mut u8,
    out_len: usize,
) -> i32 {
    if in_len != INPUT_LEN || in_ptr.is_null() || out_ptr.is_null() {
        return 2;
    }
    // SAFETY: the caller guarantees the regions and their lengths.
    let input = unsafe { std::slice::from_raw_parts(in_ptr, in_len) };
    // SAFETY: as above; the provider never aliases the input and output.
    let out = unsafe { std::slice::from_raw_parts_mut(out_ptr, out_len) };
    let Some(horizon) = recur64_coproc::world::WorldHorizon::from_code(horizon as u8) else {
        return 1;
    };
    match recur64_coproc::world::world_model(input, w_cap, r_cap, horizon, out) {
        Ok(()) => 0,
        Err(recur64_coproc::CoprocError::Capacity(_)) => 4,
        Err(recur64_coproc::CoprocError::InvalidObservation(_))
        | Err(recur64_coproc::CoprocError::InvalidBoard(_)) => 3,
        Err(recur64_coproc::CoprocError::BadInputLength(_))
        | Err(recur64_coproc::CoprocError::BadOutputLength(_)) => 2,
        Err(_) => 1,
    }
}

/// Input-buffer length the guest expects.
#[unsafe(no_mangle)]
pub extern "C" fn coproc_input_len() -> usize {
    INPUT_LEN
}

/// Output-buffer length the guest expects.
#[unsafe(no_mangle)]
pub extern "C" fn coproc_output_len() -> usize {
    OUTPUT_LEN
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abi_lengths_match_the_coprocessor() {
        assert_eq!(coproc_input_len(), INPUT_LEN);
        assert_eq!(coproc_output_len(), OUTPUT_LEN);
    }

    #[test]
    fn abi_reports_bad_lengths() {
        let input = [0u8; 8];
        let mut out = vec![0u8; OUTPUT_LEN];
        // SAFETY: live buffers of the stated lengths.
        let status =
            unsafe { coproc_compute(input.as_ptr(), input.len(), out.as_mut_ptr(), out.len()) };
        assert_eq!(status, 2);
    }

    #[test]
    fn alloc_dealloc_round_trip() {
        let ptr = coproc_alloc(64);
        assert!(!ptr.is_null());
        // SAFETY: paired with the allocation above.
        unsafe { coproc_dealloc(ptr, 64) };
    }
}
