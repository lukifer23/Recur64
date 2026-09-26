//! Periodic GPU telemetry via `nvidia-smi` (VRAM, utilization, temperature).
//!
//! Sampling is best-effort observation: when `nvidia-smi` is unavailable the
//! summary simply has zero samples and `None` fields, which reports show as
//! missing data rather than as zeros.

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

/// Default polling period.
pub const SAMPLE_PERIOD: Duration = Duration::from_millis(500);

/// Sample (memory.used MiB, utilization %, temperature C) for GPU 0.
/// `None` when nvidia-smi is unavailable or its output cannot be parsed.
pub fn sample_gpu() -> Option<(u64, u64, u64)> {
    let out = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.used,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout);
    let mut fields = s
        .lines()
        .next()?
        .split(',')
        .map(|f| f.trim().parse::<u64>());
    Some((
        fields.next()?.ok()?,
        fields.next()?.ok()?,
        fields.next()?.ok()?,
    ))
}

/// Accumulated samples over one measured interval.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct GpuSamples {
    pub peak_vram_mb: Option<u64>,
    pub min_vram_mb: Option<u64>,
    /// VRAM of the last sample (end of the interval).
    pub last_vram_mb: Option<u64>,
    /// Mean utilization over samples with utilization > 0.
    pub util_busy_mean: Option<f64>,
    pub util_max: Option<u64>,
    pub temp_max_c: Option<u64>,
    pub samples: u64,
    #[serde(skip)]
    busy_util_sum: u64,
    #[serde(skip)]
    busy_samples: u64,
}

impl GpuSamples {
    pub fn record(&mut self, (mem, util, temp): (u64, u64, u64)) {
        self.samples += 1;
        self.peak_vram_mb = Some(self.peak_vram_mb.map_or(mem, |p| p.max(mem)));
        self.min_vram_mb = Some(self.min_vram_mb.map_or(mem, |p| p.min(mem)));
        self.last_vram_mb = Some(mem);
        self.util_max = Some(self.util_max.map_or(util, |p| p.max(util)));
        self.temp_max_c = Some(self.temp_max_c.map_or(temp, |p| p.max(temp)));
        if util > 0 {
            self.busy_util_sum += util;
            self.busy_samples += 1;
            self.util_busy_mean = Some(self.busy_util_sum as f64 / self.busy_samples as f64);
        }
    }

    /// Record one sample now (no-op when nvidia-smi is unavailable).
    pub fn sample_now(&mut self) {
        if let Some(v) = sample_gpu() {
            self.record(v);
        }
    }
}

/// Run `work` while a background thread samples the GPU every
/// [`SAMPLE_PERIOD`]. One sample is taken before and one after `work`, so
/// even short intervals report their boundary VRAM. When `enabled` is false
/// no sampling happens and the summary is empty.
pub fn monitor<T>(enabled: bool, work: impl FnOnce() -> T) -> (T, GpuSamples) {
    if !enabled {
        return (work(), GpuSamples::default());
    }
    let samples = Mutex::new(GpuSamples::default());
    samples.lock().expect("GPU sampler mutex").sample_now();
    let running = AtomicBool::new(true);
    let out = std::thread::scope(|scope| {
        scope.spawn(|| {
            while running.load(Ordering::Relaxed) {
                std::thread::sleep(SAMPLE_PERIOD);
                if running.load(Ordering::Relaxed) {
                    samples.lock().expect("GPU sampler mutex").sample_now();
                }
            }
        });
        let out = work();
        running.store(false, Ordering::Relaxed);
        out
    });
    let mut samples = samples.into_inner().expect("GPU sampler mutex");
    samples.sample_now();
    (out, samples)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn samples_track_peak_min_last_and_busy_mean() {
        let mut s = GpuSamples::default();
        s.record((100, 0, 40));
        s.record((300, 50, 45));
        s.record((200, 70, 43));
        assert_eq!(s.samples, 3);
        assert_eq!(s.peak_vram_mb, Some(300));
        assert_eq!(s.min_vram_mb, Some(100));
        assert_eq!(s.last_vram_mb, Some(200));
        assert_eq!(s.util_max, Some(70));
        assert_eq!(s.temp_max_c, Some(45));
        assert_eq!(s.util_busy_mean, Some(60.0), "idle samples excluded");
    }

    #[test]
    fn disabled_monitor_reports_no_samples() {
        let (v, s) = monitor(false, || 7);
        assert_eq!(v, 7);
        assert_eq!(s.samples, 0);
        assert!(s.peak_vram_mb.is_none());
    }
}
