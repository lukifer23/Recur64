//! Cooperative cancellation.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// A cloneable cancellation token. Set by Ctrl+C (or programmatically).
#[derive(Clone, Default)]
pub struct CancelToken(Arc<AtomicBool>);

impl CancelToken {
    pub fn new() -> Self {
        Self(Arc::new(AtomicBool::new(false)))
    }

    pub fn cancel(&self) {
        self.0.store(true, Ordering::SeqCst);
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }

    /// Install a Ctrl+C handler that sets this token.
    pub fn install_handler(&self) -> anyhow::Result<()> {
        let token = self.clone();
        ctrlc::set_handler(move || token.cancel())
            .map_err(|e| anyhow::anyhow!("install Ctrl+C handler: {e}"))
    }
}
