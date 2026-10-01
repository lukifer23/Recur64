//! Host-side flattening of a batch of [`AllInfoTree`]s into index arrays.
//!
//! Layout of the global state list: all depth-1 states first (example-major, then root
//! candidate order), then all depth-2 states (example, branch, reply order). Per-branch
//! token slots are `[b * w + i, t]` with slot 0 the depth-1 state and slots `1..` its
//! replies. Nothing here reads anything but raw state fields.

use recur64_core::ActionId;

use super::tree::AllInfoTree;

/// Squares x observation features per state.
const OBS_LEN: usize = 64 * 119;

/// Index arrays for one batch (all integers are row indices into device tables).
pub struct TreeBatch {
    pub batch: usize,
    /// Root candidate width (the policy width).
    pub width: usize,
    /// Slots per branch: 1 + the largest reply count in the batch.
    pub slots: usize,
    pub n1: usize,
    pub n2: usize,
    /// `[(n1 + n2) * 64 * 119]` observations: depth-1 states then depth-2 states.
    pub obs: Vec<f32>,
    /// Width of the depth-1 action tables (at least 1).
    pub action_width: usize,
    /// `[n1 * action_width]` from / to squares and promotion codes of every depth-1
    /// state's own legal actions (padding repeats index 0 and is never gathered).
    pub from: Vec<i32>,
    pub to: Vec<i32>,
    pub promo: Vec<i32>,
    /// Per slot `[b * w * slots]`: row into the node table / edge table.
    pub own: Vec<i32>,
    pub parent: Vec<i32>,
    pub edge: Vec<i32>,
    /// Per slot: the root candidate token of its branch (row into the edge table).
    pub root_token: Vec<i32>,
    /// Per slot: 0 = depth-1 state, 1 = depth-2 reply.
    pub depth: Vec<i32>,
    /// Per slot `[.., 2]`: terminal, in_check.
    pub flags: Vec<f32>,
    /// Per slot: true where the slot holds no state.
    pub slot_pad: Vec<bool>,
    /// `(depth-1 states, depth-2 states)` of every example.
    pub counts: Vec<(usize, usize)>,
    /// Sentinel (all-zero) rows of the node and edge tables.
    pub node_zero: i32,
    pub edge_zero: i32,
}

impl TreeBatch {
    /// Flatten `trees`. `width` must be the root candidate width of the batch (the largest
    /// root legal count) and every tree's branch count must be at most `width`.
    pub fn build(trees: &[AllInfoTree], width: usize) -> anyhow::Result<Self> {
        let batch = trees.len();
        anyhow::ensure!(batch > 0, "empty batch");
        let mut n1 = 0usize;
        let mut n2 = 0usize;
        let mut max_replies = 0usize;
        let mut a_w = 1usize;
        let mut counts = Vec::with_capacity(batch);
        for t in trees {
            anyhow::ensure!(
                t.branches.len() <= width,
                "a tree has {} branches, wider than the candidate width {width}",
                t.branches.len()
            );
            let c = t.counts();
            counts.push((c.depth1, c.depth2));
            n1 += c.depth1;
            n2 += c.depth2;
            for br in &t.branches {
                max_replies = max_replies.max(br.replies.len());
                a_w = a_w.max(br.child.legal_actions.len());
            }
        }
        let slots = 1 + max_replies;
        let n = n1 + n2;
        let node_zero = (batch + n) as i32;
        let edge_zero = (batch * width + n1 * a_w) as i32;

        let mut obs = vec![0.0f32; n * OBS_LEN];
        let mut from = vec![0i32; n1 * a_w];
        let mut to = vec![0i32; n1 * a_w];
        let mut promo = vec![0i32; n1 * a_w];
        let total_slots = batch * width * slots;
        let mut own = vec![node_zero; total_slots];
        let mut parent = vec![node_zero; total_slots];
        let mut edge = vec![edge_zero; total_slots];
        let root_token: Vec<i32> = (0..total_slots).map(|p| (p / slots) as i32).collect();
        let mut depth = vec![0i32; total_slots];
        let mut flags = vec![0.0f32; total_slots * 2];
        let mut slot_pad = vec![true; total_slots];

        let (mut j, mut r) = (0usize, 0usize); // depth-1 / depth-2 ordinals
        for (b, tree) in trees.iter().enumerate() {
            for (i, br) in tree.branches.iter().enumerate() {
                let branch_row = b * width + i;
                let base = branch_row * slots;
                // depth-1 state
                obs[j * OBS_LEN..(j + 1) * OBS_LEN]
                    .copy_from_slice(br.child.observation.as_slice());
                for (k, &act) in br.child.legal_actions.iter().enumerate() {
                    let (f, t, p) = ActionId::from_index(u32::from(act))?.decode();
                    from[j * a_w + k] = f as i32;
                    to[j * a_w + k] = t as i32;
                    promo[j * a_w + k] = i32::from(p.code());
                }
                own[base] = (batch + j) as i32;
                parent[base] = b as i32;
                edge[base] = branch_row as i32;
                depth[base] = 0;
                flags[base * 2] = f32::from(u8::from(br.child.terminal));
                flags[base * 2 + 1] = f32::from(u8::from(br.child.in_check));
                slot_pad[base] = false;
                anyhow::ensure!(
                    br.replies.len() == br.child.legal_actions.len(),
                    "branch {i} of example {b}: {} replies for {} legal replies (truncation)",
                    br.replies.len(),
                    br.child.legal_actions.len()
                );
                for (k, rep) in br.replies.iter().enumerate() {
                    let at = n1 + r;
                    obs[at * OBS_LEN..(at + 1) * OBS_LEN]
                        .copy_from_slice(rep.observation.as_slice());
                    let slot = base + 1 + k;
                    own[slot] = (batch + n1 + r) as i32;
                    parent[slot] = (batch + j) as i32;
                    edge[slot] = (batch * width + j * a_w + k) as i32;
                    depth[slot] = 1;
                    flags[slot * 2] = f32::from(u8::from(rep.terminal));
                    flags[slot * 2 + 1] = f32::from(u8::from(rep.in_check));
                    slot_pad[slot] = false;
                    r += 1;
                }
                j += 1;
            }
        }
        debug_assert_eq!((j, r), (n1, n2));
        Ok(Self {
            batch,
            width,
            slots,
            n1,
            n2,
            obs,
            action_width: a_w,
            from,
            to,
            promo,
            own,
            parent,
            edge,
            root_token,
            depth,
            flags,
            slot_pad,
            counts,
            node_zero,
            edge_zero,
        })
    }
}
