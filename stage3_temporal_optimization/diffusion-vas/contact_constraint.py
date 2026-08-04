"""
Contact / penetration constraints for hand-object fitting (Stage 5/6).

Design overview
---------------
- Per-step (with grad):
    * `compute_penetration_loss`: differentiable one-sided push-out for inside
      hand vertices.  Uses KNN(hand -> obj_verts) + obj vertex normals every
      step, no cache needed.
    * `compute_soft_contact_loss`: differentiable contact attraction using a
      cached top-k (face_id, bary_coords, weight) correspondence built every
      `reclassify_every` steps.

- Periodic (no_grad, every `reclassify_every` steps):
    * `classify_and_build_correspondence`: classifies each hand vertex into
      {far, inside, near_contact} and builds the soft top-k correspondence for
      near_contact vertices.  Optionally consumes the previous-frame argmax
      face as a sticky prior (temporal consistency).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from pytorch3d.ops import knn_points

from utils import get_NN, get_interior, batched_index_select


BUCKET_FAR = 0
BUCKET_INSIDE = 1
BUCKET_NEAR_CONTACT = 2


@torch.no_grad()
def compute_interaction_boundary_blend_offsets(
    num_frames: int,
    i0: int,
    i1: int,
    jump: torch.Tensor,
    gamma: float = 2.0,
    max_jump: float = 0.05,
) -> Tuple[torch.Tensor, dict]:
    """Ramp a boundary translation jump back into the interaction segment.

    Offsets are zero outside [i0, i1), and reach the clamped `jump` at
    frame i1 - 1.  Applying the same offset to hand and object translations
    preserves their relative pose while smoothing the global trajectory.
    """
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if not (0 <= i0 < i1 <= num_frames):
        raise ValueError(f"invalid interaction range [{i0}, {i1}) for {num_frames} frames")
    if jump.numel() != 3:
        raise ValueError(f"jump must have 3 elements, got shape {tuple(jump.shape)}")

    jump = jump.reshape(3)
    offsets = torch.zeros(num_frames, 3, dtype=jump.dtype, device=jump.device)
    jump_norm_t = torch.linalg.norm(jump)
    jump_norm = float(jump_norm_t.item())
    max_jump_f = float(max_jump)
    if jump_norm > max_jump_f > 0:
        jump_for_apply = jump * (max_jump_f / max(jump_norm, 1e-12))
    else:
        jump_for_apply = jump

    n_inter = i1 - i0
    if n_inter == 1:
        alpha = torch.ones(1, dtype=jump.dtype, device=jump.device)
    else:
        alpha = torch.linspace(0.0, 1.0, n_inter, dtype=jump.dtype, device=jump.device)
        alpha = alpha.clamp(0.0, 1.0).pow(float(gamma))
    offsets[i0:i1] = alpha[:, None] * jump_for_apply[None]

    applied_norm = torch.linalg.norm(offsets, dim=1)
    return offsets, {
        "jump_norm": jump_norm,
        "applied_max_norm": float(applied_norm.max().item()) if applied_norm.numel() > 0 else 0.0,
        "clamped": bool(jump_norm > max_jump_f > 0),
    }


@torch.no_grad()
def compute_interaction_alignment_offsets(
    obj_trans: torch.Tensor,
    hand_trans: torch.Tensor,
    i0: int,
    i1: int,
    pre_buffer: int = 6,
    post_buffer: int = 6,
    max_offset: float = 0.08,
    gamma: float = 2.0,
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """Compute object/hand offsets that smooth interaction boundary jumps.

    Object offsets are applied only inside [i0, i1) so the interaction segment
    aligns with the neighboring static object positions.  Hand offsets use the
    same correction inside the interaction segment, then ramp into/out of that
    correction over non-interaction buffers to avoid hand discontinuities.
    """
    if obj_trans.ndim != 2 or obj_trans.shape[1] != 3:
        raise ValueError(f"obj_trans must have shape (N, 3), got {tuple(obj_trans.shape)}")
    if hand_trans.shape != obj_trans.shape:
        raise ValueError(
            f"hand_trans must match obj_trans shape, got {tuple(hand_trans.shape)} "
            f"vs {tuple(obj_trans.shape)}")
    num_frames = obj_trans.shape[0]
    if not (0 <= i0 < i1 <= num_frames):
        raise ValueError(f"invalid interaction range [{i0}, {i1}) for {num_frames} frames")

    obj_offsets = torch.zeros_like(obj_trans)
    hand_offsets = torch.zeros_like(obj_trans)
    dtype = obj_trans.dtype
    device = obj_trans.device

    start_offset = torch.zeros(3, dtype=dtype, device=device)
    end_offset = torch.zeros(3, dtype=dtype, device=device)
    if i0 > 0:
        start_offset = obj_trans[i0 - 1] - obj_trans[i0]
    if i1 < num_frames:
        end_offset = obj_trans[i1] - obj_trans[i1 - 1]

    def _clamp_vec(vec: torch.Tensor) -> torch.Tensor:
        norm = torch.linalg.norm(vec)
        max_offset_f = float(max_offset)
        if max_offset_f > 0 and float(norm.item()) > max_offset_f:
            return vec * (max_offset_f / max(float(norm.item()), 1e-12))
        return vec

    start_offset = _clamp_vec(start_offset)
    end_offset = _clamp_vec(end_offset)

    n_inter = i1 - i0
    if n_inter == 1:
        inter_alpha = torch.ones(1, dtype=dtype, device=device)
    else:
        inter_alpha = torch.linspace(0.0, 1.0, n_inter, dtype=dtype, device=device)
    inter_offsets = (1.0 - inter_alpha[:, None]) * start_offset[None] + inter_alpha[:, None] * end_offset[None]
    obj_offsets[i0:i1] = inter_offsets
    hand_offsets[i0:i1] = inter_offsets

    pre_hand_offset = torch.zeros(3, dtype=dtype, device=device)
    post_hand_offset = torch.zeros(3, dtype=dtype, device=device)
    if i0 > 0:
        pre_hand_offset = hand_trans[i0] + start_offset - hand_trans[i0 - 1]
    if i1 < num_frames:
        post_hand_offset = hand_trans[i1 - 1] + end_offset - hand_trans[i1]
    pre_hand_offset = _clamp_vec(pre_hand_offset)
    post_hand_offset = _clamp_vec(post_hand_offset)

    pre_start = max(0, i0 - max(0, int(pre_buffer)))
    pre_len = i0 - pre_start
    if pre_len > 0:
        alpha = torch.linspace(0.0, 1.0, pre_len, dtype=dtype, device=device).clamp(0.0, 1.0)
        alpha = alpha.pow(float(gamma))
        hand_offsets[pre_start:i0] = alpha[:, None] * pre_hand_offset[None]

    post_end = min(num_frames, i1 + max(0, int(post_buffer)))
    post_len = post_end - i1
    if post_len > 0:
        alpha = torch.linspace(1.0, 0.0, post_len, dtype=dtype, device=device).clamp(0.0, 1.0)
        alpha = alpha.pow(float(gamma))
        hand_offsets[i1:post_end] = alpha[:, None] * post_hand_offset[None]

    return obj_offsets, hand_offsets, {
        "start_jump_norm": float(torch.linalg.norm(obj_trans[i0] - obj_trans[i0 - 1]).item()) if i0 > 0 else 0.0,
        "end_jump_norm": float(torch.linalg.norm(obj_trans[i1] - obj_trans[i1 - 1]).item()) if i1 < num_frames else 0.0,
        "applied_start_norm": float(torch.linalg.norm(start_offset).item()),
        "applied_end_norm": float(torch.linalg.norm(end_offset).item()),
        "hand_pre_norm": float(torch.linalg.norm(pre_hand_offset).item()),
        "hand_post_norm": float(torch.linalg.norm(post_hand_offset).item()),
        "pre_buffer": int(pre_len),
        "post_buffer": int(post_len),
    }


@dataclass
class ContactCache:
    """Per-frame cached classification + soft correspondence.

    All tensors live on the same device as the input hand verts.
    Shapes:
        bucket:        (N_frames, 778)            int8
        signed_dist:   (N_frames, 778)            float (positive = inside depth)
        face_id_topk:  (N_frames, V_active, K)    long
        bary_topk:     (N_frames, V_active, K, 3) float
        weight_topk:   (N_frames, V_active, K)    float (sum to ~1 per vertex)
        active_idx:    (N_frames, V_active)       long  (hand vertex indices in [0,778))
        active_mask:   (N_frames, V_active)       bool
        argmax_face:   (N_frames, V_active)       long  (face_id with largest weight)
        contact_confidence:
                       (N_frames, V_active)       float in [0,1].  Temporal
                       memory for how likely this hand vertex should stay in
                       contact; multiplies the contact loss.
    """
    bucket: torch.Tensor
    signed_dist: torch.Tensor
    face_id_topk: torch.Tensor
    bary_topk: torch.Tensor
    weight_topk: torch.Tensor
    active_idx: torch.Tensor
    active_mask: torch.Tensor
    argmax_face: torch.Tensor
    contact_confidence: Optional[torch.Tensor] = None
    observed_contact_confidence: Optional[torch.Tensor] = None
    raw_observed_contact_confidence: Optional[torch.Tensor] = None
    raw_geometry_contact_confidence: Optional[torch.Tensor] = None


@torch.no_grad()
def propagate_dense_contact_memory(
    contact_confidence: torch.Tensor,       # (N_frames, N_hand)
    anchor_face: torch.Tensor,              # (N_frames, N_hand)
    anchor_bary: torch.Tensor,              # (N_frames, N_hand, 3)
    temporal_decay: float = 0.6,
    max_steps: int = 2,
    min_seed_conf: float = 0.35,
    min_keep_conf: float = 0.05,
    release_mask: Optional[torch.Tensor] = None,  # True blocks propagated contact
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Propagate dense contact memory across neighboring video frames.

    The memory is keyed by MANO vertex id.  A stable contact on frame `t`
    can seed the same hand vertex on `t-1` / `t+1`, carrying its object
    surface anchor (`face_id + bary`).  This gives short gaps a pull-back
    force without hard-coding frame ranges.
    """
    if contact_confidence.ndim != 2:
        raise ValueError("contact_confidence must have shape (N_frames, N_hand)")
    if anchor_face.shape != contact_confidence.shape:
        raise ValueError(
            f"anchor_face must match contact_confidence shape, got {tuple(anchor_face.shape)} "
            f"vs {tuple(contact_confidence.shape)}")
    if anchor_bary.shape != contact_confidence.shape + (3,):
        raise ValueError(
            f"anchor_bary must have shape {tuple(contact_confidence.shape + (3,))}, "
            f"got {tuple(anchor_bary.shape)}")
    if release_mask is not None and release_mask.shape != contact_confidence.shape:
        raise ValueError(
            f"release_mask must match contact_confidence shape, got {tuple(release_mask.shape)} "
            f"vs {tuple(contact_confidence.shape)}")

    out_conf = contact_confidence.clamp(0.0, 1.0).clone()
    out_face = anchor_face.clone()
    out_bary = anchor_bary.clone()

    if release_mask is not None:
        out_conf = torch.where(release_mask, torch.zeros_like(out_conf), out_conf)
        out_face = torch.where(release_mask, torch.full_like(out_face, -1), out_face)
        out_bary = torch.where(release_mask.unsqueeze(-1), torch.zeros_like(out_bary), out_bary)

    if out_conf.shape[0] <= 1 or max_steps <= 0:
        keep = out_conf >= float(min_keep_conf)
        out_conf = torch.where(keep, out_conf, torch.zeros_like(out_conf))
        out_face = torch.where(keep, out_face, torch.full_like(out_face, -1))
        out_bary = torch.where(keep.unsqueeze(-1), out_bary, torch.zeros_like(out_bary))
        return out_conf, out_face, out_bary

    decay = float(temporal_decay)
    min_seed = float(min_seed_conf)
    min_keep = float(min_keep_conf)

    def _propagate_slice(src_slice, dst_slice):
        src_conf = prev_conf[src_slice]
        prop_conf = (src_conf * decay).clamp(0.0, 1.0)
        prop_face = prev_face[src_slice]
        prop_bary = prev_bary[src_slice]
        valid = (src_conf >= min_seed) & (prop_conf >= min_keep) & (prop_face >= 0)
        if release_mask is not None:
            valid = valid & (~release_mask[dst_slice])
        replace = valid & (prop_conf > out_conf[dst_slice])
        out_conf[dst_slice] = torch.where(replace, prop_conf, out_conf[dst_slice])
        out_face[dst_slice] = torch.where(replace, prop_face, out_face[dst_slice])
        out_bary[dst_slice] = torch.where(replace.unsqueeze(-1), prop_bary, out_bary[dst_slice])

    for _ in range(int(max_steps)):
        prev_conf = out_conf.clone()
        prev_face = out_face.clone()
        prev_bary = out_bary.clone()
        _propagate_slice(slice(0, -1), slice(1, None))
        _propagate_slice(slice(1, None), slice(0, -1))

    keep = out_conf >= min_keep
    out_conf = torch.where(keep, out_conf, torch.zeros_like(out_conf))
    out_face = torch.where(keep, out_face, torch.full_like(out_face, -1))
    out_bary = torch.where(keep.unsqueeze(-1), out_bary, torch.zeros_like(out_bary))
    return out_conf, out_face, out_bary


@torch.no_grad()
def expand_reliable_contact_frontier(
    observed_confidence: torch.Tensor,      # (N_frames, N_hand)
    observed_anchor_face: torch.Tensor,     # (N_frames, N_hand)
    template_anchor_face: torch.Tensor,     # (N_frames, N_hand)
    reliable_frame: torch.Tensor,           # (N_frames,)
    min_expand_conf: float = 0.35,
    min_expand_contacts: int = 4,
    min_expand_overlap: float = 0.35,
    min_expand_face_agree: float = 0.6,
    max_new_frames_per_side: int = 1,
    release_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Expand a frozen reliable contact window by checking adjacent frames.

    A non-reliable frame can join only if its observed hand vertices overlap the
    current template and agree on object face ids.  Expansion is frontier-only:
    it starts from the current reliable window boundary and never jumps over a
    failed frame.
    """
    if observed_confidence.ndim != 2:
        raise ValueError("observed_confidence must have shape (N_frames, N_hand)")
    if observed_anchor_face.shape != observed_confidence.shape:
        raise ValueError("observed_anchor_face must match observed_confidence")
    if template_anchor_face.shape != observed_confidence.shape:
        raise ValueError("template_anchor_face must match observed_confidence")
    if reliable_frame.shape != observed_confidence.shape[:1]:
        raise ValueError("reliable_frame must have shape (N_frames,)")
    if release_mask is not None and release_mask.shape != observed_confidence.shape:
        raise ValueError("release_mask must match observed_confidence")

    expanded = reliable_frame.clone()
    if not expanded.any():
        return expanded

    candidate_contact = (observed_confidence >= float(min_expand_conf)) & (observed_anchor_face >= 0)
    if release_mask is not None:
        candidate_contact = candidate_contact & (~release_mask)
    candidate_frame = candidate_contact.sum(dim=1) >= int(min_expand_contacts)

    def _matches_template(frame_idx: int) -> bool:
        candidate = candidate_contact[frame_idx]
        reference = template_anchor_face[frame_idx] >= 0
        overlap = candidate & reference
        n_candidate = int(candidate.sum().item())
        n_overlap = int(overlap.sum().item())
        if n_candidate <= 0 or n_overlap <= 0:
            return False
        overlap_ratio = float(n_overlap) / float(max(n_candidate, 1))
        if overlap_ratio < float(min_expand_overlap):
            return False
        face_agree = (
            observed_anchor_face[frame_idx][overlap] == template_anchor_face[frame_idx][overlap]
        ).float().mean()
        return float(face_agree.item()) >= float(min_expand_face_agree)

    reliable_idx = expanded.nonzero(as_tuple=False).squeeze(-1)
    left = int(reliable_idx.min().item()) - 1
    right = int(reliable_idx.max().item()) + 1
    max_new = expanded.numel() if int(max_new_frames_per_side) < 0 else max(0, int(max_new_frames_per_side))
    for _ in range(max_new):
        if left < 0 or not bool(candidate_frame[left].item()) or not _matches_template(left):
            break
        expanded[left] = True
        left -= 1
    for _ in range(max_new):
        if right >= expanded.numel() or not bool(candidate_frame[right].item()) or not _matches_template(right):
            break
        expanded[right] = True
        right += 1
    return expanded


@torch.no_grad()
def propagate_grasp_template_from_reliable_frames(
    contact_confidence: torch.Tensor,       # (N_frames, N_hand)
    anchor_face: torch.Tensor,              # (N_frames, N_hand)
    anchor_bary: torch.Tensor,              # (N_frames, N_hand, 3)
    min_seed_conf: float = 0.45,
    min_seed_contacts: int = 8,
    temporal_decay: float = 0.82,
    max_steps: int = 8,
    min_keep_conf: float = 0.05,
    release_mask: Optional[torch.Tensor] = None,
    seed_confidence: Optional[torch.Tensor] = None,
    min_reliable_run: int = 1,
    expand_reliable: bool = False,
    min_expand_conf: float = 0.35,
    min_expand_contacts: int = 4,
    min_expand_overlap: float = 0.35,
    min_expand_face_agree: float = 0.6,
    bootstrap_if_no_core: bool = False,
    bootstrap_min_seed_conf: float = 0.35,
    bootstrap_min_seed_contacts: int = 4,
    bootstrap_min_reliable_run: int = 2,
    seed_frame_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Propagate a grasp template extracted only from reliable contact frames.

    Short-range memory propagation stabilizes flicker, but it still trusts the
    current geometry everywhere.  This helper first finds frames that already
    contain a coherent contact patch, then uses only those frames as seeds for
    longer temporal propagation.  The template is still keyed by MANO vertex id
    and object `face_id + bary`, so it does not hard-code a frame range.
    """
    if contact_confidence.ndim != 2:
        raise ValueError("contact_confidence must have shape (N_frames, N_hand)")
    if anchor_face.shape != contact_confidence.shape:
        raise ValueError(
            f"anchor_face must match contact_confidence shape, got {tuple(anchor_face.shape)} "
            f"vs {tuple(contact_confidence.shape)}")
    if anchor_bary.shape != contact_confidence.shape + (3,):
        raise ValueError(
            f"anchor_bary must have shape {tuple(contact_confidence.shape + (3,))}, "
            f"got {tuple(anchor_bary.shape)}")
    if release_mask is not None and release_mask.shape != contact_confidence.shape:
        raise ValueError(
            f"release_mask must match contact_confidence shape, got {tuple(release_mask.shape)} "
            f"vs {tuple(contact_confidence.shape)}")
    if seed_confidence is not None and seed_confidence.shape != contact_confidence.shape:
        raise ValueError(
            f"seed_confidence must match contact_confidence shape, got {tuple(seed_confidence.shape)} "
            f"vs {tuple(contact_confidence.shape)}")
    if seed_frame_mask is not None and seed_frame_mask.shape != contact_confidence.shape[:1]:
        raise ValueError(
            f"seed_frame_mask must have shape {tuple(contact_confidence.shape[:1])}, "
            f"got {tuple(seed_frame_mask.shape)}")

    seed_source = contact_confidence if seed_confidence is None else seed_confidence
    seed_source = seed_source.to(device=contact_confidence.device, dtype=contact_confidence.dtype).clamp(0.0, 1.0)
    valid_contact = (seed_source >= float(min_seed_conf)) & (anchor_face >= 0)
    if release_mask is not None:
        valid_contact = valid_contact & (~release_mask)
    if seed_frame_mask is not None:
        seed_frame_mask = seed_frame_mask.to(device=contact_confidence.device, dtype=torch.bool)
        valid_contact = valid_contact & seed_frame_mask[:, None]
    reliable_candidate = valid_contact.sum(dim=1) >= int(min_seed_contacts)
    reliable_frame = reliable_candidate.clone()

    def _longest_reliable_run(candidate: torch.Tensor, contact_mask: torch.Tensor,
                              min_run: int) -> torch.Tensor:
        run_frame = torch.zeros_like(candidate)
        if not candidate.any():
            return run_frame
        min_run = max(1, int(min_run))
        best_start = -1
        best_end = -1
        best_score = None
        start = None
        for idx, is_reliable in enumerate(candidate.detach().cpu().tolist()):
            if is_reliable and start is None:
                start = idx
            if (not is_reliable or idx == candidate.numel() - 1) and start is not None:
                end = idx + 1 if is_reliable and idx == candidate.numel() - 1 else idx
                if end - start >= min_run:
                    run_score = seed_source[start:end][contact_mask[start:end]].sum()
                    if (
                        best_score is None
                        or (end - start) > (best_end - best_start)
                        or ((end - start) == (best_end - best_start) and run_score > best_score)
                    ):
                        best_start = start
                        best_end = end
                        best_score = run_score
                start = None
        if best_start >= 0:
            run_frame[best_start:best_end] = True
        return run_frame

    min_reliable_run = max(1, int(min_reliable_run))
    if min_reliable_run > 1 and reliable_candidate.any():
        reliable_frame = _longest_reliable_run(
            reliable_candidate, valid_contact, min_reliable_run)

    seed_contact = valid_contact
    if bootstrap_if_no_core and not reliable_frame.any():
        bootstrap_contact = (seed_source >= float(bootstrap_min_seed_conf)) & (anchor_face >= 0)
        if release_mask is not None:
            bootstrap_contact = bootstrap_contact & (~release_mask)
        if seed_frame_mask is not None:
            bootstrap_contact = bootstrap_contact & seed_frame_mask[:, None]
        bootstrap_candidate = bootstrap_contact.sum(dim=1) >= int(bootstrap_min_seed_contacts)
        reliable_frame = _longest_reliable_run(
            bootstrap_candidate, bootstrap_contact, bootstrap_min_reliable_run)
        if reliable_frame.any():
            seed_contact = bootstrap_contact

    if expand_reliable and reliable_frame.any():
        expand_contact = (seed_source >= float(min_expand_conf)) & (anchor_face >= 0)
        if release_mask is not None:
            expand_contact = expand_contact & (~release_mask)
        if seed_frame_mask is not None:
            expand_contact = expand_contact & seed_frame_mask[:, None]
        expand_candidate = expand_contact.sum(dim=1) >= int(min_expand_contacts)

        def _frame_matches_template(frame_idx: int, ref_idx: int) -> bool:
            candidate = expand_contact[frame_idx]
            reference = expand_contact[ref_idx] | valid_contact[ref_idx]
            overlap = candidate & reference
            n_candidate = int(candidate.sum().item())
            n_overlap = int(overlap.sum().item())
            if n_candidate <= 0 or n_overlap <= 0:
                return False
            overlap_ratio = float(n_overlap) / float(max(n_candidate, 1))
            if overlap_ratio < float(min_expand_overlap):
                return False
            face_agree = (
                anchor_face[frame_idx][overlap] == anchor_face[ref_idx][overlap]
            ).float().mean()
            return float(face_agree.item()) >= float(min_expand_face_agree)

        reliable_idx = reliable_frame.nonzero(as_tuple=False).squeeze(-1)
        left = int(reliable_idx.min().item()) - 1
        right = int(reliable_idx.max().item()) + 1
        while left >= 0:
            ref = left + 1
            if not bool(expand_candidate[left].item()) or not _frame_matches_template(left, ref):
                break
            reliable_frame[left] = True
            left -= 1
        while right < reliable_frame.numel():
            ref = right - 1
            if not bool(expand_candidate[right].item()) or not _frame_matches_template(right, ref):
                break
            reliable_frame[right] = True
            right += 1
        seed_contact = valid_contact | (reliable_frame[:, None] & expand_contact)

    seed_conf = torch.where(
        reliable_frame[:, None] & seed_contact,
        seed_source,
        torch.zeros_like(contact_confidence))
    seed_face = torch.where(seed_conf > 0, anchor_face, torch.full_like(anchor_face, -1))
    seed_bary = torch.where(seed_conf.unsqueeze(-1) > 0, anchor_bary, torch.zeros_like(anchor_bary))

    prop_conf, prop_face, prop_bary = propagate_dense_contact_memory(
        seed_conf,
        seed_face,
        seed_bary,
        temporal_decay=temporal_decay,
        max_steps=max_steps,
        min_seed_conf=min_keep_conf,
        min_keep_conf=min_keep_conf,
        release_mask=release_mask,
    )

    # Return the reliable-frame template itself, not a merge with current
    # observations.  Current-frame geometry can be wrong in gap frames; letting
    # its confidence win would put the loss back on the wrong surface point.
    out_conf = prop_conf
    out_face = prop_face
    out_bary = prop_bary
    keep = out_conf >= float(min_keep_conf)
    out_conf = torch.where(keep, out_conf, torch.zeros_like(out_conf))
    out_face = torch.where(keep, out_face, torch.full_like(out_face, -1))
    out_bary = torch.where(keep.unsqueeze(-1), out_bary, torch.zeros_like(out_bary))
    return out_conf, out_face, out_bary, reliable_frame


# ------------------------------------------------------------------
# 1. Penetration (per-step, differentiable)
# ------------------------------------------------------------------

def compute_penetration_loss(
    hand_verts: torch.Tensor,         # (B, 778, 3)
    obj_verts: torch.Tensor,          # (B, V_obj, 3)
    obj_vertex_normals: torch.Tensor, # (B, V_obj, 3) outward
    clamp_depth: float = 0.04,
    dead_zone: float = 0.0,           # NEW: penetration depth tolerated for free
    exclude_hand_idx: Optional[torch.Tensor] = None,  # (K,) hand vertex idx to ignore
    detach_object: bool = False,      # if True, gradient only flows to hand_verts
) -> Tuple[torch.Tensor, dict]:
    """
    One-sided push-out loss.

    For each hand vertex v:
        nn_o = nearest obj vertex
        signed_inside = (nn_o - v) · n(nn_o)         # >0 means inside
        L_pen += relu(signed_inside - dead_zone)

    `dead_zone` (metres): penetration depths shallower than this contribute
    zero loss / zero gradient.  Useful for thin objects (handles, pens)
    where finger wrap-around inherently penetrates a few mm and the
    penetration loss would otherwise prevent the hand from getting close
    enough to grasp.  Diagnostics (n_inside, mean/max depth) still reflect
    the raw penetration so the user can see what's happening.

    Returns (loss, info_dict) where info_dict has diagnostics:
        - n_inside (mean count per frame, depth > 0)
        - n_inside_active (mean count per frame, depth > dead_zone)
        - mean_pen_depth, max_pen_depth (over all inside verts, in metres)
    """
    B, N_h, _ = hand_verts.shape

    if detach_object:
        obj_verts_for_loss = obj_verts.detach()
        obj_normals_for_loss = obj_vertex_normals.detach()
    else:
        obj_verts_for_loss = obj_verts
        obj_normals_for_loss = obj_vertex_normals

    nn_dist, nn_idx = get_NN(hand_verts, obj_verts_for_loss)        # (B, N_h)
    nn_pts = batched_index_select(obj_verts_for_loss, nn_idx)       # (B, N_h, 3)
    nn_normals = batched_index_select(obj_normals_for_loss, nn_idx) # (B, N_h, 3)

    inside_score = ((nn_pts - hand_verts) * nn_normals).sum(-1)  # (B, N_h), >0 inside
    if exclude_hand_idx is not None and exclude_hand_idx.numel() > 0:
        mask = torch.ones(N_h, dtype=torch.bool, device=hand_verts.device)
        mask[exclude_hand_idx] = False
        inside_score = inside_score * mask.float().unsqueeze(0)

    pen_per_vert = F.relu(inside_score - dead_zone).clamp(max=clamp_depth)  # (B, N_h)

    inside_mask = (inside_score > 0)
    n_inside = inside_mask.float().sum(dim=1).mean().item()
    n_inside_active = (inside_score > dead_zone).float().sum(dim=1).mean().item()
    if inside_mask.any():
        depths = inside_score[inside_mask]
        mean_pen = depths.mean().item()
        max_pen = depths.max().item()
    else:
        mean_pen = 0.0
        max_pen = 0.0

    loss = pen_per_vert.mean()
    info = {
        'n_inside': n_inside,
        'n_inside_active': n_inside_active,
        'mean_pen_depth': mean_pen,
        'max_pen_depth': max_pen,
    }
    return loss, info


# ------------------------------------------------------------------
# 2. Classification + soft correspondence (no_grad, periodic)
# ------------------------------------------------------------------

def _sample_obj_surface_with_face_bary(
    obj_verts: np.ndarray,   # (V, 3)
    obj_faces: np.ndarray,   # (F, 3)
    n_samples: int,
):
    """trimesh-based surface sampling that also returns face id + bary coords."""
    mesh = trimesh.Trimesh(vertices=obj_verts, faces=obj_faces, process=False)
    pts, face_ids = trimesh.sample.sample_surface(mesh, n_samples)
    # barycentric coords for each sample
    tri = obj_verts[obj_faces[face_ids]]                     # (n, 3, 3)
    bary = trimesh.triangles.points_to_barycentric(tri, pts) # (n, 3)
    normals = trimesh.triangles.normals(tri)[0] if False else mesh.face_normals[face_ids]
    return pts.astype(np.float32), face_ids.astype(np.int64), bary.astype(np.float32), normals.astype(np.float32)


@torch.no_grad()
def classify_and_build_correspondence(
    hand_verts: torch.Tensor,         # (N_frames, 778, 3)
    hand_normals: torch.Tensor,       # (N_frames, 778, 3)
    obj_verts: torch.Tensor,          # (N_frames, V_obj, 3)
    obj_vertex_normals: torch.Tensor, # (N_frames, V_obj, 3)
    obj_faces: torch.Tensor,          # (F, 3) shared across frames
    contact_subset_idx: Optional[torch.Tensor] = None,  # (K,) hand idx allowed to be NEAR_CONTACT
    dist_thresh: float = 0.02,
    cone_angle_deg: float = 60.0,
    n_surface_samples: int = 8000,
    topk: int = 8,
    sigma: float = 0.01,
    prev_argmax_face: Optional[torch.Tensor] = None,  # (N_frames, V_active) face id; -1 if absent
    prev_face_logit_bonus: float = 0.7,  # additive logit bonus for prev face hits
    prev_contact_confidence: Optional[torch.Tensor] = None,  # dense (N_frames, 778)
    prev_anchor_face: Optional[torch.Tensor] = None,  # dense (N_frames, 778)
    prev_anchor_bary: Optional[torch.Tensor] = None,  # dense (N_frames, 778, 3)
    prev_anchor_logit_bonus: float = 2.0,
    confidence_decay: float = 0.85,
    confidence_keep_thresh: float = 0.15,
) -> ContactCache:
    """Build per-frame classification + soft correspondence."""
    device = hand_verts.device
    N_frames, N_h, _ = hand_verts.shape
    cos_thresh = float(np.cos(np.deg2rad(cone_angle_deg)))

    # ---- 2.1 inside / outside via obj vertex normals (cheap, batched) ----
    nn_dist, nn_idx = get_NN(hand_verts, obj_verts)               # (N, N_h)
    nn_pts = batched_index_select(obj_verts, nn_idx)              # (N, N_h, 3)
    nn_normals_v = batched_index_select(obj_vertex_normals, nn_idx)
    inside_score = ((nn_pts - hand_verts) * nn_normals_v).sum(-1)  # (N, N_h)
    is_inside = inside_score > 0
    bucket = torch.full((N_frames, N_h), BUCKET_FAR, dtype=torch.int8, device=device)
    bucket[is_inside] = BUCKET_INSIDE

    # ---- 2.2 contact observation + temporal hand-contact memory ----
    dist_eu = torch.sqrt(nn_dist.clamp(min=0))  # nn_dist from get_NN is squared
    cos_alignment = -F.cosine_similarity(hand_normals, nn_normals_v, dim=-1)  # hand n vs -obj n
    # Observation confidence is strong for close, normal-consistent contacts.
    # Previous confidence decays slowly, so a trusted hand vertex does not
    # disappear just because one noisy object pose moves its nearest surface.
    dist_conf = torch.exp(-(dist_eu / max(float(dist_thresh), 1e-6)).pow(2))
    cone_conf = ((cos_alignment - cos_thresh) / max(1.0 - cos_thresh, 1e-6)).clamp(0.0, 1.0)
    geom_observed_mask = (dist_eu < dist_thresh) & (cos_alignment > cos_thresh)
    raw_geometry_conf = geom_observed_mask.float() * dist_conf * (0.5 + 0.5 * cone_conf)

    observed_mask = geom_observed_mask
    if contact_subset_idx is not None and contact_subset_idx.numel() > 0:
        subset_mask_1d = torch.zeros(N_h, dtype=torch.bool, device=device)
        subset_mask_1d[contact_subset_idx.to(device)] = True
        observed_mask = observed_mask & subset_mask_1d.unsqueeze(0)

    near_mask = (~is_inside) & observed_mask
    obs_conf = observed_mask.float() * dist_conf * (0.5 + 0.5 * cone_conf)

    if prev_contact_confidence is not None:
        if prev_contact_confidence.shape != (N_frames, N_h):
            raise ValueError(
                "prev_contact_confidence must have shape "
                f"{(N_frames, N_h)}, got {tuple(prev_contact_confidence.shape)}"
            )
        prev_conf = prev_contact_confidence.to(device=device, dtype=obs_conf.dtype)
        contact_conf_full = torch.maximum(obs_conf, prev_conf * float(confidence_decay))
    else:
        contact_conf_full = obs_conf

    active_candidate_mask = observed_mask | (contact_conf_full >= float(confidence_keep_thresh))

    bucket[near_mask] = BUCKET_NEAR_CONTACT

    # ---- 2.3 active vertex set per frame (padded across frames) ----
    n_active_per_frame = active_candidate_mask.sum(dim=1)
    V_active = int(n_active_per_frame.max().item()) if N_frames > 0 else 0
    active_idx = torch.zeros(N_frames, max(V_active, 1), dtype=torch.long, device=device)
    active_mask = torch.zeros(N_frames, max(V_active, 1), dtype=torch.bool, device=device)
    for f in range(N_frames):
        idx_f = active_candidate_mask[f].nonzero(as_tuple=False).squeeze(-1)
        n_f = idx_f.shape[0]
        if n_f > 0:
            active_idx[f, :n_f] = idx_f
            active_mask[f, :n_f] = True

    if V_active == 0:
        empty_long = torch.zeros(N_frames, 1, topk, dtype=torch.long, device=device)
        empty_bary = torch.zeros(N_frames, 1, topk, 3, dtype=torch.float32, device=device)
        empty_w = torch.zeros(N_frames, 1, topk, dtype=torch.float32, device=device)
        empty_arg = torch.full((N_frames, 1), -1, dtype=torch.long, device=device)
        empty_conf = torch.zeros(N_frames, 1, dtype=torch.float32, device=device)
        return ContactCache(
            bucket=bucket,
            signed_dist=inside_score,
            face_id_topk=empty_long,
            bary_topk=empty_bary,
            weight_topk=empty_w,
            active_idx=active_idx,
            active_mask=active_mask,
            argmax_face=empty_arg,
            contact_confidence=empty_conf,
            observed_contact_confidence=empty_conf,
            raw_observed_contact_confidence=obs_conf,
            raw_geometry_contact_confidence=raw_geometry_conf,
        )

    # ---- 2.4 surface sampling per frame (CPU, only every K steps) ----
    obj_verts_cpu = obj_verts.detach().cpu().numpy()
    obj_faces_cpu = obj_faces.detach().cpu().numpy()
    sample_pts = torch.zeros(N_frames, n_surface_samples, 3, dtype=torch.float32, device=device)
    sample_fid = torch.zeros(N_frames, n_surface_samples, dtype=torch.long, device=device)
    sample_bary = torch.zeros(N_frames, n_surface_samples, 3, dtype=torch.float32, device=device)
    sample_normals = torch.zeros(N_frames, n_surface_samples, 3, dtype=torch.float32, device=device)
    for f in range(N_frames):
        pts, fid, bary, nrm = _sample_obj_surface_with_face_bary(
            obj_verts_cpu[f], obj_faces_cpu, n_surface_samples)
        sample_pts[f] = torch.from_numpy(pts).to(device)
        sample_fid[f] = torch.from_numpy(fid).to(device)
        sample_bary[f] = torch.from_numpy(bary).to(device)
        sample_normals[f] = torch.from_numpy(nrm).to(device)

    # ---- 2.5 KNN: active hand verts -> surface samples ----
    active_hand_pts = torch.gather(
        hand_verts, 1, active_idx.unsqueeze(-1).expand(-1, -1, 3))  # (N, V_active, 3)
    knn_out = knn_points(active_hand_pts, sample_pts, K=topk, return_nn=False)
    knn_dists_sq = knn_out.dists                  # (N, V_active, K)
    knn_sample_idx = knn_out.idx                  # (N, V_active, K)

    face_id_topk = torch.gather(
        sample_fid.unsqueeze(1).expand(-1, max(V_active, 1), -1),
        2, knn_sample_idx)                        # (N, V_active, K)
    bary_topk = torch.gather(
        sample_bary.unsqueeze(1).expand(-1, max(V_active, 1), -1, -1),
        2, knn_sample_idx.unsqueeze(-1).expand(-1, -1, -1, 3))  # (N, V_active, K, 3)

    # If a same-frame object-surface anchor survived from the previous
    # reclassification, keep it as an explicit candidate. This protects a
    # stable face+bary contact point when the current object pose is noisy
    # enough that random surface sampling would miss it.
    prev_anchor_valid = None
    if prev_anchor_face is not None and prev_anchor_bary is not None and topk > 0:
        if prev_anchor_face.shape != (N_frames, N_h):
            raise ValueError(
                "prev_anchor_face must have shape "
                f"{(N_frames, N_h)}, got {tuple(prev_anchor_face.shape)}"
            )
        if prev_anchor_bary.shape != (N_frames, N_h, 3):
            raise ValueError(
                "prev_anchor_bary must have shape "
                f"{(N_frames, N_h, 3)}, got {tuple(prev_anchor_bary.shape)}"
            )
        prev_face_active = torch.gather(prev_anchor_face.to(device=device), 1, active_idx)
        prev_bary_active = torch.gather(
            prev_anchor_bary.to(device=device, dtype=bary_topk.dtype),
            1,
            active_idx.unsqueeze(-1).expand(-1, -1, 3))
        prev_anchor_valid = (prev_face_active >= 0) & active_mask
        if prev_anchor_valid.any():
            face_id_topk = face_id_topk.clone()
            bary_topk = bary_topk.clone()
            knn_dists_sq = knn_dists_sq.clone()
            face_id_topk[..., -1] = torch.where(
                prev_anchor_valid, prev_face_active, face_id_topk[..., -1])
            bary_topk[..., -1, :] = torch.where(
                prev_anchor_valid.unsqueeze(-1), prev_bary_active, bary_topk[..., -1, :])
            prev_fv_idx = obj_faces[prev_face_active.clamp(min=0)]  # (N, V_active, 3)
            prev_fv_flat = prev_fv_idx.reshape(N_frames, -1)
            prev_fv_pos_flat = torch.gather(
                obj_verts, 1, prev_fv_flat.unsqueeze(-1).expand(-1, -1, 3))
            prev_fv_pos = prev_fv_pos_flat.reshape(N_frames, V_active, 3, 3)
            prev_anchor_pts = (prev_fv_pos * prev_bary_active.unsqueeze(-1)).sum(dim=2)
            prev_anchor_dist_sq = (active_hand_pts - prev_anchor_pts).pow(2).sum(-1)
            knn_dists_sq[..., -1] = torch.where(
                prev_anchor_valid, prev_anchor_dist_sq, knn_dists_sq[..., -1])

    # ---- 2.6 softmax weights with prev-face / anchor sticky bonuses ----
    sigma2 = sigma * sigma
    logits = -knn_dists_sq / max(sigma2, 1e-12)   # (N, V_active, K)
    if prev_argmax_face is not None and prev_face_logit_bonus > 0:
        # Match prev_argmax_face onto current per-frame active vertex layout.
        # prev_argmax_face shape: (N_frames, V_active_prev). Sticky bonus only
        # applies where current active idx matches prev active idx (same hand vertex).
        # Caller is responsible for aligning by hand vertex id; here we accept a
        # dense (N_frames, N_h) lookup by passing -1 for absent verts.
        if prev_argmax_face.dim() == 2 and prev_argmax_face.shape[1] == N_h:
            prev_face_per_active = torch.gather(prev_argmax_face, 1, active_idx)  # (N, V_active)
            match = (face_id_topk == prev_face_per_active.unsqueeze(-1))         # (N, V_active, K)
            logits = logits + match.float() * prev_face_logit_bonus
    if prev_anchor_valid is not None and prev_anchor_logit_bonus > 0:
        logits[..., -1] = logits[..., -1] + prev_anchor_valid.float() * float(prev_anchor_logit_bonus)

    # mask out padding (active_mask=False) before softmax to avoid NaN propagation
    logits = logits.masked_fill(~active_mask.unsqueeze(-1), -1e9)
    weight_topk = torch.softmax(logits, dim=-1)   # (N, V_active, K)
    weight_topk = weight_topk * active_mask.unsqueeze(-1).float()

    argmax_face_active = face_id_topk.gather(2, weight_topk.argmax(dim=-1, keepdim=True)).squeeze(-1)
    argmax_face_active = torch.where(active_mask, argmax_face_active, torch.full_like(argmax_face_active, -1))
    contact_conf_active = torch.gather(contact_conf_full, 1, active_idx) * active_mask.float()
    observed_conf_active = torch.gather(obs_conf, 1, active_idx) * active_mask.float()

    return ContactCache(
        bucket=bucket,
        signed_dist=inside_score,
        face_id_topk=face_id_topk,
        bary_topk=bary_topk,
        weight_topk=weight_topk,
        active_idx=active_idx,
        active_mask=active_mask,
        argmax_face=argmax_face_active,
        contact_confidence=contact_conf_active,
        observed_contact_confidence=observed_conf_active,
        raw_observed_contact_confidence=obs_conf,
        raw_geometry_contact_confidence=raw_geometry_conf,
    )


# ------------------------------------------------------------------
# 3. Soft contact loss (per-step, differentiable)
# ------------------------------------------------------------------

def compute_soft_contact_loss(
    hand_verts: torch.Tensor,    # (N_frames, 778, 3)
    obj_verts: torch.Tensor,     # (N_frames, V_obj, 3)
    obj_faces: torch.Tensor,     # (F, 3)
    cache: ContactCache,
    detach_object: bool = False,
    obj_grad_scale=1.0,          # float OR sequence of 3 floats (per-axis)
) -> Tuple[torch.Tensor, dict]:
    """L_contact = mean over (frame, active_vert) of  Σ_k w_k * ‖v - bary_pt_k‖².

    `obj_grad_scale`: scales the gradient that flows into obj_verts.
    Can be either a scalar (isotropic) or a length-3 sequence (per-axis).

        1.0          — symmetric.  Hand and object share the pull.
        0.0          — fully detached.  Object frozen w.r.t. contact.
        (0.0, 0.0, 1.0) — xy detached, z free.  Useful when silhouette
                          loss already constrains object xy but doesn't
                          touch z (depth) — contact can then freely
                          pull the object in depth without dragging it
                          off its mask alignment.

    `detach_object=True` is a hard alias for obj_grad_scale=0.0 (kept
    for backward compatibility).
    """
    if cache.weight_topk.numel() == 0 or not cache.active_mask.any():
        return (torch.tensor(0.0, device=hand_verts.device, requires_grad=True),
                {'n_active': 0, 'mean_contact_dist': 0.0})

    N_frames, V_active, K = cache.face_id_topk.shape
    # Gather hand active verts (with grad)
    hand_active = torch.gather(
        hand_verts, 1, cache.active_idx.unsqueeze(-1).expand(-1, -1, 3))  # (N, V_active, 3)

    # Resolve obj-grad routing.
    # obj_for_loss = obj.detach() + α·(obj - obj.detach())  →  d/dobj = α
    # (α can be (3,) for per-axis scaling; broadcasts over (B, V, 3).)
    if detach_object:
        obj_verts_for_loss = obj_verts.detach()
    elif isinstance(obj_grad_scale, (int, float)):
        s = float(obj_grad_scale)
        if s <= 0.0:
            obj_verts_for_loss = obj_verts.detach()
        elif s >= 1.0:
            obj_verts_for_loss = obj_verts
        else:
            obj_verts_for_loss = obj_verts.detach() + s * (obj_verts - obj_verts.detach())
    else:
        # Per-axis (length-3 sequence/tensor).
        s_axis = torch.as_tensor(
            obj_grad_scale, dtype=obj_verts.dtype, device=obj_verts.device)
        assert s_axis.numel() == 3, f"obj_grad_scale must be scalar or length-3, got {tuple(s_axis.shape)}"
        obj_verts_for_loss = obj_verts.detach() + s_axis * (obj_verts - obj_verts.detach())
    face_verts_idx = obj_faces[cache.face_id_topk]  # (N, V_active, K, 3) of vert idx
    # gather vertex positions
    fv_idx_flat = face_verts_idx.reshape(N_frames, -1)  # (N, V_active*K*3)
    fv_pos_flat = torch.gather(
        obj_verts_for_loss, 1, fv_idx_flat.unsqueeze(-1).expand(-1, -1, 3))  # (N, V_active*K*3, 3)
    fv_pos = fv_pos_flat.reshape(N_frames, V_active, K, 3, 3)  # (N, V_active, K, 3 verts, 3 xyz)
    bary_pts = (fv_pos * cache.bary_topk.unsqueeze(-1)).sum(dim=3)  # (N, V_active, K, 3)

    diff = hand_active.unsqueeze(2) - bary_pts                       # (N, V_active, K, 3)
    sq = diff.pow(2).sum(-1)                                         # (N, V_active, K)
    weighted = (sq * cache.weight_topk).sum(-1)                      # (N, V_active)
    mask_f = cache.active_mask.float()
    if cache.contact_confidence is None:
        contact_weight = mask_f
    else:
        contact_weight = cache.contact_confidence.to(weighted.dtype).clamp(0.0, 1.0) * mask_f
    n_valid = contact_weight.sum().clamp(min=1.0)
    loss = (weighted * contact_weight).sum() / n_valid

    with torch.no_grad():
        # diagnostic: argmax-face distance per active vert
        argmax_k = cache.weight_topk.argmax(dim=-1, keepdim=True)         # (N, V_active, 1)
        amax_dist_sq = sq.gather(2, argmax_k).squeeze(-1)                  # (N, V_active)
        d = torch.sqrt(amax_dist_sq.clamp(min=0))
        mean_contact_dist = (d * contact_weight).sum().item() / float(n_valid.item())
        n_active = int((contact_weight > 1e-4).sum().item())

    info = {
        'n_active': n_active,
        'mean_contact_dist': mean_contact_dist,
    }
    return loss, info


def compute_confident_gap_closing_loss(
    hand_verts: torch.Tensor,    # (N_frames, 778, 3)
    obj_verts: torch.Tensor,     # (N_frames, V_obj, 3)
    obj_faces: torch.Tensor,     # (F, 3)
    cache: ContactCache,
    high_conf_thresh: float = 0.7,
    gap_start: float = 0.008,
    gap_clamp: float = 0.06,
    obj_grad_scale=(0.1, 0.1, 1.0),
) -> Tuple[torch.Tensor, dict]:
    """Robust attraction for high-confidence contacts with visible gaps.

    Unlike the sticky contact loss, this term only activates when the current
    hand-anchor distance is larger than `gap_start`.  It is intended as a
    conservative pull-back for propagated contacts, with object gradients
    scaled separately from the regular contact loss.
    """
    if cache.weight_topk.numel() == 0 or not cache.active_mask.any() or cache.contact_confidence is None:
        return (torch.tensor(0.0, device=hand_verts.device, requires_grad=True),
                {'n_gap_active': 0, 'mean_gap_dist': 0.0})

    N_frames, V_active, K = cache.face_id_topk.shape
    hand_active = torch.gather(
        hand_verts, 1, cache.active_idx.unsqueeze(-1).expand(-1, -1, 3))

    s_axis = torch.as_tensor(obj_grad_scale, dtype=obj_verts.dtype, device=obj_verts.device)
    if s_axis.numel() == 1:
        s_axis = s_axis.expand(3)
    assert s_axis.numel() == 3, f"obj_grad_scale must be scalar or length-3, got {tuple(s_axis.shape)}"
    obj_verts_for_loss = obj_verts.detach() + s_axis * (obj_verts - obj_verts.detach())

    face_verts_idx = obj_faces[cache.face_id_topk]
    fv_idx_flat = face_verts_idx.reshape(N_frames, -1)
    fv_pos_flat = torch.gather(
        obj_verts_for_loss, 1, fv_idx_flat.unsqueeze(-1).expand(-1, -1, 3))
    fv_pos = fv_pos_flat.reshape(N_frames, V_active, K, 3, 3)
    bary_pts = (fv_pos * cache.bary_topk.unsqueeze(-1)).sum(dim=3)

    diff = hand_active.unsqueeze(2) - bary_pts
    dist = torch.sqrt(diff.pow(2).sum(-1).clamp(min=1e-12))
    weighted_dist = (dist * cache.weight_topk).sum(-1)

    conf = cache.contact_confidence.to(weighted_dist.dtype).clamp(0.0, 1.0)
    active = cache.active_mask & (conf >= float(high_conf_thresh)) & (weighted_dist > float(gap_start))
    active_f = active.float()
    n_active = active_f.sum().clamp(min=1.0)

    excess = (weighted_dist - float(gap_start)).clamp(min=0.0, max=float(gap_clamp))
    loss = (excess.pow(2) * conf * active_f).sum() / n_active

    with torch.no_grad():
        if active.any():
            mean_gap = (weighted_dist * active_f).sum().item() / float(n_active.item())
            n_gap = int(active.sum().item())
        else:
            mean_gap = 0.0
            n_gap = 0

    return loss, {'n_gap_active': n_gap, 'mean_gap_dist': mean_gap}


def compute_grasp_template_anchor_loss(
    hand_verts: torch.Tensor,        # (N_frames, 778, 3)
    obj_verts: torch.Tensor,         # (N_frames, V_obj, 3)
    obj_faces: torch.Tensor,         # (F, 3)
    template_confidence: torch.Tensor,   # dense (N_frames, 778)
    template_anchor_face: torch.Tensor,  # dense (N_frames, 778)
    template_anchor_bary: torch.Tensor,  # dense (N_frames, 778, 3)
    high_conf_thresh: float = 0.45,
    gap_start: float = 0.006,
    gap_clamp: float = 0.08,
    obj_grad_scale=(0.0, 0.0, 1.0),
) -> Tuple[torch.Tensor, dict]:
    """Direct grasp-template pull using dense propagated object anchors.

    This bypasses current-frame KNN/top-k competition: if a reliable frame
    propagated a hand-vertex/object-surface anchor, the loss pulls that hand
    vertex toward the exact `face_id + bary` point on the current object pose.
    """
    device = hand_verts.device
    if template_confidence.shape != hand_verts.shape[:2]:
        raise ValueError(
            "template_confidence must match hand_verts first two dims, got "
            f"{tuple(template_confidence.shape)} vs {tuple(hand_verts.shape[:2])}")
    if template_anchor_face.shape != hand_verts.shape[:2]:
        raise ValueError(
            "template_anchor_face must match hand_verts first two dims, got "
            f"{tuple(template_anchor_face.shape)} vs {tuple(hand_verts.shape[:2])}")
    if template_anchor_bary.shape != hand_verts.shape:
        raise ValueError(
            "template_anchor_bary must match hand_verts shape, got "
            f"{tuple(template_anchor_bary.shape)} vs {tuple(hand_verts.shape)}")

    conf = template_confidence.to(device=device, dtype=hand_verts.dtype).clamp(0.0, 1.0)
    anchor_face = template_anchor_face.to(device=device)
    anchor_bary = template_anchor_bary.to(device=device, dtype=hand_verts.dtype)
    active = (conf >= float(high_conf_thresh)) & (anchor_face >= 0)
    if not active.any():
        return (torch.tensor(0.0, device=device, requires_grad=True),
                {'n_template_active': 0,
                 'mean_template_gap_mm': 0.0,
                 'mean_template_conf': 0.0,
                 'template_active_count': torch.zeros(hand_verts.shape[0], device=device),
                 'template_gap_mean': torch.zeros(hand_verts.shape[0], device=device),
                 'template_conf_mean': torch.zeros(hand_verts.shape[0], device=device)})

    s_axis = torch.as_tensor(obj_grad_scale, dtype=obj_verts.dtype, device=obj_verts.device)
    if s_axis.numel() == 1:
        s_axis = s_axis.expand(3)
    assert s_axis.numel() == 3, f"obj_grad_scale must be scalar or length-3, got {tuple(s_axis.shape)}"
    obj_verts_for_loss = obj_verts.detach() + s_axis * (obj_verts - obj_verts.detach())

    N_frames, N_hand, _ = hand_verts.shape
    face_verts_idx = obj_faces[anchor_face.clamp(min=0)]  # (N, N_hand, 3)
    fv_idx_flat = face_verts_idx.reshape(N_frames, -1)
    fv_pos_flat = torch.gather(
        obj_verts_for_loss, 1, fv_idx_flat.unsqueeze(-1).expand(-1, -1, 3))
    fv_pos = fv_pos_flat.reshape(N_frames, N_hand, 3, 3)
    anchor_pts = (fv_pos * anchor_bary.unsqueeze(-1)).sum(dim=2)

    dist = torch.sqrt((hand_verts - anchor_pts).pow(2).sum(-1).clamp(min=1e-12))
    active = active & (dist > float(gap_start))
    active_f = active.float()
    n_active = active_f.sum().clamp(min=1.0)
    excess = (dist - float(gap_start)).clamp(min=0.0, max=float(gap_clamp))
    loss = (excess.pow(2) * conf * active_f).sum() / n_active

    with torch.no_grad():
        frame_count = active_f.sum(dim=1)
        frame_denom = frame_count.clamp(min=1.0)
        frame_gap_mean = (dist * active_f).sum(dim=1) / frame_denom
        frame_conf_mean = (conf * active_f).sum(dim=1) / frame_denom
        has_frame_contact = frame_count > 0
        frame_gap_mean = torch.where(has_frame_contact, frame_gap_mean, torch.zeros_like(frame_gap_mean))
        frame_conf_mean = torch.where(has_frame_contact, frame_conf_mean, torch.zeros_like(frame_conf_mean))
        if active.any():
            mean_gap_mm = (dist * active_f).sum().item() / float(n_active.item()) * 1000.0
            mean_conf = (conf * active_f).sum().item() / float(n_active.item())
            n_template = int(active.sum().item())
        else:
            mean_gap_mm = 0.0
            mean_conf = 0.0
            n_template = 0

    return loss, {
        'n_template_active': n_template,
        'mean_template_gap_mm': mean_gap_mm,
        'mean_template_conf': mean_conf,
        'template_active_count': frame_count.detach(),
        'template_gap_mean': frame_gap_mean.detach(),
        'template_conf_mean': frame_conf_mean.detach(),
    }


def compute_neighbor_template_bridge_loss(
    hand_verts: torch.Tensor,
    obj_verts: torch.Tensor,
    obj_faces: torch.Tensor,
    template_confidence: torch.Tensor,
    template_anchor_face: torch.Tensor,
    template_anchor_bary: torch.Tensor,
    reliable_frames: torch.Tensor,
    max_bridge_distance: int = 6,
    high_conf_thresh: float = 0.05,
    gap_start: float = 0.006,
    gap_clamp: float = 0.06,
    frame_weight_gamma: float = 1.5,
    obj_grad_scale=(0.05, 0.05, 0.5),
) -> Tuple[torch.Tensor, dict]:
    """Template-guided attraction for non-reliable frames adjacent to reliable ones.

    Reliable frames already receive the normal grasp-template anchor loss.  This
    bridge is intentionally one-way: it lets the frozen reliable template pull
    nearby missing frames, but never promotes those frames into the template.
    """
    device = hand_verts.device
    N_frames = hand_verts.shape[0]
    if reliable_frames.shape[0] != N_frames:
        raise ValueError(
            "reliable_frames must match hand_verts first dim, got "
            f"{tuple(reliable_frames.shape)} vs {tuple(hand_verts.shape[:1])}")

    zero_info = {
        'n_bridge_active': 0,
        'n_bridge_frames': 0,
        'mean_bridge_gap_mm': 0.0,
        'mean_bridge_conf': 0.0,
        'bridge_frame_weight': torch.zeros(N_frames, device=device),
    }
    if max_bridge_distance <= 0:
        return torch.tensor(0.0, device=device, requires_grad=True), zero_info

    reliable = reliable_frames.to(device=device, dtype=torch.bool)
    if not reliable.any():
        return torch.tensor(0.0, device=device, requires_grad=True), zero_info

    frame_idx = torch.arange(N_frames, device=device)
    reliable_idx = torch.nonzero(reliable, as_tuple=False).flatten()
    nearest_dist = (frame_idx[:, None] - reliable_idx[None, :]).abs().min(dim=1).values
    bridge_frame = (~reliable) & (nearest_dist <= int(max_bridge_distance))
    bridge_weight = torch.zeros(N_frames, device=device, dtype=hand_verts.dtype)
    bridge_weight[bridge_frame] = (
        (float(max_bridge_distance) + 1.0 - nearest_dist[bridge_frame].to(hand_verts.dtype)) /
        max(float(max_bridge_distance), 1.0)
    ).clamp(0.0, 1.0).pow(float(frame_weight_gamma))

    conf = template_confidence.to(device=device, dtype=hand_verts.dtype)
    weighted_conf = conf * bridge_weight[:, None]
    loss, info = compute_grasp_template_anchor_loss(
        hand_verts,
        obj_verts,
        obj_faces,
        weighted_conf,
        template_anchor_face,
        template_anchor_bary,
        high_conf_thresh=high_conf_thresh,
        gap_start=gap_start,
        gap_clamp=gap_clamp,
        obj_grad_scale=obj_grad_scale,
    )

    with torch.no_grad():
        valid_template_frame = ((template_anchor_face.to(device=device) >= 0) & (conf > 0.0)).any(dim=1)
        n_bridge_frames = int((bridge_frame & valid_template_frame).sum().item())

    return loss, {
        'n_bridge_active': int(info.get('n_template_active', 0)),
        'n_bridge_frames': n_bridge_frames,
        'mean_bridge_gap_mm': float(info.get('mean_template_gap_mm', 0.0)),
        'mean_bridge_conf': float(info.get('mean_template_conf', 0.0)),
        'bridge_frame_weight': bridge_weight.detach(),
        'bridge_active_count': info.get('template_active_count'),
        'bridge_gap_mean': info.get('template_gap_mean'),
        'bridge_conf_mean': info.get('template_conf_mean'),
    }


def compute_contact_patch_centroid_loss(
    hand_verts: torch.Tensor,    # (N_frames, 778, 3)
    obj_verts: torch.Tensor,     # (N_frames, V_obj, 3)
    obj_faces: torch.Tensor,     # (F, 3)
    cache: ContactCache,
    high_conf_thresh: float = 0.45,
    min_contacts: int = 6,
    gap_start: float = 0.006,
    gap_clamp: float = 0.08,
    obj_grad_scale=(0.05, 0.05, 1.0),
) -> Tuple[torch.Tensor, dict]:
    """Frame-level contact patch attraction.

    High-confidence point correspondences can be noisy individually.  This
    loss first aggregates them into a hand contact-patch centroid and an
    object surface-patch centroid per frame, then closes the patch-level gap.
    """
    if cache.weight_topk.numel() == 0 or not cache.active_mask.any() or cache.contact_confidence is None:
        return (torch.tensor(0.0, device=hand_verts.device, requires_grad=True),
                {'n_patch_active': 0, 'mean_patch_dist': 0.0, 'mean_patch_contacts': 0.0})

    N_frames, V_active, K = cache.face_id_topk.shape
    hand_active = torch.gather(
        hand_verts, 1, cache.active_idx.unsqueeze(-1).expand(-1, -1, 3))

    s_axis = torch.as_tensor(obj_grad_scale, dtype=obj_verts.dtype, device=obj_verts.device)
    if s_axis.numel() == 1:
        s_axis = s_axis.expand(3)
    assert s_axis.numel() == 3, f"obj_grad_scale must be scalar or length-3, got {tuple(s_axis.shape)}"
    obj_verts_for_loss = obj_verts.detach() + s_axis * (obj_verts - obj_verts.detach())

    face_verts_idx = obj_faces[cache.face_id_topk]
    fv_idx_flat = face_verts_idx.reshape(N_frames, -1)
    fv_pos_flat = torch.gather(
        obj_verts_for_loss, 1, fv_idx_flat.unsqueeze(-1).expand(-1, -1, 3))
    fv_pos = fv_pos_flat.reshape(N_frames, V_active, K, 3, 3)
    bary_pts = (fv_pos * cache.bary_topk.unsqueeze(-1)).sum(dim=3)
    obj_contact_pts = (bary_pts * cache.weight_topk.unsqueeze(-1)).sum(dim=2)

    conf = cache.contact_confidence.to(hand_active.dtype).clamp(0.0, 1.0)
    active = cache.active_mask & (conf >= float(high_conf_thresh))
    weights = conf * active.float()
    sum_w = weights.sum(dim=1).clamp(min=1e-8)
    n_contacts = active.float().sum(dim=1)

    hand_centroid = (hand_active * weights.unsqueeze(-1)).sum(dim=1) / sum_w.unsqueeze(-1)
    obj_centroid = (obj_contact_pts * weights.unsqueeze(-1)).sum(dim=1) / sum_w.unsqueeze(-1)
    patch_dist = torch.sqrt((hand_centroid - obj_centroid).pow(2).sum(-1).clamp(min=1e-12))

    valid = (n_contacts >= int(min_contacts)) & (patch_dist > float(gap_start))
    valid_f = valid.float()
    n_valid = valid_f.sum().clamp(min=1.0)
    excess = (patch_dist - float(gap_start)).clamp(min=0.0, max=float(gap_clamp))
    loss = (excess.pow(2) * valid_f).sum() / n_valid

    with torch.no_grad():
        if valid.any():
            mean_dist = (patch_dist * valid_f).sum().item() / float(n_valid.item())
            mean_contacts = (n_contacts * valid_f).sum().item() / float(n_valid.item())
            n_patch = int(valid.sum().item())
        else:
            mean_dist = 0.0
            mean_contacts = 0.0
            n_patch = 0

    return loss, {
        'n_patch_active': n_patch,
        'mean_patch_dist': mean_dist,
        'mean_patch_contacts': mean_contacts,
    }


# ------------------------------------------------------------------
# 3b. Temporal lock — mode-filter on argmax_face across a time window
#     to eliminate per-frame contact target jitter (root cause of
#     wrist jitter when wrist params are unfrozen).
# ------------------------------------------------------------------

@torch.no_grad()
def temporal_lock_argmax(
    cache: ContactCache,
    window: int = 2,
    min_consensus: int = 3,
) -> ContactCache:
    """For each (frame t, active hand vert v), look at argmax_face in the
    temporal window [t-window, t+window].  If a single face id appears
    >= min_consensus times AND is one of the current frame's topk faces,
    lock weight_topk[t, v, :] to one-hot on that face.

    This keeps the contact target on a *single, temporally consistent face*
    across the local time window, so wrist gradients no longer get pulled
    in different directions on neighboring frames.

    Returns a NEW ContactCache (only weight_topk modified; everything else
    shared).
    """
    if cache.weight_topk.numel() == 0 or not cache.active_mask.any():
        return cache

    device = cache.face_id_topk.device
    N_frames, V_active, K = cache.face_id_topk.shape

    # ---- 1. dense (N, 778) argmax_face map (-1 where vertex inactive) ----
    full = torch.full((N_frames, 778), -1, dtype=torch.long, device=device)
    safe_amax = torch.where(cache.active_mask, cache.argmax_face,
                            torch.full_like(cache.argmax_face, -1))
    full.scatter_(1, cache.active_idx, safe_amax)

    # ---- 2. for each (t, v): find mode in [t-W, t+W] excluding -1 ----
    # Build a (2W+1, N, 778) stack of shifted maps.
    W = int(window)
    shifts = list(range(-W, W + 1))                           # length 2W+1
    stack = torch.stack([
        torch.roll(full, shifts=s, dims=0) for s in shifts
    ], dim=0)                                                 # (2W+1, N, 778)
    # Zero out wraparound at the boundaries (set to -1).
    for i, s in enumerate(shifts):
        if s > 0:
            stack[i, :s] = -1
        elif s < 0:
            stack[i, s:] = -1

    # Mode over the time-window dim. We need the most-common non-negative
    # face id per (n, v). Implementation: for every face id present in the
    # column, count occurrences; pick max-count id with count >= min_consensus.
    # Vectorised exact mode is awkward; do it in a small loop over the
    # unique face ids that actually appear in `stack` per frame to keep
    # memory bounded.
    consensus_face = torch.full((N_frames, 778), -1, dtype=torch.long, device=device)

    # Reshape: (T_win, N, V) -> (N, V, T_win)
    stack_nv = stack.permute(1, 2, 0).contiguous()           # (N, 778, T_win)
    # For each (n, v) compute mode by sorting then run-length finding the
    # longest consecutive equal-id run. Ignore -1 entries at the start.
    sorted_ids, _ = torch.sort(stack_nv, dim=-1)             # (N, 778, T_win)
    # Build mask of "is this entry equal to its left neighbour" — runs.
    eq_left = torch.zeros_like(sorted_ids, dtype=torch.bool)
    eq_left[..., 1:] = (sorted_ids[..., 1:] == sorted_ids[..., :-1])
    # Run-length via cumulative sum over the eq_left mask — same trick as
    # numpy run-length, but adapted: each new run starts where eq_left=False,
    # so run_id = cumsum(~eq_left) along last dim, then we count per-run.
    new_run = (~eq_left).long()                              # (N, 778, T_win)
    run_id = new_run.cumsum(dim=-1)                          # 1..n_runs
    # For each entry, count of items in its run = total minus runs above with
    # run_id > my run_id starting after my position. Simpler: per-(n,v) bincount.
    # Use scatter_add_ with one-hot of run_id for tiny T_win (<=11).
    T_win = sorted_ids.shape[-1]
    # one-hot run_id over [1, T_win], shape (N, 778, T_win, T_win)
    # but that's heavy; instead compute counts by iterating run boundaries.
    # T_win is small (5 by default), do a simple python loop over run lengths.
    # Per-(n,v) we just need: longest run that corresponds to a face id != -1.
    NV = N_frames * 778
    sorted_flat = sorted_ids.view(NV, T_win)
    new_run_flat = new_run.view(NV, T_win)

    best_face = torch.full((NV,), -1, dtype=torch.long, device=device)
    best_count = torch.zeros(NV, dtype=torch.long, device=device)

    # Walk the small T_win loop — for each position track current run length.
    cur_face = sorted_flat[:, 0]                              # (NV,)
    cur_len  = torch.ones(NV, dtype=torch.long, device=device)
    for j in range(1, T_win):
        same = (sorted_flat[:, j] == cur_face)
        cur_len = torch.where(same, cur_len + 1, torch.ones_like(cur_len))
        cur_face = torch.where(same, cur_face, sorted_flat[:, j])
        # candidate update
        valid = (cur_face >= 0)
        better = valid & (cur_len > best_count)
        best_face = torch.where(better, cur_face, best_face)
        best_count = torch.where(better, cur_len, best_count)
    # also consider j=0 (single-element run)
    valid0 = (sorted_flat[:, 0] >= 0)
    init_better = valid0 & (best_count == 0)
    best_face = torch.where(init_better, sorted_flat[:, 0], best_face)
    best_count = torch.where(init_better, torch.ones_like(best_count), best_count)

    # consensus only if best_count >= min_consensus
    keep = best_count >= int(min_consensus)
    best_face = torch.where(keep, best_face, torch.full_like(best_face, -1))
    consensus_face = best_face.view(N_frames, 778)

    # ---- 3. for each active vert, check if consensus face is in topk ----
    # consensus_face_per_active: (N, V_active)
    consensus_per_active = torch.gather(consensus_face, 1, cache.active_idx)  # (N, V_active)
    # locate position k where face_id_topk == consensus
    match_kk = (cache.face_id_topk == consensus_per_active.unsqueeze(-1))     # (N, V_active, K)
    has_match = match_kk.any(dim=-1) & (consensus_per_active >= 0) & cache.active_mask  # (N, V_active)
    # one-hot weight on the matched k (use first match)
    match_idx = match_kk.float().argmax(dim=-1)                               # (N, V_active)

    # Build new weight_topk: copy original; override locked rows.
    new_weight = cache.weight_topk.clone()
    if has_match.any():
        zeros_K = torch.zeros(K, device=device)
        # mask out the locked rows
        new_weight[has_match] = zeros_K
        # gather indices we need to set to 1.0
        n_idx, v_idx = torch.nonzero(has_match, as_tuple=True)
        k_idx = match_idx[n_idx, v_idx]
        new_weight[n_idx, v_idx, k_idx] = 1.0

    new_argmax = cache.face_id_topk.gather(2, new_weight.argmax(dim=-1, keepdim=True)).squeeze(-1)
    new_argmax = torch.where(cache.active_mask, new_argmax, torch.full_like(new_argmax, -1))

    return ContactCache(
        bucket=cache.bucket,
        signed_dist=cache.signed_dist,
        face_id_topk=cache.face_id_topk,
        bary_topk=cache.bary_topk,
        weight_topk=new_weight,
        active_idx=cache.active_idx,
        active_mask=cache.active_mask,
        argmax_face=new_argmax,
        contact_confidence=cache.contact_confidence,
        observed_contact_confidence=cache.observed_contact_confidence,
        raw_observed_contact_confidence=cache.raw_observed_contact_confidence,
        raw_geometry_contact_confidence=cache.raw_geometry_contact_confidence,
    )


# ------------------------------------------------------------------
# 4. Bary smoothness (temporal stickiness on same face)
# ------------------------------------------------------------------

def compute_bary_smoothness_loss(
    cache_t: ContactCache,
    cache_tm1: ContactCache,
) -> torch.Tensor:
    """
    For each (frame, hand_vertex) that appears as NEAR_CONTACT in both
    `cache_t` and `cache_tm1` and whose argmax_face matches between frames,
    penalise change in bary coords.

    Both caches are expected to share the same N_frames timeline; we use the
    same frame index (inter-step within an optimisation; not adjacent video
    frames). Use this only between two successive *reclassification* snapshots
    of the same frame.
    """
    if cache_t.weight_topk.numel() == 0 or cache_tm1.weight_topk.numel() == 0:
        return torch.tensor(0.0, device=cache_t.bary_topk.device)

    # NOTE: this is a lightweight regulariser used only when the caller wants to
    # discourage bary jumps after a reclassification. In the pipeline below we
    # call it across reclassification snapshots, not video frames, so the
    # implementation uses argmax bary directly.
    N1 = cache_t.face_id_topk.shape[0]
    N2 = cache_tm1.face_id_topk.shape[0]
    if N1 != N2:
        return torch.tensor(0.0, device=cache_t.bary_topk.device)

    am_t = cache_t.argmax_face                # (N, V_active_t)
    am_p = cache_tm1.argmax_face              # (N, V_active_p)
    # Align on hand-vertex identity via active_idx.
    # Build a (N, 778) map: hand vertex -> argmax face (or -1)
    device = cache_t.bary_topk.device
    Nf = N1
    map_t = torch.full((Nf, 778), -1, dtype=torch.long, device=device)
    map_p = torch.full((Nf, 778), -1, dtype=torch.long, device=device)
    map_t.scatter_(1, cache_t.active_idx, torch.where(cache_t.active_mask, am_t, torch.full_like(am_t, -1)))
    map_p.scatter_(1, cache_tm1.active_idx, torch.where(cache_tm1.active_mask, am_p, torch.full_like(am_p, -1)))
    same = (map_t == map_p) & (map_t >= 0)
    if not same.any():
        return torch.tensor(0.0, device=device)

    # Bary at argmax for both caches, broadcast onto (N, 778, 3)
    def _argmax_bary_to_full(cache: ContactCache) -> torch.Tensor:
        ak = cache.weight_topk.argmax(dim=-1, keepdim=True)             # (N, V_a, 1)
        b = cache.bary_topk.gather(2, ak.unsqueeze(-1).expand(-1, -1, -1, 3)).squeeze(2)  # (N, V_a, 3)
        full = torch.zeros(Nf, 778, 3, device=device)
        full.scatter_(1, cache.active_idx.unsqueeze(-1).expand(-1, -1, 3), b)
        return full

    full_t = _argmax_bary_to_full(cache_t)
    full_p = _argmax_bary_to_full(cache_tm1)
    diff = (full_t - full_p) * same.unsqueeze(-1).float()
    return diff.pow(2).sum() / same.float().sum().clamp(min=1.0)
