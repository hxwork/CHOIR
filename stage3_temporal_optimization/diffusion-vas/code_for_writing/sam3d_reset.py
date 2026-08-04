"""SAM3D rotation-reset helpers for diffusion-vas.

Contains:
  - SAM3D import bootstrapping and availability flags
  - TemporalController / _TemporalSolver / _temporal_hook
  - Sam3dResetController
  - calculate_iou
  - probe_sam3d_rotation
"""

import contextlib
import math
import os
import sys
import logging

import numpy as np
import torch
from pytorch3d.renderer import (
    MeshRasterizer,
    MeshRenderer,
    RasterizationSettings,
    SoftSilhouetteShader,
    TexturesVertex,
)
from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.structures import Meshes
from pytorch3d.transforms import Transform3d

# ---------------------------------------------------------------------------
# SAM3D in-process integration (rotation reset for PnP)
#
# We import sam-3d-objects from a local checkout. The same conda env is used
# (see project README), so this is just a sys.path append. Imports are
# wrapped in try/except so this script still runs unchanged when SAM3D is
# unavailable (the reset feature simply stays disabled).
# ---------------------------------------------------------------------------

SAM3D_REPO = os.environ.get(
    "SAM3D_REPO",
    "../../stage1_preprocess/sam-3d-objects",
)
SAM3D_CONFIG_PATH = os.environ.get(
    "SAM3D_CONFIG_PATH",
    os.path.join(SAM3D_REPO, "checkpoints/hf/pipeline.yaml"),
)

_SAM3D_AVAILABLE = False
_SAM3D_IMPORT_ERROR = None
Sam3dInference = None
_Sam3dODESolver = object  # placeholder so solver subclassing below doesn't crash

try:
    if SAM3D_REPO not in sys.path:
        sys.path.insert(0, SAM3D_REPO)
    _sam3d_notebook = os.path.join(SAM3D_REPO, "notebook")
    if _sam3d_notebook not in sys.path:
        sys.path.insert(0, _sam3d_notebook)
    from inference import Inference as Sam3dInference  # type: ignore  # noqa: E402
    from sam3d_objects.model.backbone.generator.flow_matching.solver import \
        ODESolver as _Sam3dODESolver  # type: ignore  # noqa: E402
    _SAM3D_AVAILABLE = True
except Exception as _e:
    _SAM3D_IMPORT_ERROR = _e
    print(f"[SAM3D-RESET] SAM3D import failed ({_e}); reset feature will be disabled.")


# Keys are the names used in SparseStructureFlowTdfyWrapper.latent_mapping
# (mirrors demo_proj_temporal.py FREEZE_KEYS / GUIDE_KEYS).
_SAM3D_FREEZE_KEYS = ("shape", "scale", "translation_scale")
_SAM3D_GUIDE_KEYS = ("6drotation_normalized", "translation")


@contextlib.contextmanager
def _suppress_sam3d_logs(enabled: bool = True):
    """Temporarily silence SAM3D's stdout/stderr/loguru noise.

    We keep this scoped to SAM3D calls so diffusion-vas' own diagnostic prints
    still appear. Exceptions still propagate; only third-party chatter is hidden.
    """
    if not enabled:
        yield
        return

    loguru_logger = None
    try:
        from loguru import logger as loguru_logger  # type: ignore
    except Exception:
        loguru_logger = None

    prev_disable = logging.root.manager.disable
    devnull = open(os.devnull, "w")
    try:
        logging.disable(logging.CRITICAL)
        if loguru_logger is not None:
            loguru_logger.disable("sam3d_objects")
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield
    finally:
        if loguru_logger is not None:
            loguru_logger.enable("sam3d_objects")
        logging.disable(prev_disable)
        devnull.close()


class TemporalController:
    """Holds keyframe latents and per-frame trajectories used by the hook.

    Mirrors ``demo_proj_temporal.TemporalController``; we copy it here to
    decouple from sam-3d-objects' demo code (which can drift) and to keep
    the SAM3D helpers self-contained inside this module.
    """

    def __init__(self, lambda_temp: float = 0.5):
        self.lambda_temp = lambda_temp
        self.frozen_state = None
        self.prev_traj = None
        self._cur_traj = []
        self._last_state = {}
        self._step_idx = 0
        self.mode = "off"  # "off" | "keyframe" | "follow"

    def begin(self, mode: str) -> None:
        assert mode in {"off", "keyframe", "follow"}
        self.mode = mode
        self._cur_traj = []
        self._step_idx = 0
        self._last_state = {}

    def end_keyframe(self) -> None:
        self.frozen_state = {k: v.detach().clone() for k, v in self._last_state.items() if k in _SAM3D_FREEZE_KEYS}
        self.prev_traj = self._cur_traj
        self.mode = "off"

    def end_follow(self, commit: bool = True) -> None:
        if commit:
            self.prev_traj = self._cur_traj
        self.mode = "off"

    @staticmethod
    def _broadcast_to(ref, target):
        ref = ref.to(device=target.device, dtype=target.dtype)
        if ref.shape[0] != target.shape[0]:
            ref = ref.expand(target.shape[0], *ref.shape[1:]).contiguous()
        return ref

    def override_state(self, x_t):
        if self.mode != "follow" or self.frozen_state is None or not isinstance(x_t, dict):
            return x_t
        x_t = dict(x_t)
        for k, ref in self.frozen_state.items():
            if k in x_t:
                x_t[k] = self._broadcast_to(ref, x_t[k])
        return x_t

    def modify_velocity(self, v, x_t):
        if self.mode == "off" or not isinstance(v, dict):
            return v
        v = dict(v)
        if self.mode == "follow":
            for k in _SAM3D_FREEZE_KEYS:
                if k in v:
                    v[k] = torch.zeros_like(v[k])
            if (self.prev_traj is not None and self._step_idx < len(self.prev_traj) and self.lambda_temp != 0.0):
                prev = self.prev_traj[self._step_idx]
                for k in _SAM3D_GUIDE_KEYS:
                    if k in v and k in prev and k in x_t:
                        ref = self._broadcast_to(prev[k], x_t[k])
                        v[k] = v[k] - self.lambda_temp * (x_t[k] - ref)
        return v

    def end_step(self, x_tp1):
        if self.mode == "off" or not isinstance(x_tp1, dict):
            return
        snap = {k: x_tp1[k][-1:].detach().clone() for k in _SAM3D_GUIDE_KEYS if k in x_tp1}
        self._cur_traj.append(snap)
        self._last_state = {k: x_tp1[k][-1:].detach().clone() for k in (set(_SAM3D_FREEZE_KEYS) | set(_SAM3D_GUIDE_KEYS)) if k in x_tp1}
        self._step_idx += 1


class _TemporalSolver(_Sam3dODESolver):
    """Wraps the original SAM3D solver, injecting controller hooks per step."""

    def __init__(self, base, ctrl):
        self.base = base
        self.ctrl = ctrl

    def step(self, dynamics_fn, x_t, t, dt, *args, **kwargs):
        x_t = self.ctrl.override_state(x_t)

        def wrapped(_x, _t, *_a, **_kw):
            v = dynamics_fn(_x, _t, *_a, **_kw)
            return self.ctrl.modify_velocity(v, _x)

        x_tp1 = self.base.step(wrapped, x_t, t, dt, *args, **kwargs)
        x_tp1 = self.ctrl.override_state(x_tp1)
        self.ctrl.end_step(x_tp1)
        return x_tp1


@contextlib.contextmanager
def _temporal_hook(inference_model, ctrl):
    ss_gen = inference_model._pipeline.models["ss_generator"]
    original = ss_gen._solver
    ss_gen._solver = _TemporalSolver(original, ctrl)
    try:
        yield
    finally:
        ss_gen._solver = original


class Sam3dResetController:
    """Runs SAM3D once at frame 0 (keyframe) then on demand at reset frames
    (follow) using a chained ``traj_history`` indexed by anchor frame.

    Anchor chaining rule:
      * Frame 0 keyframe -> traj stored under key 0.
      * Reset @ frame i with anchor = j -> follow with prev_traj=traj[j];
        the new trajectory is stored under key i and becomes the anchor for
        the next reset.

    The controller is ``cotracker_model``-style worker-state friendly: it
    creates the heavy ``Inference`` object lazily on first use.
    """

    def __init__(self, config_path: str = SAM3D_CONFIG_PATH, lambda_temp: float = 0.5, quiet: bool = True):
        if not _SAM3D_AVAILABLE:
            raise RuntimeError(
                "SAM3D not importable; cannot instantiate Sam3dResetController. "
                f"Last import error: {_SAM3D_IMPORT_ERROR}"
            )
        self.config_path = config_path
        self.lambda_temp = lambda_temp
        self.quiet = quiet
        self.inference = None  # lazy
        self.ctrl = TemporalController(lambda_temp=lambda_temp)
        self.traj_history = {}      # {frame_idx: prev_traj list}
        self.outputs = {}           # {frame_idx: dict (rotation/translation/scale/intrinsics/glb)}
        self.keyframe_idx = None    # which frame_idx supplied frozen_state

    def _ensure_inference(self):
        if self.inference is None:
            print(f"[SAM3D-RESET] Lazy-loading SAM3D Inference from {self.config_path}")
            # SAM3D's Dino embedder uses torch.hub.load(repo_or_dir="dinov2",
            # source="local"), which resolves "dinov2" against the current
            # working directory. The actual repo lives at SAM3D_REPO/dinov2,
            # so we chdir there for the duration of model construction (and
            # restore CWD afterwards so the rest of diffusion-vas isn't
            # affected). Same trick demo_proj_temporal.py implicitly relies
            # on by being executed from the SAM3D root.
            _prev_cwd = os.getcwd()
            try:
                os.chdir(SAM3D_REPO)
                with _suppress_sam3d_logs(self.quiet):
                    self.inference = Sam3dInference(self.config_path, compile=False)
            finally:
                os.chdir(_prev_cwd)

    @contextlib.contextmanager
    def _sam3d_cwd(self):
        _prev = os.getcwd()
        try:
            os.chdir(SAM3D_REPO)
            yield
        finally:
            os.chdir(_prev)

    def keyframe(self, image_np, mask_np, frame_idx: int = 0, seed: int = 42) -> dict:
        """Run SAM3D in keyframe mode; cache frozen_state and traj_history[frame_idx]."""
        self._ensure_inference()
        self.ctrl.begin("keyframe")
        with self._sam3d_cwd(), _temporal_hook(self.inference, self.ctrl), _suppress_sam3d_logs(self.quiet):
            out = self.inference(image_np, mask_np, seed=seed)
        self.ctrl.end_keyframe()
        # Snapshot the trajectory off-graph so we can re-use it as anchor later.
        self.traj_history[frame_idx] = [
            {k: v.detach().clone() for k, v in step.items()}
            for step in self.ctrl.prev_traj
        ]
        self.outputs[frame_idx] = {
            "rotation": out["rotation"].detach().clone(),
            "translation": out["translation"].detach().clone(),
            "scale": out["scale"].detach().clone(),
            "intrinsics": out["intrinsics"].detach().clone() if hasattr(out.get("intrinsics", None), "detach") else out.get("intrinsics"),
            "glb": out.get("glb", None),
        }
        self.keyframe_idx = frame_idx
        return self.outputs[frame_idx]

    def follow_begin(self, anchor_idx: int) -> None:
        """Begin a follow session anchored on ``traj_history[anchor_idx]`` (call ``follow_run`` then ``follow_end``)."""
        self._ensure_inference()
        if anchor_idx not in self.traj_history:
            raise KeyError(f"Anchor traj for frame {anchor_idx} not in history; keys={list(self.traj_history)}")
        self.ctrl.prev_traj = self.traj_history[anchor_idx]
        self.ctrl.begin("follow")

    def follow_run(self, image_np, mask_np, seed: int = 42) -> dict:
        """Run stage-1 follow inference after ``follow_begin``."""
        with self._sam3d_cwd(), _temporal_hook(self.inference, self.ctrl), _suppress_sam3d_logs(self.quiet):
            return self.inference._pipeline.run(
                self.inference.merge_mask_to_rgba(image_np, mask_np),
                None,
                seed=seed,
                stage1_only=True,
                pointmap=None,
            )

    def follow_end(self, frame_idx: int, commit: bool, decoded: dict) -> dict:
        """Finish follow: ``decoded`` is stored in ``outputs`` when ``commit`` is True (rotation path may pass scale-locked / flipped quats)."""
        self.ctrl.end_follow(commit=commit)
        out_dec = {
            "rotation": decoded["rotation"].detach().clone(),
            "translation": decoded["translation"].detach().clone(),
            "scale": decoded["scale"].detach().clone(),
            "intrinsics": decoded.get("intrinsics"),
        }
        intr = out_dec.get("intrinsics")
        if intr is not None and hasattr(intr, "detach"):
            out_dec["intrinsics"] = intr.detach().clone()
        if commit:
            self.traj_history[frame_idx] = [
                {k: v.detach().clone() for k, v in step.items()}
                for step in self.ctrl.prev_traj
            ]
            self.outputs[frame_idx] = out_dec
        return out_dec

    def follow_abort(self) -> None:
        """Discard an in-progress follow (e.g. after ``follow_begin`` + failed ``follow_run``)."""
        if getattr(self.ctrl, "mode", "off") == "follow":
            self.ctrl.end_follow(commit=False)

    def follow(
        self,
        image_np,
        mask_np,
        anchor_idx: int,
        frame_idx: int,
        seed: int = 42,
        commit: bool = True,
    ) -> dict:
        """Run SAM3D follow in one shot (same as ``follow_begin`` → ``follow_run`` → ``follow_end``).

        For local outlier handling (``demo_proj_temporal.py``), use the split API so
        ``commit`` can depend on the processed quaternion vs ``trusted_quat``.
        """
        self.follow_begin(anchor_idx)
        try:
            out = self.follow_run(image_np, mask_np, seed=seed)
            decoded = {
                "rotation": out["rotation"].detach().clone(),
                "translation": out["translation"].detach().clone(),
                "scale": out["scale"].detach().clone(),
                "intrinsics": out.get("intrinsics"),
            }
            return self.follow_end(frame_idx, commit=commit, decoded=decoded)
        except Exception:
            self.follow_abort()
            raise


def calculate_iou(mask1, mask2):
    """Calculates Intersection over Union for two binary masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0.0


def probe_sam3d_rotation(
    sam3d_controller,
    sam3d_raw_rgbs_np: np.ndarray,
    sam3d_raw_masks_np: np.ndarray,
    sampled_indices,                    # length-N list of raw frame idx for each sampled frame
    pnp_rot_col_major: torch.Tensor,    # (N, 3, 3) col-major
    pnp_trans: torch.Tensor,            # (N, 3)
    iou_pnp_per_frame: np.ndarray,      # (N,) IoU from compute_pnp_health
    verts: torch.Tensor,                # (V, 3)
    faces: torch.Tensor,                # (F, 3)
    scale_full: torch.Tensor,           # (3,)
    amodal_masks_np: np.ndarray,        # (N, H, W)
    focal_length: torch.Tensor,         # (N, 2)
    principal_point: torch.Tensor,      # (N, 2)
    H_out: int, W_out: int,
    device: torch.device,
    probe_frames,                       # iterable of sampled-frame indices to probe
    last_anchor_idx: int,
    sam3d_seed: int = 42,
    angle_threshold_deg: float = 25.0,
    iou_improvement_margin: float = 0.05,
):
    """Sparsely call SAM3D follow at probe frames; return rotation-disagreement
    candidates and a cache of SAM3D outputs for later reuse.

    A frame is a candidate if:
      angle(R_pnp, R_sam) > angle_threshold_deg

    IoU is still logged for diagnostics, but it is not used as a trigger gate:
    symmetric-object rotation failures can have nearly identical silhouette IoU,
    and requiring IoU improvement delays the reset until much later.

    Returns
    -------
    candidates : list of (frame_idx, reason_str)
    cache      : dict frame_idx -> {"R_sam","T_pnp","iou_sam","iou_pnp","angle_deg"}
    """
    from pytorch3d.transforms import quaternion_to_matrix as _q2m

    cameras_one = lambda fi: PerspectiveCameras(
        focal_length=focal_length[fi:fi + 1],
        principal_point=principal_point[fi:fi + 1],
        image_size=((H_out, W_out),),
        in_ndc=False,
        device=device,
    )
    raster_settings = RasterizationSettings(image_size=(H_out, W_out), blur_radius=1e-4, faces_per_pixel=20)
    cands = []
    cache = {}
    for fi in probe_frames:
        if fi < 0 or fi >= pnp_rot_col_major.shape[0]:
            continue
        if not np.isfinite(iou_pnp_per_frame[fi]):
            continue
        try:
            fi_raw = int(sampled_indices[fi])
            img = np.ascontiguousarray(sam3d_raw_rgbs_np[fi_raw])
            msk = np.ascontiguousarray(sam3d_raw_masks_np[fi_raw])
            if img.dtype != np.uint8 and img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            msk_bool = (msk > 0).astype(np.uint8)
            sam_out = sam3d_controller.follow(
                img, msk_bool, anchor_idx=last_anchor_idx, frame_idx=fi, seed=sam3d_seed,
            )
            sam_q = sam_out["rotation"]
            sam_q = sam_q.squeeze(1) if sam_q.dim() == 3 else sam_q
            R_sam = _q2m(sam_q).to(device).reshape(3, 3)
        except Exception as _e:
            print(f"[SAM3D-RESET]   probe @ frame {fi} failed: {_e}")
            continue

        R_pnp_row = pnp_rot_col_major[fi].T  # col-major -> row-major
        T_pnp = pnp_trans[fi]
        rel = R_pnp_row.T @ R_sam
        cos = ((rel.diagonal().sum() - 1.0) * 0.5).clamp(-1.0, 1.0)
        angle_deg = float(torch.acos(cos).item() * 180.0 / math.pi)

        cam = cameras_one(fi)
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cam, raster_settings=raster_settings),
            shader=SoftSilhouetteShader(),
        )
        _tf = Transform3d(device=device).scale(scale_full).rotate(R_sam.unsqueeze(0)).translate(T_pnp.unsqueeze(0))
        _pv = _tf.transform_points(verts.unsqueeze(0))[0]
        _mesh = Meshes(
            verts=[_pv],
            faces=[faces],
            textures=TexturesVertex(verts_features=torch.ones_like(_pv)[None]),
        )
        with torch.no_grad():
            _frag = renderer.rasterizer(_mesh, cameras=cam)
            _sil = renderer.shader(_frag, _mesh, cameras=cam)[0, ..., 3].cpu().numpy()
        sil_bin = (_sil > 0.5).astype(bool)
        gt = (amodal_masks_np[fi] > 0).astype(bool)
        inter = float(np.logical_and(sil_bin, gt).sum())
        union = float(np.logical_or(sil_bin, gt).sum())
        iou_sam = (inter / union) if union > 0 else 0.0
        iou_pnp_fi = float(iou_pnp_per_frame[fi])

        cache[fi] = {
            "R_sam": R_sam.detach().cpu(),
            "T_pnp": T_pnp.detach().cpu(),
            "iou_sam": iou_sam,
            "iou_pnp": iou_pnp_fi,
            "angle_deg": angle_deg,
        }

        if angle_deg > angle_threshold_deg:
            reason = (f"R disagreement {angle_deg:.1f} deg vs SAM3D; "
                      f"sam_iou={iou_sam:.2f}, pnp_iou={iou_pnp_fi:.2f} (IoU not gated)")
            cands.append((int(fi), reason))

    return cands, cache
