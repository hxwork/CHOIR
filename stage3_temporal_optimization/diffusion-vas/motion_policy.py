"""Motion policy helpers."""

from __future__ import annotations

from typing import Any, Mapping


FORCE_ROTATION_MODE = "rotation_likely"
POLICY_OVERRIDE_FORCE_ROTATION = "force_rotation_likely"


def force_rotation_likely_profile(
    profile: Mapping[str, Any] | None,
    *,
    reason: str,
) -> dict[str, Any]:
    """Return a profile copy whose policy is forced to rotation-likely."""
    out = dict(profile or {})
    original_mode = out.get("suggested_mode")
    out["suggested_mode"] = FORCE_ROTATION_MODE
    out["policy_original_suggested_mode"] = original_mode
    out["policy_override"] = POLICY_OVERRIDE_FORCE_ROTATION
    out["policy_override_reason"] = reason
    return out
