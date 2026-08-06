"""Shared CHOIR-relative paths for GraspFlowMatching."""

from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = MODULE_ROOT.parents[1]  # CHOIR/

DEFAULT_MESHDATA = REPO_ROOT / "stage2_grasp_correction" / "DexGraspNet_table" / "meshdata"
DEFAULT_MANO_ASSETS = REPO_ROOT / "stage2_grasp_correction" / "DexGraspNet_table" / "grasp_generation" / "mano"
DEFAULT_SOURCE_DIR = REPO_ROOT / "output"
DEFAULT_CONTACT_INDICES = DEFAULT_MANO_ASSETS / "contact_indices.json"
