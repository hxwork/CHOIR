"""Merge two one-hand/object HOI mesh sequences into a two-hand sequence.

CHOIR Stage 3 fits one hand at a time. For two-hand / one-object videos, run
``run_temporal_optimization.py`` twice with ``--hand_side left`` and
``--hand_side right``, then merge the resulting
``stage3/final/optimized_meshes`` trees with this script.

Each input sequence must contain ``hand/`` and ``object/`` PLY folders (current
CHOIR layout). The right-hand source is aligned to the left-hand base by a
per-frame similarity transform estimated from the two object meshes.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np


PLY_DTYPE_MAP = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}

STRUCT_FORMAT_MAP = {
    "char": "b",
    "int8": "b",
    "uchar": "B",
    "uint8": "B",
    "short": "h",
    "int16": "h",
    "ushort": "H",
    "uint16": "H",
    "int": "i",
    "int32": "i",
    "uint": "I",
    "uint32": "I",
    "float": "f",
    "float32": "f",
    "double": "d",
    "float64": "d",
}


@dataclass(frozen=True)
class PlyHeader:
    fmt: str
    vertex_count: int
    face_count: int
    vertex_properties: list[tuple[str, str]]
    face_list_types: tuple[str, str]


@dataclass(frozen=True)
class MergeSummary:
    left_seq: str
    right_seq: str
    output_seq: str
    frame_count: int
    alignment_rms: dict[str, float]
    scale_mode: str
    similarity_scale: dict[str, float]


def frame_number(path: Path) -> int:
    nums = re.findall(r"\d+", path.stem)
    return int(nums[0]) if nums else -1


def parse_ply_header(stream: BinaryIO) -> PlyHeader:
    first_line = stream.readline().decode("ascii", errors="replace").strip()
    if first_line != "ply":
        raise ValueError("Only PLY meshes are supported.")

    fmt = ""
    vertex_count = 0
    face_count = 0
    vertex_properties: list[tuple[str, str]] = []
    face_list_types: tuple[str, str] | None = None
    current_element = None

    while True:
        line = stream.readline()
        if not line:
            raise ValueError("Unexpected end of file while reading PLY header.")
        text = line.decode("ascii", errors="replace").strip()
        if text == "end_header":
            break
        if not text or text.startswith("comment"):
            continue

        parts = text.split()
        if parts[0] == "format":
            fmt = parts[1]
        elif parts[:2] == ["element", "vertex"]:
            vertex_count = int(parts[2])
            current_element = "vertex"
        elif parts[:2] == ["element", "face"]:
            face_count = int(parts[2])
            current_element = "face"
        elif parts[0] == "element":
            current_element = parts[1]
        elif parts[0] == "property" and current_element == "vertex" and len(parts) == 3:
            vertex_properties.append((parts[2], parts[1]))
        elif parts[:2] == ["property", "list"] and current_element == "face" and len(parts) >= 5:
            face_list_types = (parts[2], parts[3])

    if fmt not in {"ascii", "binary_little_endian"}:
        raise ValueError(f"Unsupported PLY format: {fmt!r}")
    if vertex_count <= 0:
        raise ValueError("PLY file has no vertices.")
    if face_list_types is None:
        face_list_types = ("uchar", "int")
    return PlyHeader(fmt, vertex_count, face_count, vertex_properties, face_list_types)


def _binary_dtype(type_name: str) -> str:
    if type_name not in PLY_DTYPE_MAP:
        raise ValueError(f"Unsupported PLY scalar type: {type_name}")
    return "<" + PLY_DTYPE_MAP[type_name]


def _struct_format(type_name: str) -> str:
    if type_name not in STRUCT_FORMAT_MAP:
        raise ValueError(f"Unsupported PLY scalar type: {type_name}")
    return "<" + STRUCT_FORMAT_MAP[type_name]


def _read_binary_faces(stream: BinaryIO, count: int, list_types: tuple[str, str]) -> np.ndarray:
    count_fmt = _struct_format(list_types[0])
    index_fmt = _struct_format(list_types[1])
    count_size = struct.calcsize(count_fmt)
    index_size = struct.calcsize(index_fmt)
    faces = []

    for _ in range(count):
        face_size_bytes = stream.read(count_size)
        if len(face_size_bytes) != count_size:
            raise ValueError("Unexpected end of file while reading face size.")
        face_size = struct.unpack(count_fmt, face_size_bytes)[0]
        index_bytes = stream.read(index_size * face_size)
        if len(index_bytes) != index_size * face_size:
            raise ValueError("Unexpected end of file while reading face indices.")
        indices = struct.unpack("<" + STRUCT_FORMAT_MAP[list_types[1]] * face_size, index_bytes)
        if face_size == 3:
            faces.append(indices)
        elif face_size > 3:
            first = indices[0]
            for i in range(1, face_size - 1):
                faces.append((first, indices[i], indices[i + 1]))

    return np.asarray(faces, dtype=np.int32)


def load_mesh(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load vertex XYZ and triangular faces from an ascii or binary little-endian PLY."""
    path = Path(path)
    with path.open("rb") as stream:
        header = parse_ply_header(stream)
        prop_names = [name for name, _ in header.vertex_properties]
        missing = {"x", "y", "z"} - set(prop_names)
        if missing:
            raise ValueError(f"{path} is missing vertex properties: {sorted(missing)}")

        if header.fmt == "ascii":
            vertices = []
            for _ in range(header.vertex_count):
                values = stream.readline().decode("ascii").split()
                vertices.append([float(values[prop_names.index(axis)]) for axis in ("x", "y", "z")])
            faces = []
            for _ in range(header.face_count):
                values = [int(v) for v in stream.readline().decode("ascii").split()]
                face_size, indices = values[0], values[1:]
                if face_size == 3:
                    faces.append(indices[:3])
                elif face_size > 3:
                    faces.extend([[indices[0], indices[i], indices[i + 1]] for i in range(1, face_size - 1)])
            return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32)

        dtype = np.dtype([(name, _binary_dtype(type_name)) for name, type_name in header.vertex_properties])
        vertex_data = np.fromfile(stream, dtype=dtype, count=header.vertex_count)
        vertices = np.column_stack([vertex_data[axis] for axis in ("x", "y", "z")]).astype(np.float32)
        faces = _read_binary_faces(stream, header.face_count, header.face_list_types)
        return vertices, faces


def write_ply(path: str | Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    """Write a triangular mesh as ascii PLY."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)

    with path.open("w", encoding="ascii") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {vertices.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element face {faces.shape[0]}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for vertex in vertices:
            f.write(f"{vertex[0]:.9g} {vertex[1]:.9g} {vertex[2]:.9g}\n")
        for face in faces:
            f.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")


def estimate_rigid_transform(source_points: np.ndarray, target_points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Return R, t, rms for target ~= source @ R.T + t."""
    source = np.asarray(source_points, dtype=np.float64)
    target = np.asarray(target_points, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(f"Expected matching point arrays shaped (N, 3), got {source.shape} and {target.shape}")
    if source.shape[0] < 3:
        raise ValueError("At least three points are required for rigid alignment.")

    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source_centered = source - source_center
    target_centered = target - target_center
    covariance = source_centered.T @ target_centered
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    translation = target_center - source_center @ rotation.T
    aligned = transform_vertices(source, rotation, translation)
    rms_error = float(np.sqrt(np.mean(np.sum((aligned - target) ** 2, axis=1))))
    return rotation, translation, rms_error


def estimate_similarity_transform(source_points: np.ndarray, target_points: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, float]:
    """Return scale, R, t, rms for target ~= scale * (source @ R.T) + t."""
    source = np.asarray(source_points, dtype=np.float64)
    target = np.asarray(target_points, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(f"Expected matching point arrays shaped (N, 3), got {source.shape} and {target.shape}")
    if source.shape[0] < 3:
        raise ValueError("At least three points are required for similarity alignment.")

    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source_centered = source - source_center
    target_centered = target - target_center
    covariance = source_centered.T @ target_centered
    u, singular_values, vt = np.linalg.svd(covariance)
    reflection = np.ones(3, dtype=np.float64)
    if np.linalg.det(vt.T @ u.T) < 0:
        reflection[-1] = -1.0
    rotation = vt.T @ np.diag(reflection) @ u.T
    source_variance = float(np.sum(source_centered * source_centered))
    if source_variance <= 0.0:
        raise ValueError("Source points are degenerate and cannot define a scale.")
    scale = float(np.sum(singular_values * reflection) / source_variance)
    translation = target_center - scale * (source_center @ rotation.T)
    aligned = scale * (source @ rotation.T) + translation
    rms_error = float(np.sqrt(np.mean(np.sum((aligned - target) ** 2, axis=1))))
    return scale, rotation, translation, rms_error


def transform_vertices(vertices: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float64)
    rotation = np.asarray(rotation, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64)
    return (vertices @ rotation.T + translation).astype(np.float64)


def transform_vertices_similarity(
    vertices: np.ndarray,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float64)
    rotation = np.asarray(rotation, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64)
    return (float(scale) * (vertices @ rotation.T) + translation).astype(np.float64)


def scale_around_center(vertices: np.ndarray, center: np.ndarray, scale: float) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    return (center + float(scale) * (vertices - center)).astype(np.float64)


def combine_hand_meshes(
    left_vertices: np.ndarray,
    left_faces: np.ndarray,
    right_vertices: np.ndarray,
    right_faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    left_vertices = np.asarray(left_vertices, dtype=np.float32)
    right_vertices = np.asarray(right_vertices, dtype=np.float32)
    left_faces = np.asarray(left_faces, dtype=np.int32)
    right_faces = np.asarray(right_faces, dtype=np.int32)
    vertices = np.vstack([left_vertices, right_vertices])
    faces = np.vstack([left_faces, right_faces + left_vertices.shape[0]])
    return vertices, faces


def list_mesh_files(seq_dir: Path, subdir: str) -> list[Path]:
    mesh_dir = seq_dir / subdir
    if not mesh_dir.is_dir():
        raise FileNotFoundError(f"Expected {mesh_dir} to exist.")
    files = sorted([*mesh_dir.glob("*.ply")], key=frame_number)
    if not files:
        raise FileNotFoundError(f"No PLY mesh files found under {mesh_dir}.")
    return files


def _reset_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    for name in ("object", "left_hand", "right_hand", "hand"):
        (output_dir / name).mkdir(parents=True, exist_ok=True)


def merge_sequences(
    left_dir: str | Path,
    right_dir: str | Path,
    output_dir: str | Path,
    max_frames: int | None = None,
    scale_mode: str = "middle",
) -> MergeSummary:
    left_dir = Path(left_dir)
    right_dir = Path(right_dir)
    output_dir = Path(output_dir)
    left_hand_files = list_mesh_files(left_dir, "hand")
    left_object_files = list_mesh_files(left_dir, "object")
    right_hand_files = list_mesh_files(right_dir, "hand")
    right_object_files = list_mesh_files(right_dir, "object")
    frame_count = min(len(left_hand_files), len(left_object_files), len(right_hand_files), len(right_object_files))
    if max_frames is not None:
        frame_count = min(frame_count, int(max_frames))
    if frame_count <= 0:
        raise ValueError("No overlapping frames to merge.")
    if scale_mode not in {"middle", "left"}:
        raise ValueError(f"Unsupported scale_mode {scale_mode!r}; expected 'middle' or 'left'.")

    _reset_output_dir(output_dir)
    rms_errors = []
    similarity_scales = []
    for frame_idx in range(frame_count):
        left_object, left_object_faces = load_mesh(left_object_files[frame_idx])
        right_object, right_object_faces = load_mesh(right_object_files[frame_idx])
        left_hand, left_hand_faces = load_mesh(left_hand_files[frame_idx])
        right_hand, right_hand_faces = load_mesh(right_hand_files[frame_idx])

        if left_object.shape != right_object.shape:
            raise ValueError(
                f"Object vertex shape mismatch at frame {frame_idx}: {left_object.shape} vs {right_object.shape}"
            )
        if left_object_faces.shape != right_object_faces.shape:
            raise ValueError(
                f"Object face shape mismatch at frame {frame_idx}: {left_object_faces.shape} vs {right_object_faces.shape}"
            )

        sim_scale, rotation, translation, rms_error = estimate_similarity_transform(right_object, left_object)
        left_center = left_object.astype(np.float64).mean(axis=0)
        if scale_mode == "middle":
            left_to_middle_scale = 1.0 / np.sqrt(sim_scale)
            transformed_object = scale_around_center(left_object, left_center, left_to_middle_scale).astype(np.float32)
            transformed_left_hand = scale_around_center(left_hand, left_center, left_to_middle_scale).astype(np.float32)
            right_hand_in_left_scale = transform_vertices_similarity(right_hand, sim_scale, rotation, translation)
            transformed_right_hand = scale_around_center(
                right_hand_in_left_scale,
                left_center,
                left_to_middle_scale,
            ).astype(np.float32)
        else:
            transformed_object = left_object
            transformed_left_hand = left_hand
            transformed_right_hand = transform_vertices_similarity(right_hand, sim_scale, rotation, translation).astype(np.float32)

        combined_hand_vertices, combined_hand_faces = combine_hand_meshes(
            transformed_left_hand,
            left_hand_faces,
            transformed_right_hand,
            right_hand_faces,
        )
        rms_errors.append(rms_error)
        similarity_scales.append(sim_scale)

        write_ply(output_dir / "object" / f"{frame_idx:04d}_mesh.ply", transformed_object, left_object_faces)
        write_ply(output_dir / "left_hand" / f"{frame_idx:04d}_hand.ply", transformed_left_hand, left_hand_faces)
        write_ply(output_dir / "right_hand" / f"{frame_idx:04d}_hand.ply", transformed_right_hand, right_hand_faces)
        write_ply(output_dir / "hand" / f"{frame_idx:04d}_hand.ply", combined_hand_vertices, combined_hand_faces)

    rms = np.asarray(rms_errors, dtype=np.float64)
    scales = np.asarray(similarity_scales, dtype=np.float64)
    summary = MergeSummary(
        left_seq=str(left_dir),
        right_seq=str(right_dir),
        output_seq=str(output_dir),
        frame_count=frame_count,
        alignment_rms={
            "mean": float(rms.mean()),
            "max": float(rms.max()),
            "min": float(rms.min()),
        },
        scale_mode=scale_mode,
        similarity_scale={
            "mean": float(scales.mean()),
            "max": float(scales.max()),
            "min": float(scales.min()),
        },
    )
    metadata = asdict(summary)
    metadata["assumptions"] = [
        "left_seq is the left-hand/object base coordinate system",
        "right_seq is the right-hand/object source sequence",
        "matching frame numbers are temporally aligned",
        "object vertex order is consistent across both sequences",
        "middle scale uses the geometric mean between left and right object scales",
    ]
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--left_seq",
        type=Path,
        required=True,
        help="Left-hand/object base sequence directory (…/stage3/final/optimized_meshes).",
    )
    parser.add_argument(
        "--right_seq",
        type=Path,
        required=True,
        help="Right-hand/object source sequence directory (…/stage3/final/optimized_meshes).",
    )
    parser.add_argument(
        "--output_seq",
        type=Path,
        required=True,
        help="Output two-hand sequence directory (…/stage3/final/optimized_meshes).",
    )
    parser.add_argument("--max_frames", type=int, default=None, help="Optional frame limit for smoke tests.")
    parser.add_argument(
        "--scale_mode",
        choices=("middle", "left"),
        default="middle",
        help="'middle' writes the object at the geometric mean scale; 'left' writes the original left object scale.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = merge_sequences(args.left_seq, args.right_seq, args.output_seq, max_frames=args.max_frames, scale_mode=args.scale_mode)
    print(
        f"Merged {summary.frame_count} frames into {summary.output_seq}; "
        f"object alignment RMS mean={summary.alignment_rms['mean']:.6g}, "
        f"max={summary.alignment_rms['max']:.6g}; "
        f"scale mean={summary.similarity_scale['mean']:.6g}, mode={summary.scale_mode}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
