"""Build a static HTML dashboard for per-video fitting results."""

from __future__ import annotations

import argparse
import datetime as _dt
import html
import json
from pathlib import Path
from urllib.parse import quote


VIDEO_FILES = [
    ("final", "optimized_fitting_SPARSE_final.mp4"),
    ("sparse", "optimized_fitting_SPARSE.mp4"),
    ("stage3", "stage3_pre_optim.mp4"),
    ("full", "optimized_fitting.mp4"),
    ("modal-amodal", "comparison_modal_amodal.mp4"),
]

DEBUG_FILES = [
    "interaction_motion_profile_stage3.json",
    "interaction_motion_profile_post_pnp.json",
    "interaction_motion_profile_sam3d_gate.json",
    "stage3_hand_anatomy_gate.csv",
    "stage3_hand_anatomy_gate.json",
    "stage5_mode_debug.json",
    "stage5_ray_delta_debug.json",
    "optimize_stage5.log",
    "optimize_stage5_reclassify.log",
    "optimize_stage5_prefix_diag.csv",
    "optimize_stage6.log",
    "stage_frame_ranges.json",
]


def _rel_url(path: Path, root: Path) -> str:
    rel = path.relative_to(root).as_posix()
    return quote(rel, safe="/._-")


def _load_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _collect_video_dir(video_dir: Path, root: Path) -> dict:
    profile = _load_json(video_dir / "interaction_motion_profile_stage3.json")
    post_profile = _load_json(video_dir / "interaction_motion_profile_post_pnp.json")
    gate_profile = _load_json(video_dir / "interaction_motion_profile_sam3d_gate.json")

    videos = []
    for label, filename in VIDEO_FILES:
        path = video_dir / filename
        if path.exists():
            videos.append({
                "label": label,
                "filename": filename,
                "url": _rel_url(path, root),
            })

    debug_links = []
    for filename in DEBUG_FILES:
        path = video_dir / filename
        if path.exists():
            debug_links.append({
                "label": filename,
                "url": _rel_url(path, root),
            })

    return {
        "video_id": video_dir.name,
        "has_final": any(v["filename"] == "optimized_fitting_SPARSE_final.mp4" for v in videos),
        "mode": str(profile.get("suggested_mode", "unknown")),
        "post_pnp_mode": str(post_profile.get("suggested_mode", "")),
        "sam3d_gate_mode": str(gate_profile.get("suggested_mode", "")),
        "n_interaction_frames": profile.get("n_interaction_frames"),
        "videos": videos,
        "debug_links": debug_links,
    }


def collect_results(input_data: Path, root: Path) -> list[dict]:
    if not input_data.exists():
        raise FileNotFoundError(f"input_data directory not found: {input_data}")
    rows = []
    for video_dir in sorted(input_data.iterdir(), key=lambda p: p.name):
        if not video_dir.is_dir() or not video_dir.name.isdigit():
            continue
        row = _collect_video_dir(video_dir, root)
        if row["videos"] or row["debug_links"]:
            rows.append(row)
    return rows


def _render_links(links: list[dict]) -> str:
    if not links:
        return '<span class="muted">no debug files</span>'
    return "\n".join(
        f'<a href="{html.escape(link["url"])}" target="_blank">{html.escape(link["label"])}</a>'
        for link in links
    )


def _render_videos(videos: list[dict]) -> str:
    if not videos:
        return '<div class="missing">No result videos found</div>'
    blocks = []
    for video in videos:
        label = html.escape(video["label"])
        filename = html.escape(video["filename"])
        url = html.escape(video["url"])
        blocks.append(f"""
          <div class="videoSlot" data-video-kind="{label}">
            <div class="videoTitle">{label}<span>{filename}</span></div>
            <video src="{url}" controls muted loop preload="metadata"></video>
          </div>
        """)
    return "\n".join(blocks)


def render_dashboard(rows: list[dict], root: Path, input_data: Path) -> str:
    generated = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cards = []
    for row in rows:
        vid = html.escape(row["video_id"])
        mode = html.escape(row["mode"])
        post = html.escape(row["post_pnp_mode"]) if row["post_pnp_mode"] else "-"
        gate = html.escape(row["sam3d_gate_mode"]) if row["sam3d_gate_mode"] else "-"
        n_inter = row["n_interaction_frames"]
        n_inter_text = "-" if n_inter is None else html.escape(str(n_inter))
        has_final = "1" if row["has_final"] else "0"
        cards.append(f"""
        <article class="card" data-video-id="{vid}" data-mode="{mode}" data-has-final="{has_final}">
          <header>
            <h2>{vid}</h2>
            <div class="badges">
              <span class="badge mode">{mode}</span>
              <span class="badge">postPnP: {post}</span>
              <span class="badge">sam3dGate: {gate}</span>
              <span class="badge">interFrames: {n_inter_text}</span>
            </div>
          </header>
          <section class="videos">
            {_render_videos(row["videos"])}
          </section>
          <section class="debugLinks">
            <strong>Debug</strong>
            {_render_links(row["debug_links"])}
          </section>
        </article>
        """)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Diffusion VAS Results Dashboard</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #111318;
      --panel: #1a1d24;
      --panel2: #222733;
      --text: #f2f4f8;
      --muted: #a7adbb;
      --line: #343b49;
      --accent: #7ab7ff;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
    }}
    .topbar {{
      position: sticky;
      top: 0;
      z-index: 10;
      padding: 16px 20px;
      background: rgba(17, 19, 24, 0.96);
      border-bottom: 1px solid var(--line);
    }}
    h1 {{ margin: 0 0 10px; font-size: 22px; }}
    .meta {{ color: var(--muted); font-size: 13px; margin-bottom: 12px; }}
    .controls {{ display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }}
    input, select, label.toggle {{
      background: var(--panel2);
      border: 1px solid var(--line);
      color: var(--text);
      border-radius: 8px;
      padding: 8px 10px;
      font-size: 14px;
    }}
    input {{ min-width: 220px; }}
    label.toggle {{ display: inline-flex; gap: 8px; align-items: center; }}
    main {{
      padding: 20px;
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(520px, 1fr));
      gap: 18px;
    }}
    .card {{
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 14px;
      overflow: hidden;
    }}
    .card header {{
      padding: 14px 14px 10px;
      border-bottom: 1px solid var(--line);
    }}
    .card h2 {{ margin: 0 0 8px; font-size: 20px; }}
    .badges {{ display: flex; gap: 8px; flex-wrap: wrap; }}
    .badge {{
      background: var(--panel2);
      border: 1px solid var(--line);
      border-radius: 999px;
      color: var(--muted);
      padding: 3px 8px;
      font-size: 12px;
    }}
    .badge.mode {{ color: var(--accent); }}
    .videos {{ padding: 12px; display: grid; grid-template-columns: 1fr; gap: 12px; }}
    .videoTitle {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      color: var(--text);
      font-size: 13px;
      margin-bottom: 6px;
    }}
    .videoTitle span {{ color: var(--muted); }}
    video {{
      display: block;
      width: 100%;
      max-height: 360px;
      background: #000;
      border-radius: 10px;
      border: 1px solid var(--line);
    }}
    .debugLinks {{
      border-top: 1px solid var(--line);
      padding: 12px;
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }}
    .debugLinks strong {{ margin-right: 4px; }}
    .debugLinks a {{
      color: var(--accent);
      text-decoration: none;
      background: var(--panel2);
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 4px 7px;
      font-size: 12px;
    }}
    .muted, .missing {{ color: var(--muted); }}
    .hidden {{ display: none; }}
  </style>
</head>
<body>
  <div class="topbar">
    <h1>Diffusion VAS Results Dashboard</h1>
    <div class="meta">
      Generated {html.escape(generated)} · root: {html.escape(root.as_posix())} · input: {html.escape(input_data.as_posix())} · videos: {len(rows)}
    </div>
    <div class="controls">
      <input id="search" placeholder="Filter by video_id, e.g. 20871">
      <select id="mode">
        <option value="">All modes</option>
        <option value="translation_likely">translation_likely</option>
        <option value="rotation_likely">rotation_likely</option>
        <option value="unknown">unknown</option>
      </select>
      <label class="toggle"><input id="finalOnly" type="checkbox"> show final video only</label>
      <span id="count" class="muted"></span>
    </div>
  </div>
  <main id="grid">
    {"".join(cards)}
  </main>
  <script>
    const search = document.getElementById('search');
    const mode = document.getElementById('mode');
    const finalOnly = document.getElementById('finalOnly');
    const count = document.getElementById('count');
    const cards = Array.from(document.querySelectorAll('.card'));

    function applyFilters() {{
      const q = search.value.trim().toLowerCase();
      const m = mode.value;
      const needFinal = finalOnly.checked;
      let shown = 0;
      for (const card of cards) {{
        const okSearch = !q || card.dataset.videoId.toLowerCase().includes(q);
        const okMode = !m || card.dataset.mode === m;
        const okFinal = !needFinal || card.dataset.hasFinal === '1';
        const visible = okSearch && okMode && okFinal;
        card.classList.toggle('hidden', !visible);
        for (const slot of card.querySelectorAll('.videoSlot')) {{
          slot.classList.toggle('hidden', needFinal && slot.dataset.videoKind !== 'final');
        }}
        if (visible) shown += 1;
      }}
      count.textContent = `${{shown}} / ${{cards.length}} shown`;
    }}

    search.addEventListener('input', applyFilters);
    mode.addEventListener('change', applyFilters);
    finalOnly.addEventListener('change', applyFilters);
    applyFilters();
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-data", default="input_data", type=Path)
    parser.add_argument("--output", default="results_dashboard.html", type=Path)
    args = parser.parse_args()

    root = Path.cwd().resolve()
    input_data = args.input_data
    if not input_data.is_absolute():
        input_data = root / input_data
    output = args.output
    if not output.is_absolute():
        output = root / output

    # Keep symlink paths unresolved so generated URLs stay relative to this
    # workspace, e.g. input_data/20871/result.mp4.
    rows = collect_results(input_data, root)
    output.write_text(render_dashboard(rows, root, input_data), encoding="utf-8")
    print(f"Wrote {output} with {len(rows)} video directories")


if __name__ == "__main__":
    main()
