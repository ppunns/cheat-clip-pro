"""Shot-cut detection used by Film mode to snap clip boundaries to camera cuts.

Instead of scanning a whole 2-hour film, only a few seconds around each clip
boundary are decoded (input-seeked, downscaled), which keeps it fast.
"""
from __future__ import annotations

import logging
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Sequence, Tuple

logger = logging.getLogger("cheat-clip-pro")

_PTS_RE = re.compile(r"pts_time:([0-9.]+)")


def detect_shot_cuts_near(
    video_path: str,
    center: float,
    before: float = 2.5,
    after: float = 2.5,
    threshold: float = 0.30,
    ffmpeg_bin: str = "ffmpeg",
) -> List[float]:
    """Return absolute timestamps of shot cuts within [center-before, center+after]."""
    ws = max(0.0, center - before)
    span = (center + after) - ws
    if span <= 0:
        return []
    cmd = [
        ffmpeg_bin, "-hide_banner", "-nostats",
        "-ss", f"{ws:.3f}", "-t", f"{span:.3f}", "-i", str(video_path),
        "-an", "-sn",
        "-vf", f"scale=160:-2,select='gt(scene,{threshold})',showinfo",
        "-f", "null", "-",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (subprocess.TimeoutExpired, OSError) as e:
        logger.warning(f"Shot detection failed near {center:.1f}s: {e}")
        return []
    cuts = []
    for line in proc.stderr.splitlines():
        if "Parsed_showinfo" not in line:
            continue
        m = _PTS_RE.search(line)
        if m:
            t = ws + float(m.group(1))
            # First decoded frame often registers as a "cut" against nothing; ignore it.
            if t - ws > 0.05:
                cuts.append(round(t, 3))
    return cuts


def pick_start_cut(cuts: Sequence[float], start: float, max_back: float = 2.5, max_fwd: float = 0.3) -> Optional[float]:
    """Latest cut shortly before the start (prefer not to clip the first spoken line)."""
    cands = [c for c in cuts if start - max_back <= c <= start + max_fwd]
    return max(cands) if cands else None


def pick_end_cut(cuts: Sequence[float], end: float, max_back: float = 0.3, max_fwd: float = 2.5) -> Optional[float]:
    """Earliest cut shortly after the end (prefer not to clip the last spoken line)."""
    cands = [c for c in cuts if end - max_back <= c <= end + max_fwd]
    return min(cands) if cands else None


def snap_clips_to_shots(
    video_path: str,
    ranges: List[Tuple[float, float]],
    min_bound: float = 0.0,
    max_bound: Optional[float] = None,
    min_duration: float = 5.0,
    ffmpeg_bin: str = "ffmpeg",
    workers: int = 4,
) -> List[Tuple[float, float]]:
    """Snap each (start, end) to nearby shot cuts. Ranges without a nearby cut are left unchanged."""
    if not ranges:
        return []

    def _snap(r: Tuple[float, float]) -> Tuple[float, float]:
        s, e = r
        ns = pick_start_cut(detect_shot_cuts_near(video_path, s, ffmpeg_bin=ffmpeg_bin), s)
        ne = pick_end_cut(detect_shot_cuts_near(video_path, e, ffmpeg_bin=ffmpeg_bin), e)
        s2 = ns if ns is not None else s
        e2 = ne if ne is not None else e
        s2 = max(min_bound, s2)
        if max_bound is not None:
            e2 = min(max_bound, e2)
        if e2 - s2 < min_duration:
            return r
        return (round(s2, 2), round(e2, 2))

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return list(pool.map(_snap, ranges))
