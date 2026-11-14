"""Helpers for analysing long transcripts in time-based chunks.

Long podcasts / streams overflow a single LLM prompt (and used to be silently
truncated). These helpers split the transcript into overlapping time windows,
apportion the requested clip count across windows, and merge the per-window
results back into one de-duplicated, ranked clip list.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

# A window is (window_start_sec, window_end_sec, first_line_idx, end_line_idx_exclusive)
Window = Tuple[float, float, int, int]


def split_transcript_windows(
    lines: List[Dict[str, Any]],
    chunk_sec: float = 1500.0,
    overlap_sec: float = 60.0,
    max_lines_per_chunk: int = 1500,
) -> List[Window]:
    """Split transcript lines (dicts with 'start'/'end') into overlapping time windows.

    Windows are cut by time, and additionally capped at `max_lines_per_chunk` lines
    so very dense transcripts still fit a prompt. A short tail window (< 30% of
    `chunk_sec`) is folded into the previous window.
    """
    if not lines:
        return []
    chunk_sec = max(60.0, float(chunk_sec))
    overlap_sec = max(0.0, min(float(overlap_sec), chunk_sec / 2))

    windows: List[Window] = []
    n = len(lines)
    i = 0
    while i < n:
        ws = float(lines[i]["start"])
        we = ws + chunk_sec
        j = i
        while j < n and float(lines[j]["start"]) < we and (j - i) < max_lines_per_chunk:
            j += 1
        j = max(j, i + 1)
        w_end = max(float(lines[j - 1]["end"]), float(lines[j - 1]["start"]))
        windows.append((ws, w_end, i, j))
        if j >= n:
            break
        # Step back by the overlap so moments spanning a boundary appear whole in one window.
        next_start_t = float(lines[j]["start"]) - overlap_sec
        k = j
        while k > i + 1 and float(lines[k - 1]["start"]) >= next_start_t:
            k -= 1
        i = k

    if len(windows) >= 2:
        last = windows[-1]
        if (last[1] - last[0]) < chunk_sec * 0.3 and (last[3] - windows[-2][2]) <= max_lines_per_chunk * 1.3:
            prev = windows[-2]
            windows[-2] = (prev[0], max(prev[1], last[1]), prev[2], last[3])
            windows.pop()
    return windows


def apportion_clip_range(min_total: int, max_total: int, window_sec: float, total_sec: float) -> Tuple[int, int]:
    """Share of a (min,max) clip-count range for one window, proportional to its length."""
    frac = (window_sec / total_sec) if total_sec > 0 else 1.0
    frac = max(0.0, min(1.0, frac))
    lo = max(1, int(round(min_total * frac)))
    hi = max(lo, int(math.ceil(max_total * frac)))
    return lo, hi


def _overlap_ratio(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    a0, a1 = float(a.get("start_time", 0.0)), float(a.get("end_time", 0.0))
    b0, b1 = float(b.get("start_time", 0.0)), float(b.get("end_time", 0.0))
    inter = min(a1, b1) - max(a0, b0)
    shorter = min(a1 - a0, b1 - b0)
    if inter <= 0 or shorter <= 0:
        return 0.0
    return inter / shorter


def merge_chunk_clips(
    clip_lists: List[List[Dict[str, Any]]],
    max_clips: Optional[int] = None,
    duplicate_overlap: float = 0.4,
) -> List[Dict[str, Any]]:
    """Merge clips from several windows: drop invalid ones, de-duplicate overlaps
    (keeping the higher virality score), rank by score, and cap the count."""
    pool: List[Dict[str, Any]] = []
    for clips in clip_lists:
        for c in clips or []:
            try:
                s, e = float(c.get("start_time", 0.0)), float(c.get("end_time", 0.0))
            except (TypeError, ValueError):
                continue
            if e > s:
                pool.append(c)

    pool.sort(key=lambda c: float(c.get("virality_score") or 0), reverse=True)
    kept: List[Dict[str, Any]] = []
    for c in pool:
        if any(_overlap_ratio(c, k) > duplicate_overlap for k in kept):
            continue
        kept.append(c)
        if max_clips and len(kept) >= max_clips:
            break
    return kept


def clips_within_window(clips: List[Dict[str, Any]], ws: float, we: float, slack: float = 5.0) -> List[Dict[str, Any]]:
    """Drop clips a model placed outside the window it was given (hallucinated timestamps)."""
    out = []
    for c in clips or []:
        try:
            s = float(c.get("start_time", 0.0))
        except (TypeError, ValueError):
            continue
        if (ws - slack) <= s <= (we + slack):
            out.append(c)
    return out


def is_auth_error(err: BaseException) -> bool:
    """Bad / missing API key: no segment or model will succeed, so stop everything."""
    s = str(err).lower()
    return any(x in s for x in ("http 401", "http 403", "invalid api key", "unauthorized", "incorrect api key"))


def is_non_retryable_error(err: BaseException) -> bool:
    """Auth / bad-request errors won't be fixed by retrying the same model."""
    return is_auth_error(err) or "http 400" in str(err).lower()


def is_rate_limit_error(err: BaseException) -> bool:
    s = str(err).lower()
    return any(x in s for x in ("429", "quota", "rate limit", "credit", "balance"))
