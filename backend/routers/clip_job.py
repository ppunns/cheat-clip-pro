"""Agent-native clip endpoint.

POST /api/clip  → {job_id}
GET  /api/clip/{job_id} → job status + results

One call to start, one to poll.  When done the response contains title,
caption, hashtag and a download URL per clip — everything an AI agent
needs to publish without further API calls.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, BackgroundTasks
from fastapi.exceptions import HTTPException
from pydantic import BaseModel

from backend.routers.analyze import analyze_video
from backend.schemas.analyze import AnalyzeRequest
from backend.schemas.render import RenderBatchRequest, RenderSettingsModel
from backend.services.render_service import (
    BATCH_REQUESTS,
    RENDER_BATCHES,
    process_batch_rendering,
)

router = APIRouter(tags=["Clip Job"])
logger = logging.getLogger("cheat-clip-pro")

_JOBS: Dict[str, Dict[str, Any]] = {}
_JOB_TTL = 7200  # keep jobs 2 hours


# ── Schemas ────────────────────────────────────────────────────────────────

class ClipJobRequest(BaseModel):
    url: str
    duration: str = "30s"
    target_clip_count: Optional[Union[int, str]] = None
    content_type: str = "podcast"
    output_language: Optional[str] = None
    # render
    render: bool = True
    aspect_ratio: str = "9:16"
    design: Optional[str] = None
    enable_subtitles: bool = True
    # AI config passthrough (optional)
    api_key: Optional[str] = None
    api_provider: Optional[str] = None
    model: Optional[str] = None
    custom_focus: Optional[str] = None
    # clip range
    range_start: Optional[float] = None
    range_end: Optional[float] = None
    # misc passthrough
    cookies: Optional[str] = None
    proxy: Optional[str] = None
    subtitles: Optional[str] = None
    subtitles_filename: Optional[str] = None
    transcript_language: Optional[str] = "id"


# ── Endpoints ──────────────────────────────────────────────────────────────

@router.post("/api/clip")
async def create_clip_job(req: ClipJobRequest, bg: BackgroundTasks):
    """Start an async clip-and-render job.  Poll GET /api/clip/{job_id}."""
    job_id = uuid.uuid4().hex[:10]
    _JOBS[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "progress": "Queued",
        "source_title": None,
        "clips": [],
        "error": None,
        "created_at": time.time(),
    }
    bg.add_task(_run_job, job_id, req)
    return {"job_id": job_id}


@router.get("/api/clip/{job_id}")
def get_clip_job(job_id: str):
    """Return current status.  Poll until status is 'done' or 'error'."""
    _purge_old_jobs()
    job = _JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


# ── Background worker ──────────────────────────────────────────────────────

async def _run_job(job_id: str, req: ClipJobRequest) -> None:
    job = _JOBS[job_id]
    try:
        # ── 1. Analyze ────────────────────────────────────────────────────
        job["status"] = "analyzing"
        job["progress"] = "Analyzing video and selecting clips…"

        analyze_req = AnalyzeRequest(
            url=req.url,
            duration=req.duration,
            api_key=req.api_key,
            api_provider=req.api_provider,
            model=req.model,
            custom_prompt=req.custom_focus,
            range_start=req.range_start,
            range_end=req.range_end,
            subtitles=req.subtitles,
            subtitles_filename=req.subtitles_filename,
            target_clip_count=req.target_clip_count,
            proxy=req.proxy,
            cookies=req.cookies,
            transcript_language=req.transcript_language,
            content_type=req.content_type,
            output_language=req.output_language,
        )

        analyze_result = await _collect_analyze(analyze_req)
        if analyze_result is None:
            raise RuntimeError("Analysis produced no result")

        clips: List[Dict[str, Any]] = analyze_result.get("clips", [])
        source_title: str = analyze_result.get("title", "")
        transcript = analyze_result.get("transcript") or []
        video_url: str = analyze_result.get("video_url") or req.url
        video_id: str = analyze_result.get("video_id") or ""

        job["source_title"] = source_title

        if not clips:
            job["status"] = "done"
            job["progress"] = "Done — no clips found"
            job["clips"] = []
            return

        # Populate result clips with analyze metadata immediately so callers
        # can see titles even before rendering finishes.
        job["clips"] = [
            {
                "index": i,
                "title": c.get("title", f"Clip {i+1}"),
                "caption": c.get("caption_suggestion", ""),
                "hashtag": c.get("hashtag_suggestion", ""),
                "virality_score": c.get("virality_score", 0),
                "start_time": c.get("start_time", 0.0),
                "end_time": c.get("end_time", 0.0),
                "key_quotes": c.get("key_quotes", []),
                "download_url": None,
                "render_status": "pending" if req.render else "skipped",
                "error": None,
            }
            for i, c in enumerate(clips)
        ]

        if not req.render:
            job["status"] = "done"
            job["progress"] = "Done (render skipped)"
            return

        # ── 2. Render ─────────────────────────────────────────────────────
        job["status"] = "rendering"
        job["progress"] = f"Rendering {len(clips)} clip(s)…"

        effective_design = req.design or req.content_type  # "podcast" | "film"
        auto_film_label = (f"🎬 {source_title}" if source_title else None) if effective_design == "film" else None
        settings = RenderSettingsModel(
            aspect_ratio=req.aspect_ratio,
            design=effective_design,
            subtitles_enabled=req.enable_subtitles,
            caption_style="viral_pop" if req.enable_subtitles else "none",
            film_label=auto_film_label,
        )

        batch_id = uuid.uuid4().hex[:10]
        clips_status: List[Dict[str, Any]] = []
        for i, c in enumerate(clips):
            base_t = (
                c.get("title_suggestion") or c.get("title") or f"Clip {i+1}"
            ).strip()
            clips_status.append({
                "index": i,
                "title": base_t,
                "base_title": base_t,
                "status": "pending",
                "progress_percent": 0,
                "download_url": None,
                "output_path": None,
                "error": None,
                "error_message": None,
                # carry through so we can read them back after render
                "caption_suggestion": c.get("caption_suggestion", ""),
                "hashtag_suggestion": c.get("hashtag_suggestion", ""),
                "virality_score": c.get("virality_score", 0),
                "start_time": c.get("start_time", 0.0),
                "end_time": c.get("end_time", 0.0),
                "key_quotes": c.get("key_quotes", []),
            })

        RENDER_BATCHES[batch_id] = {
            "overall_status": "running",
            "total_clips": len(clips),
            "clips": clips_status,
            "current_clip_index": 0,
        }
        BATCH_REQUESTS[batch_id] = RenderBatchRequest(
            video_url=video_url,
            video_id=video_id,
            clips=clips,
            settings=settings,
            transcript=transcript,
            cookies=req.cookies,
        )

        # monitor render progress on a side task
        monitor = asyncio.create_task(
            _monitor_render(job_id, batch_id, len(clips))
        )
        try:
            await process_batch_rendering(batch_id, BATCH_REQUESTS[batch_id])
        finally:
            monitor.cancel()
            try:
                await monitor
            except asyncio.CancelledError:
                pass

        # ── 3. Package results ────────────────────────────────────────────
        batch = RENDER_BATCHES.get(batch_id, {})
        result_clips = []
        for cs in batch.get("clips", []):
            result_clips.append({
                "index": cs.get("index", 0),
                "title": cs.get("title", ""),
                "caption": cs.get("caption_suggestion", ""),
                "hashtag": cs.get("hashtag_suggestion", ""),
                "virality_score": cs.get("virality_score", 0),
                "start_time": cs.get("start_time", 0.0),
                "end_time": cs.get("end_time", 0.0),
                "key_quotes": cs.get("key_quotes", []),
                "download_url": cs.get("download_url"),
                "render_status": cs.get("status", "unknown"),
                "error": cs.get("error_message") or cs.get("error"),
            })

        job["clips"] = result_clips
        job["status"] = "done"
        job["progress"] = "Done"

    except Exception as exc:
        logger.exception(f"Clip job {job_id} failed: {exc}")
        job["status"] = "error"
        job["progress"] = "Failed"
        job["error"] = str(exc)


async def _collect_analyze(req: AnalyzeRequest) -> Optional[Dict[str, Any]]:
    """Run analyze_video and collect the final SSE result dict."""
    response = await analyze_video(req)
    final: Optional[Dict[str, Any]] = None
    async for chunk in response.body_iterator:
        text = chunk if isinstance(chunk, str) else chunk.decode()
        for line in text.splitlines():
            if not line.startswith("data:"):
                continue
            try:
                ev = json.loads(line[5:])
            except json.JSONDecodeError:
                continue
            if ev.get("error"):
                raise RuntimeError(ev.get("message") or "Analysis error")
            if ev.get("done"):
                final = ev.get("result")
    return final


async def _monitor_render(job_id: str, batch_id: str, total: int) -> None:
    """Update job progress string while render runs."""
    while True:
        await asyncio.sleep(3)
        batch = RENDER_BATCHES.get(batch_id)
        if not batch:
            break
        done = sum(
            1 for c in batch.get("clips", [])
            if c.get("status") in ("completed", "error")
        )
        job = _JOBS.get(job_id)
        if job:
            job["progress"] = f"Rendering: {done}/{total} clips done"


def _purge_old_jobs() -> None:
    cutoff = time.time() - _JOB_TTL
    stale = [jid for jid, j in _JOBS.items() if j.get("created_at", 0) < cutoff]
    for jid in stale:
        del _JOBS[jid]
