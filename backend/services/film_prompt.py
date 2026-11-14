"""Prompt pieces for Film mode (movies / series / drama clips).

Podcast prompts optimise for a talking host ("attribute to the host/channel").
Films need scene-based selection: story beats, characters, emotion, action —
including moments with no dialogue at all.
"""
from typing import Any, Dict, List, Optional

from backend.utils.heatmap import get_average_heatmap_value

NO_DIALOGUE_TAG = "[NO DIALOGUE"
MIN_GAP_SEC = 5.0


def build_film_analysis_lines(
    enriched_transcript: List[Dict[str, Any]],
    heatmap: Optional[List[Dict[str, Any]]],
    start_bound: float,
    end_bound: float,
) -> List[Dict[str, Any]]:
    """Dialogue lines plus synthetic markers for gaps without dialogue (action, music, visual
    storytelling) so the model can pick scenes a transcript alone would hide.
    Markers are for the prompt only; they never enter the returned clip transcripts."""
    out: List[Dict[str, Any]] = []
    cursor = float(start_bound)
    for line in enriched_transcript:
        ls, le = float(line["start"]), float(line["end"])
        if ls - cursor >= MIN_GAP_SEC:
            out.append(_gap_line(cursor, ls, heatmap))
        out.append(line)
        cursor = max(cursor, le)
    if end_bound and end_bound - cursor >= MIN_GAP_SEC:
        out.append(_gap_line(cursor, end_bound, heatmap))
    return out


def _gap_line(gs: float, ge: float, heatmap) -> Dict[str, Any]:
    score = get_average_heatmap_value(gs, ge, heatmap) if heatmap else 0.0
    return {
        "start": round(gs, 2),
        "end": round(ge, 2),
        "text": f"{NO_DIALOGUE_TAG} {int(ge - gs)}s — music, action or visual scene]",
        "engagement": round(score, 3),
    }


def film_system_instruction(output_lang_name: str, spoken_lang_name: str) -> str:
    return (
        "You are an expert film & series clip editor for TikTok / Reels / Shorts movie-moment accounts. "
        "Select the most gripping standalone scenes from the provided film transcript.\n"
        "Strict rules:\n"
        f"1. Write title, title_suggestion, caption_suggestion, hashtag_suggestion and summary in {output_lang_name}. "
        f"key_quotes stay verbatim in the original spoken language ({spoken_lang_name}).\n"
        "2. Titles are third-person scene descriptions or curiosity hooks; never first person, never refer to a host or channel.\n"
        "3. You MUST respond ONLY with a valid JSON object matching this schema:\n"
        "{\n"
        '  "summary": "1-2 sentence description of the film/episode, followed by 2-4 hashtags",\n'
        '  "clips": [\n'
        "    {\n"
        '      "title": "Hook title under 8 words",\n'
        '      "start_time": 10.0,\n'
        '      "end_time": 45.0,\n'
        '      "hook_time": 12.0,\n'
        '      "virality_score": 92,\n'
        '      "key_quotes": ["quote 1"],\n'
        '      "title_suggestion": "Alternative hook title",\n'
        '      "caption_suggestion": "Social media post caption",\n'
        '      "hashtag_suggestion": "#hashtag1 #hashtag2"\n'
        "    }\n"
        "  ]\n"
        "}"
    )


def build_film_prompt(
    *,
    transcript_text: str,
    clip_range: str,
    clip_count_instruction: str,
    window_note: str,
    title: str,
    start_bound: float,
    end_bound: float,
    duration: float,
    dur_range: str,
    has_heatmap: bool,
    spoken_lang_name: str,
    spoken_lang_code: str,
    output_lang_name: str,
    duration_instruction: str,
    custom_focus: str,
) -> str:
    heat = (
        "Columns: start|end|interest(0-1). Interest = audience replay peaks or audio intensity; high values often mark action, shouting, music swells or emotional peaks."
        if has_heatmap else "No interest data — judge by story beats and dialogue."
    )
    if output_lang_name == spoken_lang_name:
        lang_rule = (
            f"LANGUAGE: The film is spoken in {spoken_lang_name} ('{spoken_lang_code}'). Write ALL fields in {spoken_lang_name}. Do not translate.\n"
        )
    else:
        lang_rule = (
            f"LANGUAGE: The film is spoken in {spoken_lang_name} ('{spoken_lang_code}'), but the audience reads {output_lang_name}.\n"
            f"- Write title, title_suggestion, caption_suggestion, hashtag_suggestion and summary in natural, native-sounding {output_lang_name}.\n"
            f"- key_quotes MUST stay verbatim in {spoken_lang_name}, copied exactly from the transcript.\n"
        )
    focus = ""
    if custom_focus:
        focus = (
            f"USER FOCUS: Prioritise scenes matching: \"{custom_focus}\" (still standalone and engaging).\n\n"
        )
    return (
        "You are an expert film & series clip editor for TikTok, Instagram Reels and YouTube Shorts movie-moment accounts.\n"
        f"Analyze this film/episode transcript and find {clip_range} gripping, standalone scene clips.\n\n"
        f"{window_note}"
        "CONTEXT:\n"
        f"- Title / file: {title}\n"
        f"- Range: {int(start_bound)}s to {int(end_bound)}s (Total: {int(duration)}s) | Target clip length: {dur_range}\n"
        f"- {heat}\n"
        f"- Lines tagged {NO_DIALOGUE_TAG} ...] are stretches with no speech (action, chase, fight, music, silent emotional or visual moments). They CAN be excellent clips, especially with high interest.\n\n"
        f"{lang_rule}\n"
        "WHAT MAKES A GREAT FILM CLIP:\n"
        "- Strong scene types: confrontations and arguments, plot twists and reveals, iconic or quotable lines, emotional/heartbreaking moments, funny moments, romantic tension, suspense and jump scares, action set pieces, satisfying payoffs.\n"
        "- It must be understandable without the rest of the film: include the setup that makes the payoff land.\n"
        "- Start where the scene or exchange begins (never mid-conversation) and end right after the payoff or reaction beat, not mid-sentence.\n"
        "- Each clip should be a different scene; do not return near-duplicates.\n"
        "- hook_time = the moment that grabs attention (reveal, punchline, first hit).\n\n"
        "TITLES:\n"
        "- Describe the situation or tease the payoff in third person, max 8 words (e.g. 'Ketika Rahasianya Akhirnya Terbongkar', 'He Never Saw This Twist Coming').\n"
        "- Use character names only if they appear in the dialogue; otherwise use roles ('Sang Detektif', 'The Mother').\n"
        "- Never first person, never mention a host, channel, uploader or file name.\n\n"
        f"{duration_instruction}\n\n"
        f"{clip_count_instruction}\n\n"
        f"{focus}"
        f"Transcript (start|end[|interest] text):\n---\n{transcript_text}\n---\n\n"
        "Rules: use exact seconds from the transcript; clips must not overlap.\n"
        "Return clips sorted by virality_score desc."
    )
