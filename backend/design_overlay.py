"""Simple fixed designs for Auto Clipper: 'podcast' and 'film'.

Both render a transparent 1080x1920 PNG overlay (red title box with highlighted words).
The video itself is rendered as 16:9 centered on a black 9:16 canvas (aspect_ratio="16:9").
"""
import os
import re
from typing import List, Optional, Tuple

from PIL import Image, ImageDraw

from backend.video_engine import get_font, get_emoji_font, split_text_and_emojis

CANVAS_W, CANVAS_H = 1080, 1920
RED = (153, 0, 0, 255)
WHITE = (255, 255, 255, 255)
LIME = (204, 255, 0, 255)
BLACK = (0, 0, 0, 255)
FONT_NAME = "Inter"

# Video occupies y=656..1264 (16:9 centered), see reference designs.
PODCAST_BOX = (50, 362, 1030, 548)      # x0, y0, x1, y1  (symmetric left/right)
FILM_BOX = (50, 1278, 1030, 1600)
FILM_LABEL_H = 62
_RADIUS = 14          # rounded corner radius
_ACCENT = 7           # left accent bar width (lime strip)


def _parse_words(title: str) -> List[Tuple[str, bool]]:
    """Return [(word, highlighted)]. Words wrapped as *word* are highlighted.
    If no markup exists: highlight the first word (and last two words if title is long)."""
    title = (title or "").strip()
    if "*" in title:
        out: List[Tuple[str, bool]] = []
        for tok in re.findall(r"\*[^*]+\*|\S+", title):
            if tok.startswith("*") and tok.endswith("*") and len(tok) > 2:
                for w in tok.strip("*").split():
                    out.append((w, True))
            else:
                out.append((tok.replace("*", ""), False))
        return out
    words = title.split()
    hl = set()
    if words:
        hl.add(0)
    if len(words) >= 5:
        hl.update({len(words) - 2, len(words) - 1})
    return [(w, i in hl) for i, w in enumerate(words)]


def _text_w(draw, text, font) -> float:
    return draw.textlength(text, font=font)


def _wrap(draw, words: List[Tuple[str, bool]], font, max_w: float) -> List[List[Tuple[str, bool]]]:
    lines: List[List[Tuple[str, bool]]] = []
    cur: List[Tuple[str, bool]] = []
    for w, hl in words:
        trial = " ".join(x[0] for x in cur + [(w, hl)])
        if cur and _text_w(draw, trial, font) > max_w:
            lines.append(cur)
            cur = [(w, hl)]
        else:
            cur.append((w, hl))
    if cur:
        lines.append(cur)
    return lines


def _draw_title_lines(draw, lines, font, x: float, y: float, line_h: int, center_w: Optional[float] = None):
    space = _text_w(draw, " ", font)
    for li, line in enumerate(lines):
        lx = x
        if center_w is not None:
            total = sum(_text_w(draw, w, font) for w, _ in line) + space * (len(line) - 1)
            lx = x + (center_w - total) / 2
        for w, hl in line:
            draw.text((lx, y + li * line_h), w, font=font, fill=LIME if hl else WHITE)
            lx += _text_w(draw, w, font) + space


def _fit(draw, words, max_w, max_lines, start_size, min_size=34):
    size = start_size
    while True:
        font = get_font(FONT_NAME, size)
        lines = _wrap(draw, words, font, max_w)
        if len(lines) <= max_lines or size <= min_size:
            return font, lines, size
        size -= 4


def _draw_label(draw, x, y, w, h, label: str):
    draw.rounded_rectangle((x, y, x + w, y + h), radius=h // 2, fill=WHITE)
    font = get_font(FONT_NAME, 32)
    emoji_font = get_emoji_font(28)
    cx = x + 22
    for kind, chunk in split_text_and_emojis(label):
        if kind == "emoji" and emoji_font:
            try:
                draw.text((cx, y + 12), chunk, font=emoji_font, embedded_color=True)
                cx += draw.textlength(chunk, font=emoji_font) + 4
                continue
            except Exception:
                pass
            continue
        draw.text((cx, y + h / 2), chunk, font=font, fill=BLACK, anchor="lm")
        cx += _text_w(draw, chunk, font)
    return cx


def render_design_overlay_png(
    design: str,
    title_text: str,
    output_png_path: str,
    film_label: Optional[str] = None,
) -> Optional[str]:
    if design not in ("podcast", "film") or not (title_text or "").strip():
        return None
    img = Image.new("RGBA", (CANVAS_W, CANVAS_H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    words = _parse_words(title_text)

    if design == "podcast":
        x0, y0, x1, y1 = PODCAST_BOX
        pad = 28
        inner_w = (x1 - x0) - pad * 2 - _ACCENT - 4
        font, lines, size = _fit(draw, words, inner_w, 3, 58)
        line_h = int(size * 1.25)
        h = max(y1 - y0, len(lines) * line_h + pad * 2)
        # keep the box bottom anchored above the video
        y_top = y1 - h if h > (y1 - y0) else y0
        draw.rounded_rectangle((x0, y_top, x1, y_top + h), radius=_RADIUS, fill=RED)
        draw.rounded_rectangle((x0, y_top, x0 + _ACCENT, y_top + h), radius=_RADIUS, fill=LIME)
        text_x = x0 + _ACCENT + 4 + pad
        text_y = y_top + (h - len(lines) * line_h) / 2
        _draw_title_lines(draw, lines, font, text_x, text_y, line_h)
    else:
        x0, y0, x1, y1 = FILM_BOX
        pad = 26
        label = (film_label or "").strip()
        inner_w = (x1 - x0) - pad * 2 - _ACCENT - 4
        font, lines, size = _fit(draw, words, inner_w, 3, 60)
        line_h = int(size * 1.2)
        top_block = (FILM_LABEL_H + 24) if label else 0
        h = max(y1 - y0, top_block + len(lines) * line_h + pad * 2)
        draw.rounded_rectangle((x0, y0, x1, y0 + h), radius=_RADIUS, fill=RED)
        draw.rounded_rectangle((x0, y0, x0 + _ACCENT, y0 + h), radius=_RADIUS, fill=LIME)
        text_x = x0 + _ACCENT + 4 + pad
        if label:
            lw = min((x1 - x0) - pad * 2, int(draw.textlength(label, font=get_font(FONT_NAME, 32))) + 72)
            _draw_label(draw, text_x, y0 + 18, lw, FILM_LABEL_H, label)
        _draw_title_lines(draw, lines, font, text_x, y0 + 18 + top_block + 14, line_h)

    os.makedirs(os.path.dirname(output_png_path), exist_ok=True)
    img.save(output_png_path, "PNG")
    return output_png_path
