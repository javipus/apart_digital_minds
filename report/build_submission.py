#!/usr/bin/env python3
"""Build the sprint submission PDF from submission.md."""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image as RLImage,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "submission.md"
FIGURE_STATS = HERE / "figure_stats.json"
FIGURES = HERE / "figures"
OUTPUT_DIR = HERE.parent / "output" / "pdf"
OUTPUT = OUTPUT_DIR / "preferences-under-pressure.pdf"

NAVY = "#274C77"
BLUE = "#4F86C6"
CORAL = "#D96C5F"
GOLD = "#D6A84B"
INK = "#17212B"
MUTED = "#5D6873"
GRID = "#D9DEE3"
PAPER = "#FFFFFF"

FONT_REGULAR = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
FONT_ITALIC = "/System/Library/Fonts/Supplemental/Arial Italic.ttf"


def pil_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size=size)


def draw_text_centered(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, font, fill: str) -> None:
    box = draw.textbbox((0, 0), text, font=font)
    draw.text((xy[0] - (box[2] - box[0]) / 2, xy[1]), text, font=font, fill=fill)


def draw_panel_axes(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    y_min: float,
    y_max: float,
    ticks: list[float],
    font,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = box
    draw.line((left, bottom, right, bottom), fill=MUTED, width=2)
    for tick in ticks:
        y = bottom - (tick - y_min) / (y_max - y_min) * (bottom - top)
        draw.line((left, y, right, y), fill=GRID, width=2)
        label = f"{tick:.1f}"
        width = draw.textbbox((0, 0), label, font=font)[2]
        draw.text((left - width - 10, y - 11), label, font=font, fill=MUTED)
    return box


def make_figure_1(path: Path) -> None:
    stats = json.loads(FIGURE_STATS.read_text(encoding="utf-8"))["models"]
    image = Image.new("RGB", (1800, 690), PAPER)
    draw = ImageDraw.Draw(image)
    title_font = pil_font(FONT_BOLD, 31)
    label_font = pil_font(FONT_REGULAR, 27)
    small_font = pil_font(FONT_REGULAR, 23)
    small_bold = pil_font(FONT_BOLD, 23)

    draw.text((80, 22), "A. Stated and consequential task scores disagree", font=title_font, fill=INK)
    draw.text((80, 65), "Pearson r across the 27 BT task scores", font=small_font, fill=MUTED)
    draw.text((930, 22), "B. Stated scores predict consequential choices poorly", font=title_font, fill=INK)
    draw.text((930, 65), "Share of held-out choices predicted correctly", font=small_font, fill=MUTED)

    # Panel A: grouped Pearson correlations with response-bootstrap intervals.
    left_box = (120, 145, 820, 585)
    draw_panel_axes(draw, left_box, 0.0, 1.0, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], small_font)
    models = ["4o-mini", "Luna", "Terra"]
    series = [
        ("Stated vs tokens", CORAL, "stated__consequential_tokens"),
        ("Stated vs time", GOLD, "stated__consequential_time"),
        ("Tokens vs time", NAVY, "consequential_tokens__consequential_time"),
    ]
    group_width = 190
    bar_width = 43
    for model_index, model in enumerate(models):
        center = 230 + model_index * 235
        for series_index, (_, color, key) in enumerate(series):
            interval = stats[model]["pearson"][key]
            value = interval["point"]
            x0 = center - group_width / 2 + 18 + series_index * 58
            x1 = x0 + bar_width
            y = left_box[3] - value * (left_box[3] - left_box[1])
            draw.rounded_rectangle((x0, y, x1, left_box[3]), radius=5, fill=color)
            x_mid = (x0 + x1) / 2
            y_low = left_box[3] - interval["lower"] * (left_box[3] - left_box[1])
            y_high = left_box[3] - interval["upper"] * (left_box[3] - left_box[1])
            draw.line((x_mid, y_low, x_mid, y_high), fill=INK, width=3)
            draw.line((x_mid - 8, y_low, x_mid + 8, y_low), fill=INK, width=3)
            draw.line((x_mid - 8, y_high, x_mid + 8, y_high), fill=INK, width=3)
        draw_text_centered(draw, (center, 603), model, label_font, INK)
    legend_x = 110
    for name, color, _ in series:
        draw.rounded_rectangle((legend_x, 642, legend_x + 24, 662), radius=3, fill=color)
        draw.text((legend_x + 34, 636), name, font=small_font, fill=MUTED)
        legend_x += 240

    # Panel B: stated-transfer versus within-condition CV accuracy.
    rows = [
        ("4o-mini  tokens", "4o-mini", "consequential_tokens"),
        ("4o-mini  time", "4o-mini", "consequential_time"),
        ("Luna  tokens", "Luna", "consequential_tokens"),
        ("Luna  time", "Luna", "consequential_time"),
        ("Terra  tokens", "Terra", "consequential_tokens"),
        ("Terra  time", "Terra", "consequential_time"),
    ]
    x_left, x_right = 1125, 1745
    y_top, y_bottom = 145, 555
    for tick in [0.5, 0.6, 0.7, 0.8, 0.9]:
        x = x_left + (tick - 0.5) / 0.45 * (x_right - x_left)
        draw.line((x, y_top, x, y_bottom), fill=GRID, width=2)
        draw_text_centered(draw, (x, 570), f"{tick:.1f}", small_font, MUTED)
    for index, (label, model, target) in enumerate(rows):
        y = 170 + index * 68
        transfer = stats[model]["hard_accuracy"][f"stated__{target}"]
        within = stats[model]["hard_accuracy"][f"{target}__{target}"]
        x_transfer = x_left + (transfer["point"] - 0.5) / 0.45 * (x_right - x_left)
        x_within = x_left + (within["point"] - 0.5) / 0.45 * (x_right - x_left)
        width = draw.textbbox((0, 0), label, font=small_font)[2]
        draw.text((1100 - width, y - 13), label, font=small_font, fill=INK)
        for interval, color in [(transfer, CORAL), (within, NAVY)]:
            x_low = x_left + (interval["lower"] - 0.5) / 0.45 * (x_right - x_left)
            x_high = x_left + (interval["upper"] - 0.5) / 0.45 * (x_right - x_left)
            draw.line((x_low, y, x_high, y), fill=color, width=5)
            draw.line((x_low, y - 7, x_low, y + 7), fill=color, width=3)
            draw.line((x_high, y - 7, x_high, y + 7), fill=color, width=3)
        draw.ellipse((x_transfer - 9, y - 9, x_transfer + 9, y + 9), fill=CORAL)
        draw.ellipse((x_within - 9, y - 9, x_within + 9, y + 9), fill=NAVY)
    draw.ellipse((1120, 635, 1138, 653), fill=CORAL)
    draw.text((1148, 629), "Stated scores", font=small_font, fill=MUTED)
    draw.ellipse((1395, 635, 1413, 653), fill=NAVY)
    draw.text((1423, 629), "Same-framing scores", font=small_font, fill=MUTED)

    image.save(path, dpi=(220, 220))


def make_figure_2(path: Path) -> None:
    stats = json.loads(FIGURE_STATS.read_text(encoding="utf-8"))["models"]
    image = Image.new("RGB", (1800, 610), PAPER)
    draw = ImageDraw.Draw(image)
    title_font = pil_font(FONT_BOLD, 31)
    label_font = pil_font(FONT_REGULAR, 27)
    small_font = pil_font(FONT_REGULAR, 23)

    draw.text((80, 22), "A. Newer models repeat pairwise choices more often", font=title_font, fill=INK)
    draw.text((80, 65), "Mean pairwise preference strength; higher means more repeatable", font=small_font, fill=MUTED)
    draw.text((930, 22), "B. Their task scores predict held-out choices better", font=title_font, fill=INK)
    draw.text((930, 65), "Held-out choices predicted correctly within each framing", font=small_font, fill=MUTED)
    models = ["4o-mini", "Luna", "Terra"]
    null_strength = [0.590, 0.603, 0.602]

    panels = [
        ((120, 145, 820, 515), 0.5, 0.9, "preference_strength", null_strength),
        ((970, 145, 1700, 515), 0.5, 0.95, "hard_accuracy", None),
    ]
    for panel_index, (box, y_min, y_max, metric, null_values) in enumerate(panels):
        ticks = [0.5, 0.6, 0.7, 0.8, 0.9]
        draw_panel_axes(draw, box, y_min, y_max, ticks, small_font)
        for model_index, model in enumerate(models):
            center = box[0] + 130 + model_index * ((box[2] - box[0] - 230) / 2)
            if metric == "preference_strength":
                intervals = [
                    stats[model][metric]["consequential_tokens"],
                    stats[model][metric]["consequential_time"],
                ]
            else:
                intervals = [
                    stats[model][metric]["consequential_tokens__consequential_tokens"],
                    stats[model][metric]["consequential_time__consequential_time"],
                ]
            for offset, interval, color in [(-29, intervals[0], BLUE), (29, intervals[1], GOLD)]:
                value = interval["point"]
                y = box[3] - (value - y_min) / (y_max - y_min) * (box[3] - box[1])
                draw.rounded_rectangle((center + offset - 24, y, center + offset + 24, box[3]), radius=5, fill=color)
                y_low = box[3] - (interval["lower"] - y_min) / (y_max - y_min) * (box[3] - box[1])
                y_high = box[3] - (interval["upper"] - y_min) / (y_max - y_min) * (box[3] - box[1])
                x = center + offset
                draw.line((x, y_low, x, y_high), fill=INK, width=3)
                draw.line((x - 8, y_low, x + 8, y_low), fill=INK, width=3)
                draw.line((x - 8, y_high, x + 8, y_high), fill=INK, width=3)
                draw_text_centered(draw, (center + offset, y_high - 29), f"{value:.2f}", small_font, INK)
            draw_text_centered(draw, (center, 530), model, label_font, INK)
            if null_values is not None:
                value = null_values[model_index]
                y = box[3] - (value - y_min) / (y_max - y_min) * (box[3] - box[1])
                draw.line((center - 73, y, center + 73, y), fill=MUTED, width=4)
        if panel_index == 0:
            draw.line((665, 579, 697, 579), fill=MUTED, width=4)
            draw.text((708, 566), "Position-only null", font=small_font, fill=MUTED)

    draw.rounded_rectangle((95, 567, 119, 587), radius=3, fill=BLUE)
    draw.text((130, 561), "Tokens", font=small_font, fill=MUTED)
    draw.rounded_rectangle((255, 567, 279, 587), radius=3, fill=GOLD)
    draw.text((290, 561), "Time", font=small_font, fill=MUTED)
    draw.rounded_rectangle((1020, 567, 1044, 587), radius=3, fill=BLUE)
    draw.text((1055, 561), "Tokens", font=small_font, fill=MUTED)
    draw.rounded_rectangle((1180, 567, 1204, 587), radius=3, fill=GOLD)
    draw.text((1215, 561), "Time", font=small_font, fill=MUTED)

    image.save(path, dpi=(220, 220))


def inline_markup(text: str) -> str:
    links: list[tuple[str, str]] = []

    def replace_link(match: re.Match[str]) -> str:
        token = f"@@LINK{len(links)}@@"
        links.append((match.group(1), match.group(2)))
        return token

    text = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", replace_link, text)
    escaped = html.escape(text, quote=False)
    escaped = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(r"\*([^*]+)\*", r"<i>\1</i>", escaped)
    for index, (label, url) in enumerate(links):
        escaped = escaped.replace(
            f"@@LINK{index}@@",
            f'<link href="{html.escape(url, quote=True)}" color="{NAVY}"><u>{html.escape(label)}</u></link>',
        )
    return escaped


def register_fonts() -> None:
    pdfmetrics.registerFont(TTFont("Arial", FONT_REGULAR))
    pdfmetrics.registerFont(TTFont("Arial-Bold", FONT_BOLD))
    pdfmetrics.registerFont(TTFont("Arial-Italic", FONT_ITALIC))


def build_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "Title",
            parent=base["Title"],
            fontName="Arial-Bold",
            fontSize=23,
            leading=26,
            textColor=colors.HexColor(INK),
            alignment=TA_CENTER,
            spaceAfter=5,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            parent=base["Heading2"],
            fontName="Arial",
            fontSize=13,
            leading=16,
            textColor=colors.HexColor(NAVY),
            alignment=TA_CENTER,
            spaceAfter=6,
        ),
        "meta": ParagraphStyle(
            "Meta",
            parent=base["Normal"],
            fontName="Arial-Italic",
            fontSize=9,
            leading=11,
            textColor=colors.HexColor(MUTED),
            alignment=TA_CENTER,
            spaceAfter=12,
        ),
        "h1": ParagraphStyle(
            "H1",
            parent=base["Heading1"],
            fontName="Arial-Bold",
            fontSize=13,
            leading=15,
            textColor=colors.HexColor(NAVY),
            spaceBefore=8,
            spaceAfter=4,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "H2",
            parent=base["Heading2"],
            fontName="Arial-Bold",
            fontSize=10.5,
            leading=13,
            textColor=colors.HexColor(INK),
            spaceBefore=6,
            spaceAfter=3,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName="Arial",
            fontSize=8.7,
            leading=10.8,
            textColor=colors.HexColor(INK),
            alignment=TA_JUSTIFY,
            spaceAfter=3.9,
            allowWidows=0,
            allowOrphans=0,
        ),
        "bullet": ParagraphStyle(
            "Bullet",
            parent=base["BodyText"],
            fontName="Arial",
            fontSize=8.7,
            leading=10.8,
            textColor=colors.HexColor(INK),
            leftIndent=14,
            firstLineIndent=-8,
            bulletIndent=4,
            spaceAfter=2.2,
        ),
        "quote": ParagraphStyle(
            "Quote",
            parent=base["BodyText"],
            fontName="Arial-Italic",
            fontSize=8.7,
            leading=10.8,
            textColor=colors.HexColor(MUTED),
            leftIndent=14,
            rightIndent=14,
            spaceAfter=5,
        ),
        "caption": ParagraphStyle(
            "Caption",
            parent=base["BodyText"],
            fontName="Arial",
            fontSize=7.8,
            leading=9.5,
            textColor=colors.HexColor(MUTED),
            alignment=TA_LEFT,
            spaceBefore=2,
            spaceAfter=6,
        ),
        "reference": ParagraphStyle(
            "Reference",
            parent=base["BodyText"],
            fontName="Arial",
            fontSize=7.7,
            leading=9.3,
            textColor=colors.HexColor(INK),
            leftIndent=12,
            firstLineIndent=-12,
            spaceAfter=2.5,
        ),
        "prompt": ParagraphStyle(
            "Prompt",
            parent=base["Code"],
            fontName="Courier",
            fontSize=7.1,
            leading=8.7,
            textColor=colors.HexColor(INK),
            backColor=colors.HexColor("#F2F4F6"),
            borderColor=colors.HexColor(GRID),
            borderWidth=0.5,
            borderPadding=5,
            leftIndent=5,
            rightIndent=5,
            spaceBefore=2,
            spaceAfter=5,
        ),
        "task": ParagraphStyle(
            "Task",
            parent=base["BodyText"],
            fontName="Arial",
            fontSize=7.5,
            leading=9.1,
            textColor=colors.HexColor(INK),
            leftIndent=0,
            firstLineIndent=0,
            spaceAfter=0,
        ),
    }


def image_flowable(path: Path, width: float, caption: str, styles) -> list:
    from PIL import Image as PILImage

    with PILImage.open(path) as image:
        ratio = image.height / image.width
    return [
        Spacer(1, 3),
        RLImage(str(path), width=width, height=width * ratio),
        Paragraph(inline_markup(caption), styles["caption"]),
    ]


def parse_markdown(styles, content_width: float) -> list:
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    story: list = []
    first_h1 = True
    in_references = False
    ordered_index = 0
    line_index = 0
    while line_index < len(lines):
        raw = lines[line_index]
        line = raw.strip()
        line_index += 1
        if not line:
            ordered_index = 0
            continue
        if line.startswith("```"):
            code_lines: list[str] = []
            while line_index < len(lines) and not lines[line_index].strip().startswith("```"):
                code_lines.append(lines[line_index])
                line_index += 1
            if line_index < len(lines):
                line_index += 1
            escaped_code = html.escape("\n".join(code_lines), quote=False)
            story.append(Paragraph(
                f'<font name="Courier">{escaped_code.replace(chr(10), "<br/>")}</font>',
                styles["prompt"],
            ))
            continue
        if line == "<!-- pagebreak -->":
            story.append(PageBreak())
            continue
        if line == "<!-- appendix-top -->":
            story.append(Spacer(1, 18))
            continue
        if line == "<!-- section-top -->":
            story.append(Spacer(1, 18))
            continue
        if line == "<!-- task-list-start -->":
            task_items: list[str] = []
            while line_index < len(lines) and lines[line_index].strip() != "<!-- task-list-end -->":
                task_line = lines[line_index].strip()
                if task_line.startswith("- "):
                    task_items.append(task_line[2:])
                line_index += 1
            if line_index < len(lines):
                line_index += 1
            split = (len(task_items) + 1) // 2
            table_rows = []
            for item_index in range(split):
                left = Paragraph(inline_markup(task_items[item_index]), styles["task"])
                right_index = item_index + split
                right = (
                    Paragraph(inline_markup(task_items[right_index]), styles["task"])
                    if right_index < len(task_items)
                    else ""
                )
                table_rows.append([left, right])
            table = Table(
                table_rows,
                colWidths=[content_width / 2, content_width / 2],
                hAlign="LEFT",
            )
            table.setStyle(TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 1),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]))
            story.append(table)
            continue
        if line == "[[FIGURE1]]":
            story.extend(image_flowable(
                FIGURES / "figure1-framing-and-transfer.png",
                content_width,
                "Figure 1. Framing changes which tasks the models favor. In A, each bar compares 27 BT task scores: r = 1 means they move together and r = 0 means no linear match. In B, red uses stated-response scores to predict consequential choices; blue uses the other nine consequential responses for the same pair. Error bars are 95% response-bootstrap intervals with BT refitting for r and task-pair bootstrap intervals for accuracy.",
                styles,
            ))
            continue
        if line == "[[FIGURE2]]":
            story.extend(image_flowable(
                FIGURES / "figure2-consistency-by-model.png",
                content_width,
                "Figure 2. Consequential choices become more repeatable in newer models. In A, .5 means no stable winner and 1 means the same task always wins; gray ticks show the position-only expectation. In B, accuracy is the share of held-out choices predicted correctly using the other nine responses per pair. Error bars are 95% task-pair bootstrap intervals.",
                styles,
            ))
            continue
        if line.startswith("# "):
            style = styles["title"] if first_h1 else styles["h1"]
            story.append(Paragraph(inline_markup(line[2:]), style))
            first_h1 = False
            continue
        if line.startswith("## "):
            story.append(Paragraph(inline_markup(line[3:]), styles["subtitle"]))
            continue
        if line.startswith("### "):
            heading = line[4:]
            in_references = heading == "References"
            story.append(Paragraph(inline_markup(heading), styles["h1"]))
            continue
        if line.startswith("#### "):
            story.append(Paragraph(inline_markup(line[5:]), styles["h2"]))
            continue
        if line.startswith("> "):
            story.append(Paragraph(inline_markup(line[2:]), styles["quote"]))
            continue
        if line.startswith("- "):
            story.append(Paragraph(inline_markup(line[2:]), styles["bullet"], bulletText="-"))
            continue
        ordered = re.match(r"^(\d+)\.\s+(.*)$", line)
        if ordered:
            ordered_index += 1
            story.append(Paragraph(inline_markup(ordered.group(2)), styles["bullet"], bulletText=f"{ordered.group(1)}."))
            continue
        if line.startswith("*") and line.endswith("*") and line.count("*") == 2:
            story.append(Paragraph(inline_markup(line), styles["meta"]))
            continue
        paragraph_style = styles["reference"] if in_references else styles["body"]
        story.append(Paragraph(inline_markup(line), paragraph_style))
    return story


class SubmissionDocTemplate(BaseDocTemplate):
    def __init__(self, filename: str):
        super().__init__(
            filename,
            pagesize=A4,
            leftMargin=0.62 * inch,
            rightMargin=0.62 * inch,
            topMargin=0.62 * inch,
            bottomMargin=0.60 * inch,
            title="Preferences Under Pressure",
            author="Apart Research Digital Minds Sprint",
            subject="Consequential framing and inferred LLM task preferences",
        )
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="body")
        self.addPageTemplates(PageTemplate(id="main", frames=[frame], onPage=self.draw_page))

    def draw_page(self, canvas, doc) -> None:
        canvas.saveState()
        canvas.setFont("Arial", 7.5)
        canvas.setFillColor(colors.HexColor(MUTED))
        canvas.drawCentredString(A4[0] / 2, 22, str(doc.page))
        canvas.restoreState()


def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    make_figure_1(FIGURES / "figure1-framing-and-transfer.png")
    make_figure_2(FIGURES / "figure2-consistency-by-model.png")
    register_fonts()
    styles = build_styles()
    doc = SubmissionDocTemplate(str(OUTPUT))
    story = parse_markdown(styles, doc.width)
    doc.build(story)
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
