"""Rebuild the PDF corpus and the public QA set without generative models."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, replace
import json
from pathlib import Path
import re
import unicodedata

import pymupdf


@dataclass(frozen=True)
class Glyph:
    text: str
    x0: float
    x1: float
    y: float
    size: float
    font: str


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.translate(str.maketrans({"\u0451": "е", "\u0401": "Е", "ﬁ": "fi", "ﬂ": "fl", "\u00a0": " "}))
    text = re.sub(r"([а-яА-Я])-\s*\n\s*([а-яА-Я])", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def decode_glyph(text: str, font: str) -> str:
    # Type3 Cyrillic fonts in this PDF expose cp1251 bytes as Latin-1.
    if re.fullmatch(r"F\d+", font):
        text = "".join(bytes([ord(c)]).decode("cp1251", errors="replace") if 0xC0 <= ord(c) <= 0xFF or ord(c) in {0xA8, 0xB8} else c for c in text)
        text = text.replace("\x16", "—").replace("\x15", "—").replace("\x9f", "§")
    if font.startswith("CMEX"):
        text = text.translate(str.maketrans({"Z": "∫", "R": "∫", "p": "√", "X": "∑", "P": "∑", "Y": "∏", "Q": "∏", "\x12": "(", "\x13": ")", "\x10": "(", "\x11": ")", "\x00": "(", "\x01": ")", "\x0c": "|", "\uf8eb": "(", "\uf8ed": "", "\uf8ec": "", "\uf8f6": ")", "\uf8f8": "", "\uf8f7": ""}))
    text = "".join(c for c in text if c in "\n\t" or ord(c) >= 32)
    return clean_text(text) if text.strip() else " "


def page_glyph_blocks(page: pymupdf.Page) -> list[tuple[list[Glyph], tuple]]:
    result = []
    for block in page.get_text("rawdict", sort=False)["blocks"]:
        if block["type"] != 0:
            continue
        chars = []
        for line in block["lines"]:
            for span in line["spans"]:
                font = span["font"]
                for ch in span["chars"]:
                    text = decode_glyph(ch["c"], font)
                    if not text:
                        continue
                    y = ch["origin"][1]
                    # CMEX display operators use the top of the glyph as origin.
                    if font.startswith("CMEX"):
                        if text in {"∫", "∑", "∏"}:
                            y += (1.36 if ch["c"] == "Z" else 0.75 if ch["c"] in {"P", "Q"} else 0.95 if text != "∫" else 0.55) * span["size"]
                        elif text in {"(", ")"}:
                            y += 1.4 * span["size"]
                    chars.append(Glyph(text, ch["bbox"][0], ch["bbox"][2], y, span["size"], font + ("-inline" if font.startswith("CMEX") and ch["c"] in {"P", "Q", "R"} else "")))
        if chars:
            result.append((chars, block["bbox"]))
    return result


def horizontal_rules(page: pymupdf.Page) -> list[tuple[float, float, float]]:
    rules = []
    for drawing in page.get_drawings():
        for item in drawing["items"]:
            if item[0] == "l":
                a, b = item[1:]
                if abs(a.y - b.y) < 0.25 and 2 < abs(a.x - b.x) < 180:
                    rules.append((min(a.x, b.x), max(a.x, b.x), (a.y + b.y) / 2))
    return sorted(set(rules), key=lambda r: (r[1] - r[0], r[2], r[0]))


def attach_overbars(glyphs: list[Glyph], rules: list[tuple]) -> list[Glyph]:
    current = list(glyphs)
    for x0, x1, y in rules:
        inside = [g for g in current if g.text.strip() and not g.font.startswith("F") and x0 - 0.5 <= (g.x0 + g.x1) / 2 <= x1 + 0.5]
        if any(2.0 <= y - g.y <= 6.2 and g.size >= 8.5 for g in inside):
            continue  # A fraction has a numerator above its rule.
        if any(g.text == "√" and x0 - 12 < g.x0 < x0 + 1 and abs(g.y - y) < 15 for g in current):
            continue
        below = [g for g in inside if 1.0 <= g.y - y <= 13.5 and g.font.startswith(("CM", "MS"))]
        if not below:
            continue
        base = max(below, key=lambda g: g.size)
        if len(below) == 1 and re.fullmatch(r"[a-zA-Zζ]", base.text):
            text = "\\bar{" + base.text + "}"
        elif base.size >= 8.5:
            text = "\\overline{" + render_row(below, base.y) + "}"
        else:
            continue
        used = set(below)
        current = [g for g in current if g not in used]
        current.append(Glyph(text, x0, x1, base.y, base.size, base.font))
    return current


def render_row(glyphs: list[Glyph], baseline: float | None = None) -> str:
    if not glyphs:
        return ""
    main = [g for g in glyphs if g.text.strip() and g.size >= 8.5 and not g.font.startswith("CMSY7")]
    if baseline is None:
        counts = Counter(round(g.y * 2) / 2 for g in main or glyphs)
        baseline = counts.most_common(1)[0][0]
    pieces = []
    ordered = sorted(glyphs, key=lambda g: (g.x0, g.y))
    last_x = None
    script_kind = None
    script = []

    def flush_script() -> None:
        nonlocal script_kind, script
        if script:
            pieces.append(script_kind + "{" + "".join(script).strip() + "}")
        script_kind, script = None, []

    for g in ordered:
        if not g.text.strip():
            if script:
                continue
            if pieces and not pieces[-1].endswith(" "):
                pieces.append(" ")
            continue
        kind = None
        if g.size < 8.5 and g.font.startswith(("CM", "MS")) and g.text not in {"′", "□"}:
            if g.y < baseline - 1.0:
                kind = "^"
            elif g.y > baseline + 0.5:
                kind = "_"
        if kind:
            if script_kind != kind:
                flush_script()
                script_kind = kind
            script.append(g.text)
        else:
            flush_script()
            if last_x is not None and g.x0 - last_x > 2.2 and pieces and not pieces[-1].endswith(" "):
                pieces.append(" ")
            pieces.append(g.text)
        last_x = max(last_x or g.x1, g.x1)
    flush_script()
    return clean_text("".join(pieces))


def collapse_fractions(glyphs: list[Glyph], rules: list[tuple]) -> list[Glyph]:
    current = list(glyphs)
    for x0, x1, y in rules:
        inside = [g for g in current if g.text.strip() and x0 - 0.6 <= (g.x0 + g.x1) / 2 <= x1 + 0.6]
        # Baseline distances distinguish fraction rules from overbars.
        above = [g for g in inside if 2.0 <= y - g.y <= 6.2 and not g.font.startswith("F")]
        below = [g for g in inside if 5.0 <= g.y - y <= 11.8 and not g.font.startswith("F")]
        if not above or not below:
            continue
        top_y = min(g.y for g in above)
        bot_y = max(g.y for g in below)
        numerator = [g for g in inside if top_y - 6.0 <= g.y <= y - 1.2 and not g.font.startswith("F")]
        denominator = [g for g in inside if y + 3.0 <= g.y <= bot_y + 3.0 and not g.font.startswith("F")]
        if not numerator or not denominator:
            continue
        num = render_row(numerator, top_y)
        den = render_row(denominator, bot_y)
        text = "(" + num + ")/(" + den + ")"
        used = set(numerator + denominator)
        current = [g for g in current if g not in used]
        current.append(Glyph(text, x0, x1, y + 2.5, 9.96, "fraction"))
    return current


def attach_radicals(glyphs: list[Glyph], rules: list[tuple]) -> list[Glyph]:
    current = list(glyphs)
    for root in [g for g in current if g.text == "√"]:
        bars = [r for r in rules if abs(r[0] - root.x1) < 1.2 and abs(r[2] - root.y) < 2]
        if not bars:
            continue
        x0, x1, y = min(bars, key=lambda r: abs(r[0] - root.x1))
        content = [g for g in current if g is not root and g.text.strip() and x0 - 0.5 <= (g.x0 + g.x1) / 2 <= x1 + 0.5 and y + 2 < g.y < y + 14]
        if not content:
            continue
        degree = [g for g in current if g.size < 6 and root.x0 - 1 <= g.x0 < root.x1 - 1 and root.y - 1 < g.y < root.y + 6]
        baseline = max(g.y for g in content)
        text = "\\sqrt" + ("[" + "".join(g.text for g in sorted(degree, key=lambda g: g.x0)) + "]" if degree else "") + "{" + render_row(content, baseline) + "}"
        used = set(content + degree + [root])
        current = [g for g in current if g not in used]
        current.append(Glyph(text, root.x0, x1, baseline, root.size, "radical"))
    return current


def attach_operator_limits(glyphs: list[Glyph]) -> list[Glyph]:
    current = list(glyphs)
    for operator in [g for g in current if g.text in {"∫", "∑", "∏"} and not g.font.endswith("-inline")]:
        nearby = [g for g in current if g is not operator and g.size < 8.5 and g.font.startswith(("CM", "MS")) and operator.x0 - 4 <= (g.x0 + g.x1) / 2 <= operator.x1 + 5]
        lower = [g for g in nearby if 7 < g.y - operator.y < 18]
        upper = [g for g in nearby if -19 < g.y - operator.y < -7]
        if not lower and not upper:
            continue
        text = operator.text
        if lower:
            text += "_{" + "".join(g.text for g in sorted(lower, key=lambda g: g.x0)).replace(" ", "") + "}"
        if upper:
            text += "^{" + "".join(g.text for g in sorted(upper, key=lambda g: g.x0)).replace(" ", "") + "}"
        used = set(lower + upper + [operator])
        current = [g for g in current if g not in used]
        current.append(replace(operator, text=text))
    return current


def render_glyphs(glyphs: list[Glyph], rules: list[tuple]) -> str:
    glyphs = attach_operator_limits(attach_radicals(collapse_fractions(attach_overbars(glyphs, rules), rules), rules))
    large = [g for g in glyphs if g.text.strip() and g.size >= 8.5]
    baselines: list[float] = []
    for g in sorted(large, key=lambda x: x.y):
        if not baselines or g.y - baselines[-1] > 4.5:
            baselines.append(g.y)
    rows = [[] for _ in baselines]
    for g in glyphs:
        if not baselines:
            return render_row(glyphs)
        index = min(range(len(baselines)), key=lambda i: abs(g.y - baselines[i]))
        rows[index].append(g)
    return clean_text("\n".join(render_row(row, baseline) for row, baseline in zip(rows, baselines)))


def is_prose(glyphs: list[Glyph]) -> bool:
    return sum(len(re.findall("[а-яА-Я]", g.text)) for g in glyphs) >= 4


def extract_corpus(pdf_path: str | Path) -> list[dict]:
    corpus = []
    section = "Начальные страницы"
    with pymupdf.open(pdf_path) as doc:
        for page_no, page in enumerate(doc, 1):
            rules = horizontal_rules(page)
            raw = [(g, b) for g, b in page_glyph_blocks(page) if not (page_no > 3 and b[3] < 40)]
            groups: list[list[Glyph]] = []
            for glyphs, bbox in raw:
                visible = [g for g in glyphs if g.text.strip()]
                prefix = "".join(g.text for g in visible)[:80]
                starts_heading = re.match(r"^(Теорема|Следствие|Лемма|Доказательство|Замечание|Определение|§)", prefix)
                continues_left = visible and not starts_heading and (visible[0].x0 < 50 or (not visible[0].font.startswith("F") and any(g.x0 < 50 and re.search("[а-яА-Я]", g.text) for g in visible)))
                if groups and (not is_prose(glyphs) or continues_left):
                    groups[-1].extend(glyphs)
                else:
                    groups.append(list(glyphs))
            texts = [render_glyphs(g, rules) for g in groups]
            merged = []
            for text in texts:
                if not text:
                    continue
                heading = re.search(r"§\s*\d+\.[^\n]+", text)
                if heading:
                    section = heading.group(0)
                # Continue theorem statements across a displayed equation.
                continues_statement = merged and re.match(r"^(Теорема|Следствие|Лемма)\b", merged[-1][0]) and not re.match(r"^(Доказательство|Теорема|Следствие|Лемма|Замечание|Определение|§)\b", text)
                if merged and (re.match(r"^(где|коэффициенты|n\s*=)", text) or continues_statement) and len(merged[-1][0]) + len(text) < 2300:
                    merged[-1] = (merged[-1][0] + "\n" + text, merged[-1][1])
                else:
                    merged.append((text, section))
            block_no = 0
            for text, block_section in merged:
                # Long proofs are split only at physical line boundaries.
                part = []
                size = 0
                for line in text.splitlines():
                    if part and size + len(line) > 2000:
                        block_no += 1
                        corpus.append({"block_id": f"b_p{page_no:04d}_{block_no:03d}", "page": page_no, "section": block_section, "text": clean_text("\n".join(part))})
                        part, size = [], 0
                    part.append(line)
                    size += len(line) + 1
                if part:
                    block_no += 1
                    corpus.append({"block_id": f"b_p{page_no:04d}_{block_no:03d}", "page": page_no, "section": block_section, "text": clean_text("\n".join(part))})
    if not corpus or len({x["block_id"] for x in corpus}) != len(corpus):
        raise ValueError("Empty corpus or duplicate block IDs")
    return corpus


CATEGORY_MAP = {"условия": "conditions", "сопоставление": "comparison", "применение": "calculation", "недостаточность данных": "insufficient"}
# Evaluation-only anchors from the source references in the supplied QA file.
EVIDENCE_ANCHORS = {
    1: [(11, "Теорема 2.1")],
    2: [(12, "В этих терминах"), (12, "и мы приходим")],
    3: [(11, "Теорема 2.1"), (13, "Определение 2.3")],
    4: [(31, "Замечание 4.1"), (28, "2πi")],
    5: [(30, "Теорема 4.4")],
    6: [(31, "Теорема 5.1"), (33, "Следствие 5.1")],
    7: [(34, "Теорема 5.3"), (20, "Экспонента")],
    8: [(35, "Теорема 5.4")],
    9: [(11, "Теорема 2.1"), (13, "Отсюда"), (34, "Теорема 5.3")],
    10: [(36, "Теорема 6.1")],
    11: [(37, "Теорема 6.2")],
    12: [(38, "Теорема 6.3"), (21, "Тригонометрические функции"), (37, "К целым функциям")],
    13: [(37, "Следствие 6.1")],
    14: [(26, "Заметим, что"), (28, "2πi")],
    15: [(36, "Теорема 6.1"), (20, "Экспонента")],
}


# Additional sufficient source sets, audited before any model predictions.
EVIDENCE_ALTERNATIVES = {
    1: [[(11, "Доказательство. Вещественная дифференцируемость")]],
    3: [[(11, "Теорема 2.1"), (119, "Напомним, что под гармонической")]],
    4: [[(30, "Теорема 4.4"), (26, "Теорема 4.1"), (28, "2πi")]],
    6: [
        [(33, "Следствие 5.1")],
        [(31, "Пусть γ — кусочно-гладкая кривая"), (33, "Следствие 5.1")],
        [(31, "Теорема 5.1"), (32, "является голоморфной в C")],
    ],
    7: [
        [(46, "Следствие 7.3"), (46, "Следствие 7.2"), (20, "Экспонента")],
        [(54, "Теорема 9.2"), (54, "Теорема 9.1"), (20, "Экспонента")],
    ],
    9: [[(119, "Напомним, что под гармонической"), (34, "Теорема 5.3")]],
    10: [[(20, "Повторяя этот процесс")]],
    11: [[(37, "При сделанных предположениях")]],
    12: [[(39, "Следствие 6.2"), (21, "Тригонометрические функции"), (37, "К целым функциям")]],
}


def find_evidence(corpus: list[dict], page: int, marker: str) -> str:
    def normalized(s: str) -> str:
        return re.sub(r"\s+", "", s).casefold()
    candidates = [b for b in corpus if b["page"] == page and normalized(marker) in normalized(b["text"])]
    if not candidates:
        raise ValueError(f"Evidence anchor missing on PDF page {page}: {marker}")
    return min(candidates, key=lambda b: (len(b["text"]), b["block_id"]))["block_id"]


def parse_qa(text: str, corpus: list[dict]) -> tuple[list[dict], list[dict]]:
    questions, gold = [], []
    sections = list(re.finditer(r"^## (\d+)\. (.+)$", text, re.MULTILINE))
    for pos, section in enumerate(sections):
        number = int(section.group(1))
        body = text[section.end():sections[pos + 1].start() if pos + 1 < len(sections) else len(text)]
        kind = re.search(r"\*\*Тип:\*\* (.+?)\.", body).group(1)
        question_text = re.search(r"\*\*Вопрос\.\*\*\s*(.+?)(?=\n- \*\*[A-Z]\.\*\*)", body, re.DOTALL).group(1)
        options = [{"id": m.group(1), "text": clean_text(m.group(2))} for m in re.finditer(r"^- \*\*([A-Z])\.\*\* (.+)$", body, re.MULTILINE)]
        answer_match = re.search(r"\*\*Ответ:\*\* (.+?)\n", body)
        answer_text = answer_match.group(1)
        abstain = "Отказ" in answer_text
        answer_ids = [] if abstain else re.findall(r"\b[A-Z]\b", answer_text)
        rationale = body[answer_match.end():body.index("**Источник:")].strip()
        qid = f"q_{number:03d}"
        questions.append({"id": qid, "question": clean_text(question_text), "options": options})
        evidence_sets = []
        for anchors in [EVIDENCE_ANCHORS[number]] + EVIDENCE_ALTERNATIVES.get(number, []):
            evidence = list(dict.fromkeys(find_evidence(corpus, page, marker) for page, marker in anchors))
            if not any(set(evidence) == set(existing) for existing in evidence_sets):
                evidence_sets.append(evidence)
        row = {"id": qid, "category": CATEGORY_MAP[kind], "answer_ids": answer_ids, "abstain": abstain, "evidence_sets": evidence_sets, "rationale": clean_text(rationale)}
        if abstain:
            row["missing_information"] = clean_text(rationale.split(". ")[0])
        gold.append(row)
        if not 4 <= len(options) <= 6 or not set(answer_ids) <= {o["id"] for o in options}:
            raise ValueError(f"Invalid public QA: {qid}")
    if len(questions) != 15:
        raise ValueError(f"Expected 15 public questions, found {len(questions)}")
    return questions, gold


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--qa", default=str(Path(__file__).with_name("tfkp_15_qa.md")))
    parser.add_argument("--questions", default="dev_questions.jsonl")
    parser.add_argument("--gold", default="dev_gold.jsonl")
    args = parser.parse_args()
    corpus = extract_corpus(args.pdf)
    questions, gold = parse_qa(Path(args.qa).read_text(encoding="utf-8"), corpus)
    write_jsonl(args.corpus, corpus)
    write_jsonl(args.questions, questions)
    write_jsonl(args.gold, gold)
    print(json.dumps({"blocks": len(corpus), "pages": max(b["page"] for b in corpus), "questions": len(questions)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
