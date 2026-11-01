import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prepare_data import (  # noqa: E402
    EVIDENCE_ANCHORS,
    EVIDENCE_ALTERNATIVES,
    Glyph,
    clean_text,
    decode_glyph,
    extract_corpus,
    parse_qa,
    render_glyphs,
)


class TextExtractionTests(unittest.TestCase):
    def test_cyrillic_font_decoding_does_not_change_math(self):
        self.assertEqual(decode_glyph("Òåîðåìà", "F50"), "Теорема")
        self.assertEqual(decode_glyph("φ", "CMMI10"), "φ")
        self.assertEqual(decode_glyph("Z", "CMEX10"), "∫")
        self.assertEqual(decode_glyph("Z", "CMMI10"), "Z")
        self.assertEqual(decode_glyph("p", "CMEX10"), "√")
        self.assertEqual(clean_text("влеч\u0451т"), "влечет")

    def test_fraction_keeps_conjugation_in_denominator(self):
        glyphs = [
            Glyph("∂f", 0, 12, 10, 10, "CMMI10"),
            Glyph("∂", 0, 5, 23.5, 10, "CMMI10"),
            Glyph("z", 5, 11, 23.5, 10, "CMMI10"),
            Glyph("=", 18, 25, 17, 10, "CMR10"),
            Glyph("0", 28, 33, 17, 10, "CMR10"),
        ]
        text = render_glyphs(glyphs, [(0, 12, 14.25), (5, 11, 17.8)])
        self.assertIn("(∂f)/(∂\\bar{z})", text)
        self.assertIn("= 0", text)

    def test_exponent_and_subscript_have_distinct_markers(self):
        glyphs = [Glyph("z", 0, 5, 20, 10, "CMMI10"), Glyph("2", 5, 8, 16, 7, "CMR7"), Glyph("a", 12, 18, 20, 10, "CMMI10"), Glyph("n", 18, 22, 21.5, 7, "CMMI7")]
        text = render_glyphs(glyphs, [])
        self.assertIn("z^{2}", text)
        self.assertIn("a_{n}", text)


class PublicQAParserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        markers = {}
        anchor_sets = list(EVIDENCE_ANCHORS.values()) + [anchors for variants in EVIDENCE_ALTERNATIVES.values() for anchors in variants]
        for anchors in anchor_sets:
            for page, marker in anchors:
                markers.setdefault(page, set()).add(marker)
        corpus = [{"block_id": f"page_{page}", "page": page, "section": "test", "text": "\n".join(sorted(values))} for page, values in markers.items()]
        cls.questions, cls.gold = parse_qa((ROOT / "tfkp_15_qa.md").read_text(encoding="utf-8"), corpus)

    def test_questions_do_not_include_labels(self):
        self.assertEqual(len(self.questions), 15)
        self.assertEqual([q["id"] for q in self.questions], [f"q_{i:03d}" for i in range(1, 16)])
        for question in self.questions:
            self.assertEqual(set(question), {"id", "question", "options"})
            self.assertEqual(len(question["options"]), 4)
            self.assertEqual([o["id"] for o in question["options"]], ["A", "B", "C", "D"])
        text = json.dumps(self.questions, ensure_ascii=False)
        self.assertNotIn("\u0451", text)
        self.assertNotIn("\u0401", text)

    def test_public_answers_and_category_coverage(self):
        expected = ["BC", "AC", "BC", "BC", "AB", "AD", "C", "B", "AC", "ABD", "AC", "AC", "", "", ""]
        self.assertEqual(["".join(g["answer_ids"]) for g in self.gold], expected)
        self.assertEqual({g["category"] for g in self.gold}, {"conditions", "comparison", "calculation", "insufficient"})
        self.assertEqual(sum(g["abstain"] for g in self.gold), 3)
        for record in self.gold[-3:]:
            self.assertTrue(record["missing_information"])
            self.assertEqual(record["answer_ids"], [])


@unittest.skipUnless((ROOT / ".local/textbook.pdf").is_file(), "Source PDF was not downloaded")
class ActualPDFTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = extract_corpus(ROOT / ".local/textbook.pdf")
        cls.by_page = {}
        for block in cls.corpus:
            cls.by_page.setdefault(block["page"], []).append(block)

    def test_all_pages_and_unique_ids(self):
        self.assertEqual(set(self.by_page), set(range(1, 148)))
        self.assertEqual(len(self.corpus), len({b["block_id"] for b in self.corpus}))
        for block in self.corpus:
            self.assertEqual(set(block), {"block_id", "page", "section", "text"})
            self.assertTrue(block["text"].strip())
            self.assertNotIn("\u0451", block["text"])

    def test_visually_checked_formulas(self):
        page12 = "\n".join(b["text"] for b in self.by_page[12])
        page34 = "\n".join(b["text"] for b in self.by_page[34])
        page36 = "\n".join(b["text"] for b in self.by_page[36])
        self.assertIn("(∂f)/(∂\\bar{z}) = 0", page12)
        self.assertIn("(∂f)/(∂x) + i(∂f)/(∂y)", page12)
        self.assertIn("(n!)/(2πi)", page34)
        self.assertIn("((ζ − z)^{n+1})", page34)
        self.assertIn("c_{n} = (f^{(n)}(z_{0}))/(n!)", page36)
        self.assertIn("∑_{n=0}^{∞}", page36)
        page20 = "\n".join(b["text"] for b in self.by_page[20])
        page28 = "\n".join(b["text"] for b in self.by_page[28])
        page35 = "\n".join(b["text"] for b in self.by_page[35])
        page38 = "\n".join(b["text"] for b in self.by_page[38])
        self.assertIn("\\sqrt[n]{n!}", page20)
        self.assertIn("\\overline{e^{z}} = e^{\\bar{z}}", page20)
        self.assertIn("(dz)/(z − a) = i∫_{0}^{2π} dt = 2πi", page28)
        self.assertIn("\\overline{O_{r}(a)}", page35)
        self.assertIn("предельную точку a, принадлежащую области D", page38)

    def test_gold_anchors_exist_and_contain_full_statement(self):
        _, gold = parse_qa((ROOT / "tfkp_15_qa.md").read_text(encoding="utf-8"), self.corpus)
        blocks = {b["block_id"]: b for b in self.corpus}
        for row in gold:
            self.assertTrue(row["evidence_sets"])
            for sources in row["evidence_sets"]:
                self.assertLessEqual(len(sources), 4)
                self.assertTrue(set(sources) <= blocks.keys())
        polynomial = blocks[gold[10]["evidence_sets"][0][0]]["text"]
        self.assertIn("полиномом степени не выше", polynomial)
        cauchy = blocks[gold[5]["evidence_sets"][0][-1]]["text"]
        self.assertIn("бесконечно дифференцируемую", cauchy)

    def test_rebuild_is_deterministic(self):
        rebuilt = extract_corpus(ROOT / ".local/textbook.pdf")
        self.assertEqual(self.corpus, rebuilt)


if __name__ == "__main__":
    unittest.main()
