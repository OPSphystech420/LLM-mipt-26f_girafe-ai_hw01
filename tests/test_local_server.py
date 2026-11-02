import unittest

from serve_local import visible_content


class ThinkingOutputTests(unittest.TestCase):
    def test_direct_content_is_preserved(self):
        raw = '{"answer_ids":["A"]}'
        self.assertEqual(visible_content(raw, False), (raw, True))

    def test_reasoning_is_removed_from_content(self):
        self.assertEqual(
            visible_content('internal text</think>\n\n{"answer_ids":["B"]}', True),
            ('{"answer_ids":["B"]}', True),
        )

    def test_unfinished_reasoning_has_no_visible_content(self):
        self.assertEqual(visible_content('internal unfinished text', True), ('', False))

    def test_explicit_opening_tag_is_supported(self):
        self.assertEqual(visible_content('<think>internal</think> final', True), ('final', True))

    def test_completed_empty_answer_is_empty(self):
        self.assertEqual(visible_content('internal</think>\n', True), ('', True))


if __name__ == '__main__':
    unittest.main()
