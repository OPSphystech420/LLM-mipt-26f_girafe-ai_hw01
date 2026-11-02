import unittest

from retrieval import tokenize


class MathSearchTests(unittest.TestCase):
    def test_pdf_and_latex_integrals_share_search_terms(self):
        self.assertEqual(tokenize(r"\int_\gamma e^z/z^3 dz"), tokenize("∫_γ e^{z}/z^3 dz"))

    def test_bound_and_zero_notation_supply_search_terms(self):
        self.assertIn("ограничен", tokenize(r"|f(z)|\le1"))
        self.assertIn("нул", tokenize("f(1/n)=0"))

    def test_conjugation_is_retained(self):
        self.assertIn("сопряжен", tokenize(r"\partial f / \partial \bar{z}"))


if __name__ == "__main__":
    unittest.main()
