import unittest

from concept_retrofit.training.synthetic_smoke import run


class SyntheticSmokeTests(unittest.TestCase):
    def test_end_to_end_bottleneck_smoke(self):
        report = run(steps=300)
        self.assertTrue(report.passed, report)


if __name__ == "__main__":
    unittest.main()
