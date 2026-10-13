"""Verify the submitted executed notebook in a fresh Python namespace."""
import json
from pathlib import Path
import unittest
import nbformat
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parents[1]

class ProjectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = nbformat.read(ROOT / 'Student_Exam_Performance_Predictor.ipynb', as_version=4)
        cls.results = json.loads((ROOT / 'outputs/results.json').read_text())
        # Replay the actual notebook implementation; avoid any dependency on saved state.
        import os
        import tempfile
        import matplotlib
        matplotlib.use('Agg')
        cls.state = {}
        old_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as temporary:
            try:
                os.chdir(temporary)
                for cell in cls.notebook.cells:
                    if cell.cell_type == 'code':
                        exec(compile(cell.source, '<notebook cell>', 'exec'), cls.state)
            finally:
                os.chdir(old_cwd)

    def test_supplied_dataset_and_derived_column(self):
        np.testing.assert_array_equal(self.state['study_hours'], [1,2,2.5,3,4,4.5,5,6,6.5,7,8,9,10])
        np.testing.assert_array_equal(self.state['exam_scores'], [42,48,51,55,62,65,69,75,78,82,87,91,95])
        self.assertFalse(self.state['df'].isna().any().any())
        self.assertEqual(int(self.state['df']['AboveAverage'].sum()), 6)
        self.assertEqual(len(self.state['stakeholder_rows']), 6)

    def test_exact_split_and_training_only_fit(self):
        s = self.state
        expected = train_test_split(np.arange(13), test_size=0.2, random_state=42)
        np.testing.assert_array_equal(s['train_indices'], expected[0])
        np.testing.assert_array_equal(s['test_indices'], expected[1])
        self.assertTrue(set(s['train_indices']).isdisjoint(s['test_indices']))
        reference = LinearRegression().fit(s['X'][expected[0]], s['y'][expected[0]])
        np.testing.assert_allclose(s['model'].coef_, reference.coef_)
        self.assertAlmostEqual(s['model'].intercept_, reference.intercept_)

    def test_predictions_and_independent_mae(self):
        s = self.state
        expected = s['model'].intercept_ + s['model'].coef_[0] * s['X_test'][:, 0]
        np.testing.assert_allclose(s['predictions'], expected)
        self.assertEqual(len(expected), 3)
        self.assertAlmostEqual(s['mae'], sum(abs(s['y_test'] - expected)) / 3)
        for hours, key in [(7.5, 'prediction_7_5'), (100, 'prediction_100')]:
            self.assertAlmostEqual(s[key], s['model'].intercept_ + s['model'].coef_[0] * hours)
        for key in ['mae', 'coefficient', 'intercept', 'prediction_7_5', 'prediction_100']:
            value = s['model'].coef_[0] if key == 'coefficient' else s['model'].intercept_ if key == 'intercept' else s[key]
            self.assertAlmostEqual(self.results[key], value)

    def test_executed_outputs_and_conclusion(self):
        nbformat.validate(self.notebook)
        codes = [c for c in self.notebook.cells if c.cell_type == 'code']
        self.assertEqual([c.execution_count for c in codes], list(range(1, len(codes) + 1)))
        self.assertFalse(any(o.output_type == 'error' for c in codes for o in c.outputs))
        image_count = sum('image/png' in o.get('data', {}) for c in codes for o in c.outputs)
        self.assertGreaterEqual(image_count, 2)
        count = len(self.state['conclusion'].split())
        self.assertTrue(150 <= count <= 250)
        self.assertEqual(self.results['conclusion_word_count'], count)
        for name in ['study_hours_scatter.png', 'study_hours_regression.png', 'test_predictions.csv']:
            self.assertTrue((ROOT / 'outputs' / name).is_file())

if __name__ == '__main__':
    unittest.main()
