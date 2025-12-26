"""
Unit tests for evaluation module.
"""
import unittest
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from src import evaluation

class TestEvaluation(unittest.TestCase):
    
    def test_calculate_test_metrics(self):
        """Test calculation of MSE, RMSE, MAE, R2."""
        # Create a dummy model that predicts perfectly
        X_test = np.array([[1], [2], [3]])
        y_test = np.array([2, 4, 6])
        
        pipeline = Pipeline([('model', LinearRegression())])
        pipeline.fit(X_test, y_test)
        
        metrics = evaluation.calculate_test_metrics(pipeline, X_test, y_test)
        
        self.assertAlmostEqual(metrics['test_mse'], 0.0)
        self.assertAlmostEqual(metrics['test_rmse'], 0.0)
        self.assertAlmostEqual(metrics['test_r2'], 1.0)

    def test_select_best_model(self):
        """Test selection of the best model based on MSE."""
        results = [
            {'run': 1, 'test_mse': 0.5, 'config': 'A'},
            {'run': 2, 'test_mse': 0.2, 'config': 'B'}, # Best
            {'run': 3, 'test_mse': 0.8, 'config': 'C'}
        ]
        
        best = evaluation.select_best_model(results, metric='test_mse')
        self.assertEqual(best['run'], 2)
        self.assertEqual(best['config'], 'B')

if __name__ == '__main__':
    unittest.main()
