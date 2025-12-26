"""
Unit tests for data preprocessing module.
"""
import unittest
import pandas as pd
import numpy as np
from src import preprocessing

class TestPreprocessing(unittest.TestCase):
    
    def setUp(self):
        """Create sample data for testing."""
        self.raw_data = pd.DataFrame({
            'Spotify Streams': ['1,000', '2,500', np.nan, '500'],
            'Artist': ['Artist A', np.nan, 'Artist B', 'Artist C'],
            'Irrelevant': [1, 2, 3, 4]
        })

    def test_clean_numeric_columns(self):
        """Test if commas are removed and NaNs are filled with median."""
        cleaned_df = preprocessing.clean_numeric_columns(
            self.raw_data, ['Spotify Streams']
        )
        
        # Check types
        self.assertTrue(pd.api.types.is_numeric_dtype(cleaned_df['Spotify Streams']))
        
        # Check values (comma removal)
        self.assertEqual(cleaned_df['Spotify Streams'].iloc[0], 1000)
        
        # Check median imputation (1000, 2500, 500 -> median is 1000)
        self.assertEqual(cleaned_df['Spotify Streams'].iloc[2], 1000)

    def test_drop_irrelevant_columns(self):
        """Test column dropping."""
        df = preprocessing.drop_irrelevant_columns(
            self.raw_data, ['Irrelevant']
        )
        self.assertNotIn('Irrelevant', df.columns)
        self.assertIn('Artist', df.columns)

    def test_fill_categorical_missing(self):
        """Test categorical imputation."""
        df = preprocessing.fill_categorical_missing(
            self.raw_data, col='Artist', value='Unknown'
        )
        self.assertEqual(df['Artist'].iloc[1], 'Unknown')

    def test_log_transformation(self):
        """Test log1p transformation."""
        y = pd.Series([0, 10, 100])
        y_log = preprocessing.apply_log_transformation(y)
        
        # log1p(0) = 0
        self.assertEqual(y_log[0], 0)
        # log1p(10) ≈ 2.397
        self.assertAlmostEqual(y_log[1], np.log(11), places=4)

if __name__ == '__main__':
    unittest.main()
