import unittest
import numpy as np

from src.deduplication import cluster_embeddings_hdbscan, detect_duplicates

class TestDeduplication(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(detect_duplicates([], 0.5), [])

    def test_detects_exact_duplicates_case_insensitive(self):
        duplicates = detect_duplicates(["Elastic Search", "elastic search", "Kibana"], 0.99)

        self.assertEqual(len(duplicates), 1)
        self.assertEqual(duplicates[0]["left_index"], 0)
        self.assertEqual(duplicates[0]["right_index"], 1)

    def test_rejects_invalid_threshold(self):
        with self.assertRaises(ValueError):
            detect_duplicates(["a", "b"], 1.1)

    def test_hdbscan_wrapper_handles_too_few_samples(self):
        embeddings = np.array([[1.0, 0.0]])

        labels = cluster_embeddings_hdbscan(embeddings, min_cluster_size=2)

        np.testing.assert_array_equal(labels, np.array([-1]))

if __name__ == '__main__':
    unittest.main()
