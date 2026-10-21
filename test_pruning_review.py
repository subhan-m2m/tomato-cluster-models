import unittest
from evaluate_pruning_review import evaluate
from pruning_association import VERSION, SETTINGS, associate


class ReviewMetricsTests(unittest.TestCase):
    def reference(self):
        association = associate([{"box": [20,20,30,30], "score": .99}, {"box": [35,20,45,30], "score": .98}],
                                [{"box": [10,10,60,60], "score": .99}], 200, 100)
        return {"association_version": VERSION, "settings": SETTINGS, "samples": [{
            "file_name": "synthetic.jpg", "partition": "development", "reviewed": True,
            "association": association,
            "reference_clusters": [{"id": "r1", "box": [10,10,60,60], "visible_fruit_count": 2},
                                   {"id": "missed", "box": [100,10,150,60], "visible_fruit_count": 3}],
            "memberships": [{"fruit_id": f["id"], "reference_cluster_id": "r1", "status": "assignable"} for f in association["tomatoes"]]}]}

    def test_missed_cluster_counts_in_flag_recall(self):
        result = evaluate(self.reference(), 1, "development")
        self.assertEqual(result["assignment_accuracy_on_assignable_detected_fruit"], 1)
        self.assertEqual(result["matched_cluster_count_mae"], 0)
        self.assertEqual(result["missed_clusters"], 1)
        self.assertEqual(result["over_limit_recall_including_missed_clusters"], .5)

    def test_unreviewed_and_missing_counts_rejected(self):
        ref = self.reference()
        ref["samples"][0]["reviewed"] = False
        with self.assertRaises(ValueError):
            evaluate(ref, 1, "development")
        ref["samples"][0]["reviewed"] = True
        ref["samples"][0]["reference_clusters"][0]["visible_fruit_count"] = None
        with self.assertRaises(ValueError):
            evaluate(ref, 1, "development")


if __name__ == "__main__":
    unittest.main()
