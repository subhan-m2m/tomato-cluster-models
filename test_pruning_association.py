import unittest
from pruning_association import associate, assess


def d(box, score=0.99):
    return {"box": box, "score": score}


class AssociationTests(unittest.TestCase):
    def test_one_owner_and_real_ambiguity(self):
        result = associate([d([20, 20, 30, 30])], [d([10, 10, 60, 60]), d([15, 15, 65, 65])], 100, 100)
        self.assertEqual(result["memberships"][0]["status"], "ambiguous")
        self.assertIsNone(result["memberships"][0]["cluster_id"])

    def test_boundary_and_outside(self):
        result = associate([d([5, 20, 15, 30]), d([70, 70, 80, 80])], [d([10, 10, 60, 60])], 100, 100)
        self.assertEqual([m["status"] for m in result["memberships"]], ["assigned", "unassigned"])

    def test_duplicates_do_not_erase_overlapping_real_fruit(self):
        result = associate([d([20, 20, 40, 40]), d([20.1, 20, 40.1, 40], .95), d([25, 20, 45, 40])], [d([10, 10, 60, 60])], 100, 100)
        self.assertEqual(len(result["tomatoes"]), 2)
        self.assertEqual(len(result["suppressed_duplicates"]), 1)

    def test_original_match_beats_expanded_candidate(self):
        result = associate([d([49, 20, 53, 24])], [d([10, 10, 50, 50]), d([50, 10, 90, 50])], 100, 100, {"margin": .05})
        self.assertEqual(result["memberships"][0]["status"], "assigned")
        self.assertEqual(result["memberships"][0]["candidates"][0]["reason"], "original_box")

    def test_limit_equality_missing_rule_partial_and_edge(self):
        a = associate([d([20, 20, 30, 30])], [d([10, 10, 60, 60])], 100, 100)
        profile = {"maximum": 1, "version": 1}
        self.assertEqual(assess(a, profile)[0]["state"], "no_flag")
        self.assertEqual(assess(a, {**profile, "maximum": None})[0]["state"], "not_evaluated")
        self.assertEqual(assess(a, profile, False)[0]["state"], "not_evaluated")
        a = associate([], [d([0, 0, 20, 20])], 100, 100)
        self.assertEqual(assess(a, profile)[0]["state"], "uncertain")

    def test_stable_order_and_empty(self):
        fruit = [d([30, 30, 40, 40]), d([10, 10, 20, 20])]
        groups = [d([5, 5, 50, 50])]
        self.assertEqual(associate(fruit, groups, 100, 100), associate(list(reversed(fruit)), groups, 100, 100))
        self.assertEqual(associate([], [], 100, 100)["memberships"], [])

    def test_ambiguous_fruit_cannot_be_counted_twice_or_clear_a_borderline_group(self):
        a = associate([d([20,20,30,30]), d([45,20,55,30]), d([65,20,75,30])],
                      [d([10,10,60,60]), d([40,10,90,60])], 100, 100)
        results = assess(a, {"maximum": 1, "version": 1})
        self.assertEqual([r["visible_count"] for r in results], [1, 1])
        self.assertEqual([r["candidate_count"] for r in results], [2, 2])
        self.assertEqual([r["state"] for r in results], ["uncertain", "uncertain"])


if __name__ == "__main__":
    unittest.main()
