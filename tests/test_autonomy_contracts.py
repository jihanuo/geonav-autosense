"""Unit tests for the AutoSense decision contract (T01-T06)."""
import math
import unittest

from scenegraphnav.autonomy.contracts import Decision, approve_locate, parse_decision


class ParseDecisionTest(unittest.TestCase):
    def test_t01_unknown_tool_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_decision({"type": "tool", "name": "exec_shell", "arguments": {}})

    def test_t02_invalid_render_kind_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_decision({"type": "tool", "name": "render_map", "arguments": {"kind": "foo"}})

    def test_t03_navigate_east_is_valid(self):
        decision = parse_decision({
            "type": "navigate", "name": "navigate",
            "arguments": {"mode": "Search", "movement": "east"}, "reason": "explore",
        })
        self.assertEqual(decision.type, "navigate")
        self.assertEqual(decision.arguments["movement"], "east")

    def test_t04_injection_like_direction_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_decision({"type": "navigate", "name": "navigate",
                            "arguments": {"mode": "Search", "movement": "DROP_TABLE"}})

    def test_t05_nan_locate_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_decision({"type": "locate", "name": "locate",
                            "arguments": {"selected_pos": [float("nan"), 1.0]}})

    def test_arguments_must_be_object(self):
        with self.assertRaises(ValueError):
            parse_decision({"type": "tool", "name": "detect_objects", "arguments": []})

    def test_unknown_decision_type_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_decision({"type": "python", "name": "os.system", "arguments": {}})

    def test_finish_accepts_no_arguments(self):
        with self.assertRaises(ValueError):
            parse_decision({"type": "finish", "name": "finish", "arguments": {"x": 1}})

    def test_query_hsg_length_limit(self):
        with self.assertRaises(ValueError):
            parse_decision({"type": "tool", "name": "query_hsg", "arguments": {"query": "a" * 300}})

    def test_tool_with_unexpected_arguments_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_decision({"type": "tool", "name": "detect_objects", "arguments": {"path": "/etc/passwd"}})

    def test_reason_is_truncated(self):
        decision = parse_decision({"type": "finish", "name": "finish", "arguments": {}, "reason": "x" * 1000})
        self.assertEqual(len(decision.reason), 256)

    def test_bool_is_not_a_valid_coordinate(self):
        with self.assertRaises(ValueError):
            parse_decision({"type": "locate", "name": "locate", "arguments": {"selected_pos": [True, 1.0]}})


class ApproveLocateTest(unittest.TestCase):
    def _locate(self, x, y):
        return Decision("locate", "locate", {"selected_pos": [x, y]})

    def test_t06_candidate_outside_tolerance_is_rejected(self):
        with self.assertRaises(ValueError):
            approve_locate(self._locate(500.0, 500.0), [[10.0, 10.0]], current_xy=[0.0, 0.0])

    def test_candidate_from_query_is_accepted(self):
        x, y = approve_locate(self._locate(11.0, 9.5), [[10.0, 10.0]], current_xy=[0.0, 0.0])
        self.assertAlmostEqual(x, 11.0)
        self.assertAlmostEqual(y, 9.5)

    def test_empty_candidates_are_rejected(self):
        with self.assertRaises(ValueError):
            approve_locate(self._locate(10.0, 10.0), [], current_xy=[0.0, 0.0])

    def test_one_step_distance_limit(self):
        with self.assertRaises(ValueError):
            approve_locate(self._locate(10.0, 10.0), [[10.0, 10.0]], current_xy=[0.0, 0.0], max_dist=5.0)

    def test_non_locate_decision_is_rejected(self):
        with self.assertRaises(ValueError):
            approve_locate(Decision("finish", "finish", {}), [[10.0, 10.0]])

    def test_distance_is_finite_for_large_candidates(self):
        with self.assertRaises(ValueError):
            approve_locate(self._locate(math.inf, 0.0), [[0.0, 0.0]])


if __name__ == "__main__":
    unittest.main()
