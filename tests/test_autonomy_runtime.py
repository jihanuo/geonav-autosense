"""Unit tests for the bounded autonomous loop (T10, T13, T15, budgets).

Heavy dependencies (torch-backed agent, Qwen helper) are stubbed so the loop
can be exercised deterministically without GPUs or network access.
"""
import json
import os
import sys
import tempfile
import types
import unittest
from collections import namedtuple

import numpy as np

# ---------------------------------------------------------------- stubs first
_UTILS = types.ModuleType("utils")
_UTILS.__path__ = []
_QWEN = types.ModuleType("utils.QwenAPI")
_QWEN.encode_image_from_pil = lambda image: "ENCODED_IMAGE"
sys.modules.setdefault("utils", _UTILS)
sys.modules["utils.QwenAPI"] = _QWEN

_GS = types.ModuleType("gsamllavanav")
_GS.__path__ = []
_SPACE = types.ModuleType("gsamllavanav.space")


class _Point(namedtuple("_Point", "x y")):
    __slots__ = ()

    def dist_to(self, other):
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5


class Pose4D(namedtuple("Pose4D", "x y z yaw")):
    __slots__ = ()

    @property
    def xy(self):
        return _Point(self.x, self.y)


class Point3D(namedtuple("Point3D", "x y z")):
    __slots__ = ()

    @property
    def xy(self):
        return _Point(self.x, self.y)


_SPACE.Pose4D = Pose4D
_SPACE.Point3D = Point3D
sys.modules.setdefault("gsamllavanav", _GS)
sys.modules["gsamllavanav.space"] = _SPACE

import scenegraphnav  # noqa: E402  (namespace package from the repo root)

_AGENT_STUB = types.ModuleType("scenegraphnav.agent")
_AGENT_STUB.extract_json_from_msg = lambda msg: json.loads(msg) if str(msg).strip().startswith("{") else None
sys.modules["scenegraphnav.agent"] = _AGENT_STUB
scenegraphnav.agent = _AGENT_STUB

from scenegraphnav.autonomy.runtime import run_autonomous_episode  # noqa: E402


# ------------------------------------------------------------------- fixtures
class FakeDetections:
    def __init__(self):
        self.xyxy = [[1, 2, 3, 4]]
        self.confidence = [0.5]


class FakeLayer:
    def __init__(self):
        self.detections = None
        self.phrases = None
        self.obj_list = []
        self.pose = None
        self.image_bgr = None
        self.update_calls = 0
        self.gsam_params = types.SimpleNamespace(use_segmentation_mask=False)

    def update_observation(self, pose, image_bgr, depth, strategy=""):
        self.update_calls += 1
        self.pose = pose
        self.image_bgr = image_bgr
        if float((image_bgr > 0).mean()) < 0.15:
            self.detections = None
            self.phrases = None
            return self
        self.detections = FakeDetections()
        self.phrases = ["car"]
        return self


class FakeSceneGraph:
    def format_summary(self, as_dict=False):
        return {"nodes": 1}


class FakeController:
    def __init__(self, target, success_dist):
        self.pose = Pose4D(0.0, 0.0, 50.0, 0.0)
        self.timestep = 0
        self.target = target
        self.success_dist = success_dist
        self.perceive_calls = 0
        self.route_calls = 0
        self.query_calls = 0
        self.understand_calls = 0
        self.scene_graph = FakeSceneGraph()
        self._rgb = np.full((16, 16, 3), 200, dtype=np.uint8)

    def perceive(self, pose, map_name):
        self.perceive_calls += 1
        self.timestep += 1
        return self._rgb, None

    def reached_target(self, pose, target):
        return pose.xy.dist_to(target.xy) <= self.success_dist

    def build_geo_nodes(self, landmarks):
        self.route_calls += 1
        return "church northeast"

    def understand(self, image_b64, episode):
        self.understand_calls += 1
        return {"nodes": [{"id": "n1", "object_type": "car", "bbox": [1, 2, 3, 4]}], "edges": []}

    def build_scene_graph(self, graph, layer):
        return None

    def query_scene_graph(self, query, debug=False):
        self.query_calls += 1
        node = types.SimpleNamespace(id="car_0", obj_class="car", position=_Point(15.0, 25.0))
        return [node], {"operation_chain": []}


class FakeNavMap:
    def __init__(self):
        self.bounds = (0.0, 0.0, 400.0, 400.0)
        self.ground_level = 0.0
        self.trajectory = []
        self.tracking_map = types.SimpleNamespace(mark_current_view_area=lambda pose: None)
        self.landmark_map = types.SimpleNamespace(landmarks=[])
        self.target_map = FakeLayer()
        self.surroundings_map = FakeLayer()


class FakeAgent:
    def __init__(self, script, args):
        self.args = args
        self.episode = types.SimpleNamespace(id="ep_test", map_name="map", target_description="go to the red car")
        self.target = Point3D(100.0, 0.0, 0.0)
        self.controller = FakeController(self.target, args.success_dist)
        self.landmark_nav_map = FakeNavMap()
        self.results = {"target": (100.0, 0.0), "steps": [], "scene_graph_summary": "", "query_logs": []}
        self.save_path = tempfile.mkdtemp()
        self.prompts = []
        self.script = list(script)
        self.saved = None
        self.move_calls = 0

    def call_response(self, system, prompt, images):
        self.prompts.append(prompt)
        if not self.script:
            return json.dumps({"type": "finish", "name": "finish", "arguments": {}, "reason": "done"})
        return self.script.pop(0)

    def get_next_position(self, mode, target_json):
        self.move_calls += 1
        if mode == "Locate":
            return tuple(target_json["selected_pos"])
        deltas = {"east": (10, 0), "west": (-10, 0), "north": (0, 10), "south": (0, -10), "stop": (0, 0)}
        dx, dy = deltas.get(target_json.get("movement", "stop"), (0, 0))
        scale = 3 if mode == "Search" else 1
        return self.controller.pose.x + dx / scale, self.controller.pose.y + dy / scale

    def gen_map(self, kind):
        return None

    def save_results(self, success):
        self.saved = success


def make_args(tmpdir, **overrides):
    defaults = dict(
        eval_max_timestep=5,
        success_dist=20.0,
        max_tools_per_step=3,
        max_total_llm_calls=100,
        max_episode_seconds=60.0,
        locate_max_step_m=80.0,
        output_dir=tmpdir + os.sep,
    )
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


def decision(kind, name, arguments=None, reason="r"):
    return json.dumps({"type": kind, "name": name, "arguments": arguments or {}, "reason": reason})


class AutonomousRuntimeTest(unittest.TestCase):
    def _run(self, script, **arg_overrides):
        tmpdir = tempfile.mkdtemp()
        args = make_args(tmpdir, **arg_overrides)
        agent = FakeAgent(script, args)
        success, pose_log = run_autonomous_episode(agent)
        return agent, success, pose_log

    def test_t13_three_tools_in_one_step_advance_a_single_timestep(self):
        script = [
            decision("tool", "get_geo_context"),
            decision("navigate", "navigate", {"mode": "Navigate", "movement": "east"}),
            decision("tool", "detect_objects"),
            decision("tool", "query_hsg", {"query": "red car"}),
            decision("locate", "locate", {"selected_pos": [15.0, 25.0]}),
            decision("finish", "finish"),
        ]
        agent, _success, pose_log = self._run(script)

        self.assertEqual(agent.controller.perceive_calls, 3)
        self.assertEqual(agent.controller.timestep, 3)
        self.assertEqual(len(pose_log), 3)
        stats = agent.results["autonomy_stats"]
        self.assertEqual(stats["n_steps"], 3)
        self.assertEqual(stats["n_llm_calls"], 6)
        self.assertEqual(stats["n_gdino_calls"], 2)
        self.assertEqual(stats["n_hsg_calls"], 0)
        self.assertEqual(stats["n_map_queries"], 1)
        self.assertEqual(agent.controller.query_calls, 1)
        self.assertEqual(agent.results["runtime_mode"], "autonomous")
        self.assertEqual(stats["termination"], "finish")

    def test_t11_heavy_perception_only_runs_when_requested(self):
        script = [
            decision("navigate", "navigate", {"mode": "Navigate", "movement": "east"}),
            decision("navigate", "navigate", {"mode": "Search", "movement": "north"}),
            decision("finish", "finish"),
        ]
        agent, _success, _pose_log = self._run(script)
        self.assertEqual(agent.landmark_nav_map.target_map.update_calls, 0)
        self.assertEqual(agent.controller.understand_calls, 0)
        self.assertEqual(agent.results["autonomy_stats"]["n_gdino_calls"], 0)

    def test_t10_invalid_json_is_repaired_without_moving(self):
        script = [
            "this is not json",
            decision("navigate", "navigate", {"mode": "Navigate", "movement": "east"}),
        ]
        agent, _success, pose_log = self._run(script, eval_max_timestep=1)
        self.assertEqual(len(pose_log), 1)
        stats = agent.results["autonomy_stats"]
        self.assertEqual(stats["n_invalid_model_responses"], 1)
        self.assertEqual(agent.controller.pose.x, 10.0)
        self.assertEqual(agent.move_calls, 1)

    def test_tool_budget_exhaustion_recovers_with_navigation(self):
        script = [
            decision("tool", "detect_objects"),
            decision("tool", "query_hsg", {"query": "car"}),
            decision("navigate", "navigate", {"mode": "Navigate", "movement": "east"}),
        ]
        agent, _success, _pose_log = self._run(script, max_tools_per_step=1)
        self.assertEqual(agent.landmark_nav_map.target_map.update_calls, 1)
        self.assertEqual(agent.controller.query_calls, 0)
        self.assertGreaterEqual(agent.results["autonomy_stats"]["n_invalid_actions"], 1)
        self.assertEqual(agent.controller.pose.x, 10.0)

    def test_locate_outside_hsg_candidates_is_rejected(self):
        script = [
            decision("locate", "locate", {"selected_pos": [300.0, 300.0]}),
            decision("navigate", "navigate", {"mode": "Search", "movement": "east"}),
        ]
        agent, _success, _pose_log = self._run(script)
        self.assertGreaterEqual(agent.results["autonomy_stats"]["n_invalid_actions"], 1)
        self.assertAlmostEqual(agent.controller.pose.x, 10.0 / 3.0, places=5)

    def test_finish_terminates_episode(self):
        agent, success, pose_log = self._run([decision("finish", "finish")])
        self.assertEqual(len(pose_log), 1)
        self.assertEqual(agent.controller.timestep, 1)
        self.assertEqual(agent.results["autonomy_stats"]["termination"], "finish")
        self.assertFalse(success)
        self.assertIs(agent.saved, False)

    def test_t15_ground_truth_never_reaches_the_model_context(self):
        script = [
            decision("tool", "get_geo_context"),
            decision("navigate", "navigate", {"mode": "Navigate", "movement": "east"}),
        ]
        agent, _success, _pose_log = self._run(script)
        joined = "\n".join(agent.prompts)
        self.assertNotIn("100.0", joined)
        self.assertNotIn("target_position", joined)

    def test_trace_file_is_written_per_step(self):
        script = [decision("finish", "finish")]
        agent, _success, _pose_log = self._run(script)
        trace = agent.results["autonomy_log"]
        with open(trace, encoding="utf-8") as handle:
            lines = [json.loads(line) for line in handle.read().splitlines()]
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0]["step"], 1)
        self.assertTrue(lines[0]["stopped"])

    def test_no_move_when_both_attempts_are_invalid(self):
        script = ["not json", "also not json", decision("navigate", "navigate",
                                                        {"mode": "Navigate", "movement": "east"})]
        agent, _success, pose_log = self._run(script, eval_max_timestep=1)
        self.assertEqual(len(pose_log), 1)
        self.assertEqual(agent.controller.pose.x, 0.0)
        self.assertEqual(agent.move_calls, 0)
        self.assertEqual(agent.results["autonomy_stats"]["no_move_steps"], 1)


if __name__ == "__main__":
    unittest.main()
