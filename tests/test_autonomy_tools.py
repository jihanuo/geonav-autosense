"""Unit tests for the whitelisted tool adapters (T07-T09, T11, T12, T15).

The fakes below mimic the parts of ``GeonavAgent`` the executor touches, so
these tests run without torch, GroundingDINO or any network access.
"""
import unittest
from types import SimpleNamespace

import numpy as np
from PIL import Image

# ``tools`` only needs PIL; keep the import surface explicit.
from scenegraphnav.autonomy.tools import ToolExecutor


class FakeDetections:
    def __init__(self):
        self.xyxy = [[10, 20, 30, 40], [50, 60, 70, 80]]
        self.confidence = [0.72, 0.31]


class FakeLayer:
    def __init__(self, use_mask=False):
        self.detections = None
        self.phrases = None
        self.obj_list = []
        self.pose = None
        self.image_bgr = None
        self.update_calls = 0
        self.gsam_params = SimpleNamespace(use_segmentation_mask=use_mask)

    def update_observation(self, pose, image_bgr, depth, strategy=""):
        self.update_calls += 1
        self.pose = pose
        self.image_bgr = image_bgr
        if float((image_bgr > 0).mean()) < 0.15:
            self.detections = None
            self.phrases = None
            return self
        self.detections = FakeDetections()
        self.phrases = ["red car", "tree"]
        return self


class FakeSceneGraph:
    def format_summary(self, as_dict=False):
        return {"nodes": 2}


class FakeController:
    def __init__(self, build_scene_graph_assertion=None):
        self.pose = SimpleNamespace(x=10.0, y=20.0, z=50.0, yaw=0.0)
        self.timestep = 1
        self.scene_graph = FakeSceneGraph()
        self.build_scene_graph_assertion = build_scene_graph_assertion
        self.geo_calls = 0
        self.understand_calls = 0
        self.query_calls = 0

    def build_geo_nodes(self, landmarks):
        self.geo_calls += 1
        return "church is northeast of the UAV"

    def understand(self, image_b64, episode):
        self.understand_calls += 1
        return {"nodes": [{"id": "n1", "object_type": "red car", "bbox": [1, 2, 3, 4]}],
                "edges": [{"source": "n1", "target": "geo"}]}

    def build_scene_graph(self, graph, layer):
        if self.build_scene_graph_assertion is not None:
            self.build_scene_graph_assertion(layer)

    def query_scene_graph(self, query, debug=False):
        self.query_calls += 1
        node = SimpleNamespace(id="car_3", obj_class="red car", position=SimpleNamespace(x=15.0, y=25.0))
        return [node], {"operation_chain": [{"op": "query"}]}


class FakeAgent:
    def __init__(self, controller=None):
        self.controller = controller or FakeController()
        self.results = {}
        self.episode = SimpleNamespace(target_description="Find the red car east of the church")
        self.target = SimpleNamespace(x=999.0, y=999.0)
        self.save_path = "/tmp"
        self.gen_map_calls = 0
        self.landmark_nav_map = SimpleNamespace(
            landmark_map=SimpleNamespace(landmarks=["church"]),
            target_map=FakeLayer(),
            surroundings_map=FakeLayer(),
        )

    def gen_map(self, kind):
        self.gen_map_calls += 1
        return Image.new("RGB", (32, 24), (12, 34, 56))


def sample_rgb(value=200):
    return np.full((16, 16, 3), value, dtype=np.uint8)


class ToolExecutorTest(unittest.TestCase):
    def setUp(self):
        self.agent = FakeAgent()
        self.executor = ToolExecutor(self.agent)
        self.rgb = sample_rgb()

    def test_geo_context_returns_text_and_no_ground_truth(self):
        result, image = self.executor.execute("get_geo_context", {}, self.rgb, "b64")
        self.assertTrue(result["ok"])
        self.assertIsNone(image)
        self.assertIn("church", result["data"]["geo_context"])
        self.assertNotIn("999", str(result))

    def test_t12_render_map_returns_base64_image(self):
        result, image = self.executor.execute("render_map", {"kind": "landmark"}, self.rgb, "b64")
        self.assertTrue(result["ok"])
        self.assertTrue(result["data"]["visual_result"])
        self.assertIsInstance(image, str)
        self.assertGreater(len(image), 100)

    def test_t12_render_map_accepts_base64_from_plot(self):
        agent = FakeAgent()
        agent.gen_map = lambda kind: "BASE64MAP"
        result, image = ToolExecutor(agent).execute("render_map", {"kind": "semantic"}, self.rgb, "b64")
        self.assertTrue(result["ok"])
        self.assertEqual(image, "BASE64MAP")

    def test_t07_detect_objects_runs_once_per_step_only(self):
        result, _ = self.executor.execute("detect_objects", {}, self.rgb, "b64")
        self.assertTrue(result["ok"])
        self.assertEqual(self.executor.counts["n_gdino_calls"], 2)
        self.assertEqual(len(result["data"]["targets"]), 2)

        duplicate, _ = self.executor.execute("detect_objects", {}, self.rgb, "b64")
        self.assertFalse(duplicate["ok"])
        self.assertEqual(duplicate["error"], "duplicate_same_step")
        self.assertEqual(self.executor.counts["n_gdino_calls"], 2)

    def test_t09_empty_detection_returns_empty_list(self):
        blank = np.zeros((16, 16, 3), dtype=np.uint8)
        result, _ = self.executor.execute("detect_objects", {}, blank, "b64")
        self.assertTrue(result["ok"])
        self.assertTrue(result["data"]["empty_image"])
        self.assertEqual(result["data"]["targets"], [])
        self.assertEqual(self.executor.counts["n_gdino_calls"], 0)

    def test_t08_extract_hsg_sets_image_and_pose_without_gdino(self):
        seen = {}

        def assertion(layer):
            seen["pose"] = layer.pose
            seen["image"] = layer.image_bgr

        agent = FakeAgent(FakeController(build_scene_graph_assertion=assertion))
        executor = ToolExecutor(agent)
        result, _ = executor.execute("extract_hsg", {}, self.rgb, "b64")
        self.assertTrue(result["ok"])
        self.assertEqual(result["data"]["nodes_added_or_seen"], 1)
        self.assertIsNotNone(seen["pose"])
        self.assertEqual(seen["image"].shape, (16, 16, 3))
        self.assertEqual(agent.landmark_nav_map.target_map.update_calls, 0)
        self.assertEqual(executor.counts["n_gdino_calls"], 0)
        self.assertEqual(executor.counts["n_hsg_calls"], 1)

    def test_query_hsg_returns_candidate_coordinates(self):
        result, _ = self.executor.execute("query_hsg", {"query": "red car east of church"}, self.rgb, "b64")
        self.assertTrue(result["ok"])
        self.assertEqual(result["data"]["matches"][0]["xy"], [15.0, 25.0])
        self.assertEqual(self.executor.counts["n_map_queries"], 1)

    def test_unknown_tool_raises(self):
        with self.assertRaises(ValueError):
            self.executor.execute("rm_rf", {}, self.rgb, "b64")

    def test_tool_exception_is_returned_as_structured_error(self):
        def boom(_layer):
            raise RuntimeError("projection failed")

        agent = FakeAgent(FakeController(build_scene_graph_assertion=boom))
        result, image = ToolExecutor(agent).execute("extract_hsg", {}, self.rgb, "b64")
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "RuntimeError")
        self.assertIsNone(image)

    def test_t11_no_heavy_calls_when_not_requested(self):
        executor = ToolExecutor(self.agent)
        executor.execute("get_geo_context", {}, self.rgb, "b64")
        self.assertEqual(self.agent.landmark_nav_map.target_map.update_calls, 0)
        self.assertEqual(self.agent.controller.understand_calls, 0)
        self.assertEqual(executor.counts["n_gdino_calls"], 0)
        self.assertEqual(executor.counts["n_hsg_calls"], 0)


if __name__ == "__main__":
    unittest.main()
