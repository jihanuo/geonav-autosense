"""Whitelisted adapters around the perception / query capabilities GeoNav has.

Nothing here imports torch at module import time: the heavy GroundingDINO /
SAM / VLM work only happens inside ``detect_objects`` and ``extract_hsg`` when
the runtime actually asks for it.  Every entry point returns a compact,
JSON-serialisable payload and records how often the expensive backend ran.
"""
from __future__ import annotations

import base64
import io
import time

from scenegraphnav.autonomy.contracts import TOOLS


def _pil_to_base64(image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


class ToolExecutor:
    """Execute one validated tool call against a live ``GeonavAgent``."""

    def __init__(self, agent):
        self.a = agent
        self.used = set()
        self.counts = {
            "n_gdino_calls": 0,
            "n_sam_calls": 0,
            "n_hsg_calls": 0,
            "n_map_queries": 0,
            "n_geo_context_calls": 0,
            "n_render_map_calls": 0,
            "n_detect_calls": 0,
            "n_duplicate_rejections": 0,
        }

    # ------------------------------------------------------------------ utils
    @property
    def controller(self):
        return self.a.controller

    @staticmethod
    def _summarize_detected(layer, limit: int = 30) -> list:
        detections = getattr(layer, "detections", None)
        phrases = list(getattr(layer, "phrases", None) or [])
        if detections is None:
            return []
        boxes = list(getattr(detections, "xyxy", []) or [])
        confidences = list(getattr(detections, "confidence", []) or [])
        summary = []
        for index, box in enumerate(boxes[:limit]):
            summary.append({
                "class": str(phrases[index]) if index < len(phrases) else "unknown",
                "bbox_xyxy": [round(float(value), 1) for value in box],
                "detector_score": round(float(confidences[index]), 4) if index < len(confidences) else None,
            })
        return summary

    def _is_empty_image(self, image_bgr) -> bool:
        try:
            return float((image_bgr > 0).mean()) < 0.15
        except Exception:
            return False

    # ----------------------------------------------------------------- public
    def execute(self, name: str, args: dict, rgb, rgb_b64: str):
        """Run one tool call.

        Returns ``(result_dict, optional_image_b64)``.  Errors are returned as
        structured failures instead of propagating, so one bad call cannot
        kill the episode.
        """
        if name not in TOOLS:
            raise ValueError(f"unknown tool: {name}")

        controller = self.controller
        pose = controller.pose
        step = controller.timestep
        key = (step, name, args.get("kind"), args.get("query"))
        if key in self.used:
            self.counts["n_duplicate_rejections"] += 1
            return {
                "ok": False,
                "tool": name,
                "error": "duplicate_same_step",
                "detail": "this tool was already called with the same arguments at this step",
            }, None
        self.used.add(key)

        started = time.perf_counter()
        image_out = None
        try:
            payload = self._dispatch(name, args, rgb, rgb_b64, pose)
            if name == "render_map":
                image_out = self.take_image()
            return {
                "ok": True,
                "tool": name,
                "elapsed_ms": round(1000 * (time.perf_counter() - started), 1),
                "data": payload,
            }, image_out
        except Exception as exc:  # noqa: BLE001 - tool failures must not crash the episode
            return {
                "ok": False,
                "tool": name,
                "elapsed_ms": round(1000 * (time.perf_counter() - started), 1),
                "error": type(exc).__name__,
                "detail": str(exc)[:160],
            }, None

    def _dispatch(self, name: str, args: dict, rgb, rgb_b64: str, pose):
        controller = self.controller
        navmap = self.a.landmark_nav_map

        if name == "get_geo_context":
            description = controller.build_geo_nodes(navmap.landmark_map.landmarks)
            self.counts["n_geo_context_calls"] += 1
            return {"geo_context": str(description)[:3000]}

        if name == "render_map":
            controller.build_geo_nodes(navmap.landmark_map.landmarks)
            image = self.a.gen_map(args["kind"])
            self.counts["n_render_map_calls"] += 1
            # LandmarkNavMap.plot() already returns a base64 string on this branch.
            self._last_image = image if isinstance(image, str) else _pil_to_base64(image)
            return {"kind": args["kind"], "visual_result": bool(self._last_image)}

        if name == "detect_objects":
            result = self._run_detection(navmap, pose, rgb)
            self.counts["n_detect_calls"] += 1
            return result

        if name == "extract_hsg":
            return self._extract_hsg(navmap, pose, rgb, rgb_b64)

        if name == "query_hsg":
            return self._query_hsg(navmap, args["query"])

        raise ValueError(f"unhandled tool: {name}")

    # ------------------------------------------------------------- internals
    def _run_detection(self, navmap, pose, rgb):
        image_bgr = rgb[..., ::-1]
        empty = self._is_empty_image(image_bgr)
        layers = (navmap.target_map, navmap.surroundings_map)
        if not empty:
            for layer in layers:
                layer.update_observation(pose, image_bgr, None, strategy="Search")
                self.counts["n_gdino_calls"] += 1
                if getattr(layer.gsam_params, "use_segmentation_mask", False) and getattr(layer, "detections", None):
                    self.counts["n_sam_calls"] += 1
        return {
            "targets": self._summarize_detected(navmap.target_map),
            "surroundings": self._summarize_detected(navmap.surroundings_map),
            "empty_image": empty,
            "segmentation_enabled": bool(getattr(navmap.target_map.gsam_params, "use_segmentation_mask", False)),
        }

    def _extract_hsg(self, navmap, pose, rgb, rgb_b64):
        controller = self.controller
        layer = navmap.target_map
        # build_scene_graph() projects VLM pixel boxes with the current frame,
        # so these two fields must be set even when GroundingDINO never ran.
        layer.pose = pose
        layer.image_bgr = rgb[..., ::-1]
        controller.build_geo_nodes(navmap.landmark_map.landmarks)
        graph = controller.understand(rgb_b64, self.a.episode)
        self.counts["n_hsg_calls"] += 1
        if not isinstance(graph, dict):
            raise ValueError("VLM did not return a scene-graph JSON object")
        controller.build_scene_graph(graph, layer)
        if not self.a.results.get("scene_graph_summary"):
            try:
                self.a.results["scene_graph_summary"] = controller.scene_graph.format_summary(as_dict=True)
            except Exception:
                pass
        nodes = graph.get("nodes", [])
        examples = [
            {
                "id": str(node.get("id", "")),
                "object_type": str(node.get("object_type", "")),
                "bbox": [float(v) for v in node.get("bbox", [])][:4],
            }
            for node in nodes[:8]
            if isinstance(node, dict)
        ]
        return {
            "nodes_added_or_seen": len(nodes),
            "edges_proposed": len(graph.get("edges", []) or []),
            "node_examples": examples,
        }

    def _query_hsg(self, navmap, query: str):
        controller = self.controller
        controller.build_geo_nodes(navmap.landmark_map.landmarks)
        nodes, info = controller.query_scene_graph(query, debug=False)
        self.counts["n_map_queries"] += 1
        if not isinstance(nodes, (list, tuple)):
            nodes = []
        matches = []
        for node in list(nodes)[:20]:
            position = getattr(node, "position", None)
            if position is None:
                continue
            matches.append({
                "id": str(getattr(node, "id", "")),
                "class": str(getattr(node, "obj_class", "")),
                "xy": [float(position.x), float(position.y)],
            })
        return {
            "matches": matches,
            "candidate_count": len(nodes),
            "operation_chain": (info or {}).get("operation_chain") if isinstance(info, dict) else None,
        }

    # ------------------------------------------------------------------ output
    def take_image(self):
        """Return the base64 image produced by the latest ``render_map`` call."""
        image = getattr(self, "_last_image", None)
        self._last_image = None
        return image

    def stats(self) -> dict:
        return dict(self.counts)
