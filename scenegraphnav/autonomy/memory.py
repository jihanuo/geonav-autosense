"""Short-lived task state used by the autonomous runtime.

GeoNav already keeps a long-term :class:`KnowledgeGraph`.  This class only
holds what the controller needs to decide the next action, and it is rebuilt
for every episode, so prompts never grow with the whole history.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class TaskMemory:
    instruction: str
    geo_context: str = ""
    last_result: dict = field(default_factory=dict)
    recent_calls: deque = field(default_factory=lambda: deque(maxlen=6))
    candidate_xy: list = field(default_factory=list)
    seen_tools: dict = field(default_factory=dict)

    def on_tool_result(self, result: dict) -> None:
        self.last_result = result
        tool = result.get("tool")
        ok = bool(result.get("ok"))
        self.recent_calls.append({"tool": tool, "ok": ok, "step": result.get("step")})
        if not ok:
            return
        data = result.get("data") or {}
        if tool == "get_geo_context":
            self.geo_context = str(data.get("geo_context", ""))[:3000]
        elif tool == "query_hsg":
            self.candidate_xy = [
                [float(m["xy"][0]), float(m["xy"][1])]
                for m in data.get("matches", [])
                if isinstance(m.get("xy"), (list, tuple)) and len(m["xy"]) == 2
            ]
        self.seen_tools[tool] = self.seen_tools.get(tool, 0) + 1

    def compact(self) -> dict:
        """Return the state slice that is safe to send to the model."""
        return {
            "instruction": self.instruction,
            "geo_context": self.geo_context[:1500],
            "latest_observation": self.last_result,
            "recent_calls": list(self.recent_calls),
            "candidate_xy": self.candidate_xy[:20],
        }
