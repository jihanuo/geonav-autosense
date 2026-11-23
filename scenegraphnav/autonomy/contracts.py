"""Strict contract for the GeoNav-AutoSense V1 decision protocol.

The VLM/LLM only *proposes* one action per turn.  Every proposal has to pass
:func:`parse_decision` (and, for ``locate``, :func:`approve_locate`) before any
Python function runs.  There is no dynamic dispatch and no ``eval``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

TOOLS = frozenset({
    "get_geo_context",
    "render_map",
    "detect_objects",
    "extract_hsg",
    "query_hsg",
})
ACTIONS = frozenset({"navigate", "locate", "finish"})
DECISION_TYPES = frozenset({"tool"}) | ACTIONS

DIRECTIONS = frozenset({
    "north", "south", "east", "west",
    "northeast", "northwest", "southeast", "southwest",
    "stop",
})
NAV_MODES = frozenset({"Navigate", "Search"})
MAP_KINDS = frozenset({"landmark", "semantic"})

MAX_QUERY_CHARS = 256
MAX_REASON_CHARS = 256


@dataclass(frozen=True)
class Decision:
    """A validated model decision."""

    type: str
    name: str
    arguments: dict
    reason: str = ""

    def as_dict(self) -> dict:
        return {
            "type": self.type,
            "name": self.name,
            "arguments": dict(self.arguments),
            "reason": self.reason,
        }


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def parse_decision(obj) -> Decision:
    """Validate one raw model output object and return a :class:`Decision`.

    Raises ``ValueError`` on anything that is not an exactly supported action.
    """
    _require(isinstance(obj, dict), "model output must be one JSON object")

    kind = obj.get("type")
    name = obj.get("name")
    args = obj.get("arguments", {})
    _require(isinstance(args, dict), "arguments must be an object")
    _require(kind in DECISION_TYPES, f"unknown decision type: {kind!r}")

    if kind == "tool":
        _require(name in TOOLS, f"unknown tool: {name!r}")
        if name == "render_map":
            _require(args.get("kind") in MAP_KINDS, "render_map.kind must be landmark or semantic")
        elif name == "query_hsg":
            query = args.get("query")
            _require(isinstance(query, str) and 1 <= len(query) <= MAX_QUERY_CHARS,
                     "query_hsg.query must be a non-empty string <=256 characters")
        else:
            _require(not args, f"{name} accepts no arguments")

    elif kind == "navigate":
        _require(name == "navigate", "navigate decision must use name=navigate")
        _require(args.get("mode") in NAV_MODES, "navigate.mode must be Navigate or Search")
        _require(args.get("movement") in DIRECTIONS, "navigate.movement must be a cardinal/diagonal direction or stop")

    elif kind == "locate":
        _require(name == "locate", "locate decision must use name=locate")
        xy = args.get("selected_pos")
        _require(isinstance(xy, (list, tuple)) and len(xy) == 2, "selected_pos must be [x, y]")
        _require(all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in xy),
                 "selected_pos must contain two finite numbers")

    else:  # finish
        _require(name == "finish", "finish decision must use name=finish")
        _require(not args, "finish accepts no arguments")

    reason = obj.get("reason", "")
    reason = str(reason)[:MAX_REASON_CHARS]
    return Decision(type=kind, name=name, arguments=dict(args), reason=reason)


def approve_locate(decision: Decision, candidate_xy, current_xy=None,
                   tolerance: float = 2.0, max_dist: float = 80.0):
    """Accept a ``locate`` target only if it comes from a real HSG candidate.

    Returns ``(x, y)`` as floats.  Raises ``ValueError`` otherwise, so a model
    can never teleport the UAV to an arbitrary coordinate.
    """
    _require(decision.type == "locate", "decision is not a locate action")
    x, y = decision.arguments["selected_pos"]
    x, y = float(x), float(y)

    _require(bool(candidate_xy), "no HSG candidate available; run query_hsg first")
    nearest = min(math.hypot(x - float(cx), y - float(cy)) for cx, cy in candidate_xy)
    _require(nearest <= tolerance,
             f"selected_pos is {nearest:.1f}m away from the nearest HSG candidate (tolerance {tolerance}m)")

    if current_xy is not None:
        step = math.hypot(x - float(current_xy[0]), y - float(current_xy[1]))
        _require(step <= max_dist,
                 f"selected_pos is {step:.1f}m away, above the one-step limit of {max_dist}m")
    return x, y
