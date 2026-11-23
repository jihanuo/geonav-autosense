"""Bounded autonomous decision loop (GeoNav-AutoSense V1).

Correctness contract for one physical CityNav step:

1. ``controller.perceive()`` is called exactly once, so ``timestep`` grows by 1;
2. inside that step the model may request up to ``max_tools_per_step`` tools,
   each result being fed back before the next decision;
3. the step ends with at most one real movement (``navigate``/``locate``) or a
   ``finish``; if no legal movement is produced the UAV stays put;
4. no global epsilon/GT target position is ever put into the model context.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

from utils.QwenAPI import encode_image_from_pil

from scenegraphnav.autonomy.contracts import approve_locate, parse_decision
from scenegraphnav.autonomy.memory import TaskMemory
from scenegraphnav.autonomy.prompts import AUTOSENSE_SYSTEM, build_state_prompt
from scenegraphnav.autonomy.tools import ToolExecutor

MAX_REPAIR_ATTEMPTS = 2
MAP_MARGIN_METERS = 0.5


def _make_pose(previous_pose, x: float, y: float):
    from gsamllavanav.space import Pose4D

    return Pose4D(float(x), float(y), previous_pose.z, previous_pose.yaw)


def _clamp_to_map(agent, x: float, y: float):
    x_min, y_min, x_max, y_max = agent.landmark_nav_map.bounds
    x = min(max(float(x), x_min + MAP_MARGIN_METERS), x_max - MAP_MARGIN_METERS)
    y = min(max(float(y), y_min + MAP_MARGIN_METERS), y_max - MAP_MARGIN_METERS)
    return x, y


def _model_decide(agent, context: dict, image_b64: str):
    from scenegraphnav.agent import extract_json_from_msg

    raw = agent.call_response(AUTOSENSE_SYSTEM, build_state_prompt(context), [image_b64] if image_b64 else [])
    if not raw:
        raise ValueError("empty model response")
    obj = extract_json_from_msg(raw)
    if obj is None:
        raise ValueError("model response is not valid JSON")
    return parse_decision(obj)


def _track_and_refresh_geometry(agent, rgb) -> str:
    """Run the cheap per-step map bookkeeping, without any heavy perception."""
    controller = agent.controller
    navmap = agent.landmark_nav_map
    navmap.trajectory.append(controller.pose)
    navmap.tracking_map.mark_current_view_area(controller.pose)
    controller.build_geo_nodes(navmap.landmark_map.landmarks)
    agent.view_width = 2 * (controller.pose.z - navmap.ground_level)
    agent.xyxy = [
        (controller.pose.x - agent.view_width / 2, controller.pose.y - agent.view_width / 2),
        (controller.pose.x + agent.view_width / 2, controller.pose.y + agent.view_width / 2),
    ]
    return encode_image_from_pil(rgb)


def run_autonomous_episode(agent):
    """Run one episode with autonomous tool selection.

    Returns ``(success, pose_log)`` where ``pose_log`` is a list of ``Pose4D``
    compatible with ``gsamllavanav.evaluate.eval_planning_metrics``.
    """
    from PIL import Image

    controller = agent.controller
    args = agent.args
    episode = agent.episode
    navmap = agent.landmark_nav_map
    executor = ToolExecutor(agent)
    memory = TaskMemory(instruction=episode.target_description)

    max_tools = max(0, int(getattr(args, "max_tools_per_step", 3)))
    max_llm_calls = int(getattr(args, "max_total_llm_calls", 200))
    max_seconds = float(getattr(args, "max_episode_seconds", 1800.0))
    locate_max_step = float(getattr(args, "locate_max_step_m", 80.0))

    save_path = Path(agent.save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    trace_path = save_path / f"autonomy_{episode.id}.jsonl"

    stats = {
        "n_steps": 0,
        "n_llm_calls": 0,
        "n_decisions": 0,
        "n_invalid_actions": 0,
        "n_invalid_model_responses": 0,
        "no_move_steps": 0,
        "termination": "max_steps",
        "model_tokens": "N/A",
    }

    pose_log = []
    stopped = False
    started_at = time.perf_counter()

    with trace_path.open("w", encoding="utf-8") as log_file:
        while controller.timestep < args.eval_max_timestep and not stopped:
            if stats["n_llm_calls"] >= max_llm_calls:
                stats["termination"] = "llm_budget"
                break
            if time.perf_counter() - started_at > max_seconds:
                stats["termination"] = "time_budget"
                break

            rgb, _depth = controller.perceive(controller.pose, episode.map_name)
            rgb_b64 = _track_and_refresh_geometry(agent, Image.fromarray(rgb))
            try:
                Image.fromarray(rgb).save(f"{args.output_dir}{episode.id}rgb_{controller.timestep}.png")
            except Exception:
                pass

            step_image = rgb_b64
            image_source = "rgb"
            last_result = {}
            moved = False
            tool_calls = 0
            invalid = 0
            decision = None
            tool_log = []
            step_started = time.perf_counter()

            for _attempt in range(max_tools + 3):
                pose = controller.pose
                context = memory.compact()
                context.update({
                    "pose": {"x": pose.x, "y": pose.y, "z": pose.z, "yaw": pose.yaw},
                    "step": controller.timestep,
                    "remaining_nav_steps": args.eval_max_timestep - controller.timestep,
                    "tools_left_this_step": max(0, max_tools - tool_calls),
                    "image_source": image_source,
                    "last_tool_result": last_result,
                    "available_tools": sorted(["get_geo_context", "render_map", "detect_objects",
                                               "extract_hsg", "query_hsg"]),
                })

                stats["n_llm_calls"] += 1
                try:
                    decision = _model_decide(agent, context, step_image)
                except Exception as exc:  # noqa: BLE001 - model output is untrusted
                    invalid += 1
                    stats["n_invalid_model_responses"] += 1
                    stats["n_invalid_actions"] += 1
                    last_result = {"ok": False, "error": "invalid_model_response",
                                   "detail": str(exc)[:150]}
                    if invalid >= MAX_REPAIR_ATTEMPTS:
                        break
                    continue

                stats["n_decisions"] += 1

                if decision.type == "tool":
                    if tool_calls >= max_tools:
                        last_result = {"ok": False, "error": "tool_budget_exhausted",
                                       "detail": "choose navigate/locate/finish now"}
                        invalid += 1
                        stats["n_invalid_actions"] += 1
                        if invalid >= MAX_REPAIR_ATTEMPTS:
                            break
                        continue
                    tool_calls += 1
                    result, optional_image = executor.execute(
                        decision.name, decision.arguments, rgb, rgb_b64)
                    result["step"] = controller.timestep
                    memory.on_tool_result(result)
                    last_result = result
                    tool_log.append(result)
                    if optional_image:
                        step_image = optional_image
                        image_source = f"map:{decision.arguments['kind']}"
                    continue

                if decision.type == "navigate":
                    pose = controller.pose
                    next_xy = agent.get_next_position(
                        decision.arguments["mode"], {"movement": decision.arguments["movement"]})
                    x, y = _clamp_to_map(agent, next_xy[0], next_xy[1])
                    controller.pose = _make_pose(pose, x, y)
                    moved = math.hypot(x - pose.x, y - pose.y) > 1e-9
                    break

                if decision.type == "locate":
                    pose = controller.pose
                    try:
                        xy = approve_locate(decision, memory.candidate_xy,
                                            current_xy=[pose.x, pose.y],
                                            max_dist=locate_max_step)
                    except ValueError as exc:
                        invalid += 1
                        stats["n_invalid_actions"] += 1
                        last_result = {"ok": False, "error": "locate_rejected",
                                       "detail": str(exc)[:150]}
                        if invalid >= MAX_REPAIR_ATTEMPTS:
                            break
                        continue
                    next_xy = agent.get_next_position("Locate", {"selected_pos": list(xy)})
                    x, y = _clamp_to_map(agent, next_xy[0], next_xy[1])
                    controller.pose = _make_pose(pose, x, y)
                    moved = math.hypot(x - pose.x, y - pose.y) > 1e-9
                    break

                # finish
                stopped = True
                stats["termination"] = "finish"
                break

            stats["n_steps"] += 1
            if not moved and not stopped:
                stats["no_move_steps"] += 1

            pose_log.append(controller.pose)
            row = {
                "episode": str(episode.id),
                "step": controller.timestep,
                "pose": [controller.pose.x, controller.pose.y, controller.pose.z, controller.pose.yaw],
                "moved": moved,
                "stopped": stopped,
                "tools": tool_log,
                "decision": None if decision is None else decision.as_dict(),
                "elapsed_ms": round(1000 * (time.perf_counter() - step_started), 1),
            }
            log_file.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            log_file.flush()

            agent.results["steps"].append({
                "time_step": controller.timestep,
                "pose": (controller.pose.x, controller.pose.y, controller.pose.z, controller.pose.yaw),
                "distance_to_target": controller.pose.xy.dist_to(agent.target.xy),
                "autonomous_decision": row["decision"],
                "tool_names": [call.get("tool") for call in tool_log],
                "moved": moved,
                "elapsed_ms": row["elapsed_ms"],
            })

    # Ground truth is used here, and only here, for offline scoring.
    success = bool(controller.reached_target(controller.pose, agent.target))

    stats.update(executor.stats())
    stats["runtime_latency_s"] = round(time.perf_counter() - started_at, 2)
    stats["tool_calls"] = sum(len(json.loads(line)["tools"]) for line in trace_path.read_text().splitlines())
    stats["invalid_action_rate"] = round(
        stats["n_invalid_actions"] / max(1, stats["n_decisions"] + stats["n_invalid_actions"]), 4)

    agent.results["runtime_mode"] = "autonomous"
    agent.results["autonomy_log"] = str(trace_path)
    agent.results["autonomy_stats"] = stats
    agent.results["tool_count"] = stats["tool_calls"]
    agent.save_results(success)
    return success, pose_log
