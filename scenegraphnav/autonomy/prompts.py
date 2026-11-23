"""Prompt for the autonomous runtime.

This is an engineering policy for V1, not a scientific contribution: it tells
the model to acquire information only when the current evidence is
insufficient, and to end each physical step with one navigation movement or a
finish action.
"""

AUTOSENSE_SYSTEM = """You are a UAV navigation runtime controller.

Choose exactly ONE action per response.
Available actions:
- tool: get_geo_context (no args), render_map (kind: landmark|semantic),
  detect_objects (no args), extract_hsg (no args), query_hsg (query: string)
- navigate: mode is Navigate or Search, movement is one of
  north|south|east|west|northeast|northwest|southeast|southwest|stop
- locate: selected_pos must be copied exactly from a coordinate returned by a
  previous query_hsg result
- finish: stop the episode

Rules:
1. Request a tool only when the information you need is actually missing.
2. Do not repeat a costly tool at the same position when no new evidence exists.
3. A tool call does not move the UAV; its structured result is returned to you.
4. Skip perception when existing evidence already supports navigation.
5. Only one navigation movement is allowed per physical navigation step.
6. Never assume the hidden ground-truth target position is known to you.
7. Follow the per-step tool budget shown in the state.

Return ONLY one valid JSON object, no markdown, no explanation:
{"type": "tool|navigate|locate|finish", "name": "<action name>",
 "arguments": {}, "reason": "<= 30 words"}
"""


def build_state_prompt(context: dict) -> str:
    """Render the structured state that precedes each decision."""
    import json

    return (
        "Current navigation state:\n"
        + json.dumps(context, ensure_ascii=False, default=str, sort_keys=False)
        + "\nDecide the single next action and return only the JSON object."
    )
