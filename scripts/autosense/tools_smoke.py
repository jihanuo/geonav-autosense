"""Step-1 tool smoke test on a real CityNav frame (no agent loop).

Runs every whitelisted tool once with fixed input and prints the structured
payload, so we have real (non-mocked) evidence for each adapter.
"""
import json
import os
import sys

ROOT = "/data3/jihanuo/others/geonav"
REPO = os.path.join(ROOT, "geonav-official")
sys.path.insert(0, ROOT)
sys.path.insert(0, REPO)

os.environ.setdefault("DEEPSEEK_API_KEY", open("/tmp/dskey").read().strip())
import run_deepseek_test  # noqa: F401,E402  applies the DeepSeek client patch

from PIL import Image  # noqa: E402
from openai import OpenAI  # noqa: E402

from utils.QwenAPI import encode_image_from_pil  # noqa: E402
from scenegraphnav.parser import parse_args  # noqa: E402
from scenegraphnav.agent import GeonavAgent  # noqa: E402
from scenegraphnav.autonomy.tools import ToolExecutor  # noqa: E402
from gsamllavanav.cityreferobject import get_city_refer_objects  # noqa: E402
from gsamllavanav.dataset.generate import generate_episodes_from_mturk_trajectories  # noqa: E402
from gsamllavanav.dataset.mturk_trajectory import load_mturk_trajectories  # noqa: E402
from gsamllavanav.observation import cropclient  # noqa: E402


def main():
    os.chdir(REPO)
    cropclient.load_image_cache()
    sys.argv = [
        "tools_smoke", "--mode", "eval", "--split", "val_seen",
        "--map_size", "240", "--map_meters", "410",
        "--gsam_use_segmentation_mask", "--output_dir", "results/", "--ablation", "full",
    ]
    args = parse_args()

    episodes = generate_episodes_from_mturk_trajectories(
        get_city_refer_objects(), load_mturk_trajectories("val_seen", "all", 50), max_episodes=None)
    episode = next(e for e in episodes if e.map_name == "birmingham_block_1" and e.id[1] == 11 and e.id[2] == 1)

    client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
    agent = GeonavAgent(args, episode.start_pose, episode, client, client, set_height=None)
    controller = agent.controller
    controller.pose = episode.start_pose
    rgb, _depth = controller.perceive(controller.pose, episode.map_name)
    rgb_b64 = encode_image_from_pil(Image.fromarray(rgb))
    print("frame", rgb.shape, "pose", controller.pose, "timestep", controller.timestep)

    executor = ToolExecutor(agent)
    plan = [
        ("get_geo_context", {}),
        ("render_map", {"kind": "landmark"}),
        ("detect_objects", {}),
        ("extract_hsg", {}),
        ("query_hsg", {"query": episode.target_description}),
    ]
    for name, tool_args in plan:
        result, image = executor.execute(name, tool_args, rgb, rgb_b64)
        payload = dict(result)
        data = payload.get("data")
        if isinstance(data, dict):
            payload["data"] = {k: (v if k != "geo_context" else v[:200]) for k, v in data.items()}
        print("\n===", name, "===")
        print(json.dumps(payload, ensure_ascii=False)[:1200])
        print("image_b64_len", len(image) if image else 0)
    print("\ncounters", json.dumps(executor.stats(), indent=1))
    print("TOOLS_SMOKE_OK")


if __name__ == "__main__":
    main()
