# GeoNav-AutoSense V1 失败案例与恢复记录

本文件只记录**真实运行中出现过**的失败（单元测试构造的用例不在此列），每条包含现象、根因、修复与验证方式。

---

## FC-01 模型输出非法 JSON（真实 Episode 内，已恢复）

- 现象：`autonomous` 模式下 `birmingham_block_1/11/1`，模型在部分决策轮返回的不是合法 JSON（包含 markdown 包裹或自然语言前缀）。
- 影响：该轮不产生工具调用或位移，Runtime 按 `MAX_REPAIR_ATTEMPTS=2` 重新请求；连续两次仍非法则本物理步不移动并进入下一步。
- 证据：该运行的 `n_llm_calls=12`、`n_decisions=10`、`n_invalid_model_responses=2`、`n_invalid_actions=2`、`invalid_action_rate=0.1667`（8 物理步 = 8 次基础决策 + 2 次工具回传后再决策 + 2 次非法响应重试）；`episode_birmingham_block_1_11_1.jsonl` 中 8 步全部 `moved=true`，说明重试后仍完成了导航动作。trace 只记录被采纳的决策，非法响应仅体现在计数中。
- 说明：更早一次运行（结果文件 `..._20261123_233204.json`）同样条件下为 `n_invalid_model_responses=5`、`invalid_action_rate=0.3333`，说明该模型输出稳定性本身是波动源；收紧状态提示后降到 2 次，但未彻底消除。
- 结论：非法 JSON 不会导致崩溃或错误位移，但会消耗 LLM 预算并降低单步有效决策率。

## FC-02 `query_hsg` 解包异常：`ValueError: not enough values to unpack (expected 2, got 1)`

- 现象：工具首次真实调用即抛异常，被 `ToolExecutor` 捕获为 `{"ok": false, "error": "ValueError"}`。
- 根因：官方 `LLMController.query_scene_graph()` 里写的是 `results, query_logs = QueryEngine.robust_subgraph_query(...)`，而该函数实际只返回单个结果列表。
- 修复：在 [llm_controller.py](../scenegraphnav/llm_controller.py) 增加兼容解包——返回二元组时解包，否则视为 `(results, None)`。只改这一处，不改查询语义。
- 验证：修复后 `query_hsg` 返回 `candidate_count=2`（`TrafficRoad_Leslie Road`、`TrafficRoad_Wellington Road`）与 6 步 `operation_chain`，耗时 1222.8 ms。

## FC-03 `render_map` 返回类型假设错误：`AttributeError: 'str' object has no attribute 'save'`

- 现象：工具内尝试对 `agent.gen_map(kind)` 的返回值调用 `PIL.Image.save` 时报错。
- 根因：`LandmarkNavMap.plot()` → `_render_output()` 在本地复现分支返回的是 **base64 字符串**，不是 `PIL.Image`。
- 修复：在 [tools.py](../scenegraphnav/autonomy/tools.py) 中改为 `image if isinstance(image, str) else _pil_to_base64(image)`，两种返回类型都能作为图像输入传给下一次决策。
- 验证：`tools_smoke` 中 `render_map` 返回 base64 长度 21672，耗时 134.7 ms。

## FC-04 首帧 `detect_objects` 空检测（非错误，但需防止误判为完成）

- 现象：工具冒烟与 Episode 首帧中 GroundingDINO 对目标/环境图层各推理一次（`n_gdino_calls=2`），该帧检测结果为空列表。
- 处理：空检测返回 `objects: []` 且 `empty_image=false`，Runtime 不将空结果解读为「目标已找到」，模型继续 `navigate` 探索；整幅近黑图像（`(image_bgr>0).mean()<0.15`）会直接返回 `empty_image=true` 并跳过推理。
- 影响：空检测只花 GDINO 时间，不产生错误动作；在 8 步预算内该 Episode 未命中目标，属于「信息不足 + 预算受限」，不是崩溃。

## FC-05 工具异常不得中断 Episode（结构化兜底）

- 现象：任何工具内部异常（模型文件缺失、网络不可达、空缓存等）在早期调试中会直接终止进程。
- 处理：`ToolExecutor.execute()` 统一捕获异常并返回 `{"ok": false, "error": <异常类名>, "detail": ...}`，Runtime 把该结果计入 `recent_calls` 后继续让模型决策，并由每步工具预算与 Episode LLM 预算兜底。
- 验证：单测 `test_tool_exception_is_returned_as_structured_error`，以及 FC-02/FC-03 的真实恢复路径。

---

## 与「失败」相关的环境问题（不属于算法行为，但影响可复现性）

- 离线下 GroundingDINO 会反复请求 huggingface.co，需要 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`；同时必须先 `cropclient.load_image_cache()`，否则报 `TypeError: 'NoneType' object is not subscriptable`。
- 本地 GSAM 缓存分辨率为 240×240，必须以 `--map_size 240` 运行，否则地图投影尺寸不匹配。
