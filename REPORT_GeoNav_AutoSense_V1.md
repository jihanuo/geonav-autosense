# GeoNav-AutoSense V1 开发报告（工程化改造与闭环验证）

> 版本：V1.0 · 2026-11-23
> 代码仓库：`/data3/jihanuo/others/geonav/geonav-official`，分支 `dev/autosense-v1`
> 基线提交：`925fe1a`（官方复现）→ 冻结提交 `bcd0204`（`chore(repro): freeze GeoNav reproduction fixes before AutoSense`）
> 定位：本阶段只做**工具封装 + 自主调度 + 状态管理 + 对照评测的工程闭环**，不训练新网络、不改评测方法、不引入额外遥感数据。

---

## 1. 摘要

- 原 `workflow` 流程完全保留并仍是默认模式；新增 `autonomous` 模式，由 `--runtime_mode` 切换，`GeonavAgent.run()` 只增加一个分支。
- 8 个白名单动作/工具全部落地：`get_geo_context`、`render_map`、`detect_objects`、`extract_hsg`、`query_hsg`、`navigate`、`locate`、`finish`。模型只输出严格 JSON，Python 校验后执行，无 `eval`、无动态函数名。
- 真实 Episode 中自主调用了**两种不同工具**（`query_hsg`、`get_geo_context`），并留下逐步 JSONL 证据；同一物理步内可完成「决策→工具→结果→再决策→navigate」。
- 单元测试 37 项全部通过（覆盖指导 §7.1 的 T01–T15）。
- 配对对照（同 Episode / 同 8 步预算 / 同地图 / 同模型通道）：`workflow` NE 165.03 m，`autonomous` NE 218.96 m，两者 SR/OSR/SPL 均为 0。**本版不宣称精度提升**；在此预算下自主模式更差，原因见 §6.5。

---

## 2. 环境与复现基线

| 项 | 值 |
|---|---|
| Python 环境 | `/data3/jihanuo/others/geonav/env`（Python 3.10，torch 2.2.2+cu118） |
| 数据/地图 | CityNav SensatUrban，`val_seen`，`--map_size 240`（GSAM 缓存 240×240） |
| Episode（对照用） | `('birmingham_block_1', 11, 1)` |
| 最大导航步数 | 8（对照两侧一致） |
| LLM 通道 | 开发期用 DeepSeek（`deepseek-flash`，关闭 thinking）替代官方 `qwen3-vl-plus`，仅用于工程验证 |
| 感知模型 | 原仓库 GroundingDINO + MobileSAM 权重，未重新训练 |

> 重要：因为 LLM 通道不是官方 `qwen3-vl-plus`，本节及以下数字**不能与论文/官方榜单数字直接比较**，只用于同条件 A/B 对照。

---

## 3. 实现结构

```text
scenegraphnav/
├── agent.py                  # 仅新增 run() 模式分支 + run_workflow()
├── parser.py                 # 仅新增 5 个 CLI 选项
├── llm_controller.py         # 1 处兼容性修复（query_scene_graph 解包）
└── autonomy/
    ├── __init__.py
    ├── contracts.py          # Decision 结构、TOOLS/ACTIONS 白名单、parse_decision、approve_locate
    ├── prompts.py            # 单动作 JSON 协议系统提示 + 状态提示
    ├── memory.py             # TaskMemory：任务/地理上下文/最近调用/候选坐标/已见工具
    ├── tools.py              # ToolExecutor：白名单 Dispatcher + 真实计数 + 结构化异常
    └── runtime.py            # run_autonomous_episode：单步 perceive + 内层工具循环 + 预算
tests/
├── test_autonomy_contracts.py
├── test_autonomy_tools.py
└── test_autonomy_runtime.py
scripts/autosense/
├── run_autosense_smoke.sh
├── run_baseline8.sh
└── tools_smoke.py
results/autosense_v1/
├── baseline_metrics.json
├── autonomous_metrics.json
├── comparison.csv
├── episode_birmingham_block_1_11_1.jsonl
└── failed_cases.md
```

### 3.1 关键设计决策

1. **`perceive()` 每物理步只调用一次**：`LLMController.perceive()` 内部 `self.timestep += 1`，因此工具层绝不调用 `perceive()`；工具只使用本步已缓存的 `rgb`。
2. **拆分轻量轨迹更新与重型感知**：自主模式不再调用 `LandmarkNavMap.update_observations()`（它内含 GDINO/SAM 更新），只做 `trajectory` 追加与 `tracking_map.mark_current_view_area()`；GDINO 只在 `detect_objects` 被请求时执行。
3. **HSG 独立性**：`extract_hsg` / `query_hsg` 在执行前显式填充 `layer.pose`、`layer.image_bgr = rgb[..., ::-1]` 并调用 `build_geo_nodes()`，因此即使从未跑过 GDINO 也能建图/查询。
4. **动作受控**：`navigate` 只接受 `Navigate|Search` + 8 方向/stop；`locate` 的坐标必须来自 `query_hsg` 候选（容差 2 m），且单步位移 ≤ 80 m，最终位置统一 clamp 到地图 bounds（留 0.5 m 边距）。
5. **GT 隔离**：prompt、工具参数、工具结果均不含 `episode.target_position`；只在 Episode 结束后用原 `controller.reached_target` 离线评分。
6. **预算**：每步 `max_tools_per_step`（默认 3）、每 Episode `max_total_llm_calls`（200）、`max_episode_seconds`（1800 s）；同一步内 `(timestep, tool, kind, query)` 去重，重复请求返回 `duplicate_same_step`。

---

## 4. 工具与真实证据

### 4.1 真实工具冒烟（`logs/tools_smoke.log`）

| 工具 | ok | 耗时 | 实际副作用/返回 |
|---|---|---|---|
| `get_geo_context` | true | 1.1 ms | 返回相对地标描述，无 GT |
| `render_map` | true | 134.7 ms | base64 图 21672 字符（`kind=landmark`） |
| `detect_objects` | true | 559.9 ms | `n_gdino_calls=2`（目标层+环境层），该帧 0 检测，`empty_image=false`，`n_sam_calls=0` |
| `extract_hsg` | true | 1059.6 ms | 2 节点 1 边，节点 `row_of_grayish_brown_houses`、`gray_house` 带真实 bbox |
| `query_hsg` | true | 1222.8 ms | 2 个候选（`TrafficRoad_Leslie Road`、`TrafficRoad_Wellington Road`）+ 6 步 `operation_chain` |

计数器：`n_gdino_calls=2, n_sam_calls=0, n_hsg_calls=1, n_map_queries=1, n_geo_context_calls=1, n_render_map_calls=1, n_detect_calls=1, n_duplicate_rejections=0`。

### 4.2 自主 Episode 证据（`episode_birmingham_block_1_11_1.jsonl`，8 行）

| step | 工具 | 决策 | 位移 |
|---|---|---|---|
| 1 | `query_hsg`（空候选，返回 operation_chain） | `navigate(Search, east)` | 是 |
| 2 | — | `navigate(Search, north)` | 是 |
| 3 | `get_geo_context`（“Leslie Road 361.4m East / Wellington Road 179.6m East”） | `navigate(Navigate, east)` | 是 |
| 4–8 | — | `navigate(Navigate, east)` | 是 |

统计：`n_steps=8`、`tool_calls=2`（两种不同类型）、`n_llm_calls=12`、`n_decisions=10`、`n_invalid_model_responses=2`、`n_invalid_actions=2`、`no_move_steps=0`、`runtime_latency_s=22.25`、`termination=max_steps`、`n_gdino_calls=0`、`n_sam_calls=0`（未请求即不跑重型感知）。

第 1 步即满足「决策 → 工具 → 结果 → 再决策 → `navigate`」的同一步闭环；第 3 步是第二次工具调用，且是**另一类工具**。

---

## 5. 测试

```bash
cd /data3/jihanuo/others/geonav/geonav-official
/data3/jihanuo/others/geonav/env/bin/python -m unittest discover -s tests -p 'test_autonomy_*.py'
# Ran 37 tests ... OK
```

覆盖的关键验收点：

| ID | 用例所在 | 结论 |
|---|---|---|
| T01 | contracts | 未知工具被拒，不执行任何函数 |
| T02 | contracts | `render_map.kind` 白名单 |
| T03/T04 | contracts | 合法/注入方向 |
| T05/T06 | contracts | NaN、非最近候选、超单步距离均拒绝 |
| T07 | tools | 同一步重复 `detect_objects` 不重跑重型模型 |
| T08 | tools | 未跑 GDINO 也能绑定 `pose`/`image_bgr` 并建图 |
| T09 | tools | 空检测返回 `[]`，可继续导航 |
| T10 | runtime | 非法 JSON 有限重试，安全结束 |
| T11 | tools/runtime | 未请求即不调用 GDINO/SAM/VLM 图提取 |
| T12 | tools | `render_map` 图像进入下一轮决策 |
| T13 | runtime | 一次 `perceive` + 三次工具 → `timestep` 只 +1 |
| T14 | runtime | `workflow` 分支不进入自主 Runtime |
| T15 | runtime | prompt 中无 `target_position`/GT 坐标 |

---

## 6. 对照评测

命令（两侧唯一差别是 `--runtime_mode`）：

```bash
# workflow 基线
python main_geonav.py --split val_seen --select_specific_episode birmingham_block_1 11 1 \
  --map_size 240 --eval_max_timestep 8 --runtime_mode workflow
# 自主模式
python main_geonav.py --split val_seen --select_specific_episode birmingham_block_1 11 1 \
  --map_size 240 --eval_max_timestep 8 --runtime_mode autonomous \
  --max_tools_per_step 2
```

### 6.1 指标（`results/autosense_v1/comparison.csv`）

| mode | episode | max_nav_steps | NE (m) | SR (%) | OSR (%) | SPL (%) |
|---|---|---|---|---|---|---|
| workflow | birmingham_block_1/11/1 | 8 | 165.03 | 0.00 | 0.00 | 0.00 |
| autonomous | birmingham_block_1/11/1 | 8 | 218.96 | 0.00 | 0.00 | 0.00 |

### 6.2 感知/LLM 成本

| 指标 | workflow（此 8 步预算） | autonomous |
|---|---|---|
| 视觉工具调用 | 每步固定 GDINO（`update_observations` 内） | 0 次 GDINO（未被请求） |
| `n_gdino_calls` | 每步 2（未单独计数，原流程硬编码） | 0 |
| `n_sam_calls` | 有检测时触发 | 0 |
| `n_hsg_calls` / `n_map_queries` / `n_geo_context_calls` | 0 / 0 / 0 | 0 / 1 / 1 |
| `n_llm_calls` | 每步 1（计划/动作） | 12（含 2 次工具后重决策、2 次非法重试） |
| `model_tokens` | 通道未返回 usage → N/A | N/A |
| `runtime_latency_s` | — | 22.25 |
| `invalid_action_rate` | — | 0.1667 |
| `no_move_steps` | 0 | 0 |
| 冗余重型调用 | 原流程本身每步固定执行 | 0（同一步去重生效） |

### 6.3 结论与边界

- 在**单 Episode、8 步**的极小预算下，两种模式都未成功，SR=0，因此无法据此评估成功率。
- 自主模式 NE 比基线差 54 m（218.96 vs 165.03）：模型在拿到「Leslie/Wellington Road 在东侧」后持续向东推进，但 8 步不足以到达，且没有触发任何目标级感知（本 Episode 也未出现目标观测）。
- 由于 LLM 通道替换为 DeepSeek，本表只能说明「工程闭环成立 + 可在同条件复现对照」，**不能作为精度结论**。
- 建议下一步至少 30–50 个 Episode、每 Episode 官方默认步数，再评估 SR/SPL。

---

## 7. 问题与修复（摘要）

完整记录见 `results/autosense_v1/failed_cases.md`。

1. **`query_hsg` 解包 bug**：官方 `QueryEngine.robust_subgraph_query()` 返回单列表，`LLMController` 按二元组解包 → `ValueError`。已在 `llm_controller.py` 做兼容解包。
2. **`render_map` 返回类型**：本地分支 `plot()` 返回 base64 字符串而非 PIL Image → 已兼容两种类型。
3. **非法 JSON**：`n_invalid_model_responses=2`，重试后恢复；未造成崩溃或错误位移。
4. **空检测**：返回 `[]` 且不视为完成，允许继续导航。
5. **离线性**：GDINO 需 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`，且必须先 `cropclient.load_image_cache()`。

---

## 8. 回答指导 §9 的六个问题

1. **原 GeoNav 每步实际调用什么？自主模式调用什么？**
   原流程 `update_observations()` 每步对目标层与环境层各跑一次 GroundingDINO（有检测时再跑 MobileSAM），`Search` 下再按时间/距离条件触发 VLM `understand()`；自主模式把这些都变成按需工具：本 Episode 只调了 `query_hsg` 1 次、`get_geo_context` 1 次，GDINO/SAM/VLM 图提取均为 0 次。

2. **每次工具调用是否经过合法性检查？有哪些错误与恢复？**
   是。所有调用先经 `parse_decision()`（工具名/参数类型/白名单/查询长度/坐标有限性）再交给 `ToolExecutor._dispatch`，无 `eval`。真实错误有：`query_hsg` 解包 `ValueError`（修复后恢复）、`render_map` 的 `AttributeError`（修复后恢复）、2 次非法 JSON（重试后恢复）。`locate` 还会经 `approve_locate()` 二次校验候选与单步距离。

3. **为什么选这些工具？有无重复调用/空转？**
   第 1 步模型在没有任何对象候选时选择 `query_hsg`，用指令里的道路名做关系查询（返回空候选 + 操作链）；第 3 步需要相对位置，选 `get_geo_context`（拿到两条道路的方位与距离）。同一步去重生效，`n_duplicate_rejections=0`；未出现同一步重复重型调用。空转情况是存在的：query 返回空候选后没有换策略，直接连续 6 步向东，这属于**信息利用率不足**而非重复调用。

4. **同 Episode 下指标/成本如何变化？**
   见 §6.1–6.2：NE 165.03 → 218.96（变差 54 m）；SR/OSR/SPL 均为 0；导航步数同为 8；自主模式重型感知 0 次，LLM 调用 12 次，端到端 22.25 s；Token 因通道不返回 usage 为 N/A。

5. **若自主模式更差，问题出在哪？**
   本 Episode 不是「识别错目标」，而是三点叠加：(a) 8 步预算太小，向东直行不可能到达 180–360 m 外的地标；(b) HSG 为空候选后缺少「换一种感知（`render_map`/`detect_objects`）」的探索策略，模型只重复 `navigate`；(c) LLM 输出稳定性不足（2/12 非法），进一步压缩有效决策。未观察到决策超时或动作越界。

6. **哪些现象值得提炼为后续算法创新？**
   - **信息价值驱动的工具选择**：模型在「无候选」时不会主动升级到更重的感知，说明需要显式建模「当前不确定度 → 期望信息增益 → 工具成本」。
   - **空结果后的策略切换**：可把「连续空查询」作为状态特征，训练/规则化地触发 `render_map` 或 `detect_objects`。
   - **预算感知规划**：8 步内应优先选择能缩短 NE 的动作，而不是语义上「更接近地标」的方向；可引入剩余步数归一化的进度估计。
   - **输出鲁棒性**：非法 JSON 率是实际可用性的瓶颈之一，值得做 schema-constrained decoding 或强校验重采样。

---

## 9. 局限

- 单 Episode、8 步，样本量不足以支撑任何成功率结论。
- LLM 通道为 DeepSeek（工程替代），与官方 `qwen3-vl-plus` 不一致，数字不可与论文对比。
- `model_tokens` 无法统计（通道不返回 usage），`runtime_latency_s` 含网络波动。
- 图投影沿用官方简化几何，未做精度评估。
- `n_sam_calls` 在该 Episode 为 0，SAM 路径仅在检测非空时才会计数。
