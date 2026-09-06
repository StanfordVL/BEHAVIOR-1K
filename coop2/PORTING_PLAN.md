# COOP² (ma_crafter) → BEHAVIOR-1K 移植计划 v2

> 目标：以最小改动把 `coop2-llm-mas/ma_crafter` 的多智能体协作架构移到 BEHAVIOR-1K，支持 individual / broadcast_chain / centralized，后续加 decentralized。**不移植 repair**。
>
> v2（2026-09-03）依据四项已定决策重写。依据：两库源码全量阅读 + 已删除的 `symbolic_llm/*.pyc` 反编译。

---

## 0. 已定决策与直接后果

| # | 决策 | 直接后果 |
|---|---|---|
| D1 | `symbolic_llm/` 源码**找不回来** | L1 全部从零写。但反编译出的分层（见 §8）仍是最好的蓝图，照它的模块划分写 |
| D2 | **从一开始就做并发交织** | 必须自建「多生成器交织 + idle 填充」driver；并要处理 cuRobo **过期碰撞世界**问题（§3.3，最大技术风险） |
| D3 | 用 **`StarterSemanticActionPrimitives`**（真实物理）模拟竞争 | ⚠️ **可用原语只剩 5 个**；单个原语要跑数千 env step；启动要 N×3 个 cuRobo MotionGen warmup；协作压力**天然存在**（导航/可达性都是真的） |
| D4 | **先跳过 BDDL**，跑通其余管线 | 用 `DummyTask` + config 里手写 `objects` + 自定义成功判据；COOP² 的 task 概念改成自己在 `coop_config.yaml` 里定义的**协作搬运任务**（与 crafter 的 cooperative_collection 一一对应） |

### D3 最重要的三条硬约束（必须先知道）

**(1) 可用原语只有 5 个。** `StarterSemanticActionPrimitiveSet` 共 9 项，但：

```
GRASP          ✅  _grasp(obj)
PLACE_ON_TOP   ✅  _place_on_top(obj)
PLACE_INSIDE   ✅  _place_inside(obj)
NAVIGATE_TO    ✅  _navigate_to_obj(obj)
RELEASE        ✅  _execute_release()            ← 无参数，且释放所有手臂
OPEN / CLOSE   ❌  raise NotImplementedError     (starter_...py:329)
TOGGLE_ON/OFF  ❌  raise NotImplementedError     (starter_...py:638)
```

→ **LLM 的动作词表（`llm_client.py` 的 Pydantic models）只能出这 5 个**，别照 symbolic 版的 14 个写。开门/开关这类任务在 v1 做不了；若必需，要么自己实现（参考 symbolic 版的 `_open_or_close`/`_toggle`，它们是可用的），要么混用两套原语集。

**(2) 一个原语 = 数千 env step。** `_move_hand` 开头就 `_settle_robot()`（50–550 步），`_execute_motion_plan` 把轨迹按 `max_inter_dist=0.01` 重新插值后逐 waypoint 跟踪（每 waypoint ≤10 步），一次成功的 `GRASP` 量级在 **10³–10⁴ env step**；失败会重试 **5 次**（`apply_ref(attempts=5)`，且每次重试都跑一遍 `_reset_robot` + `_settle_robot`）。

→ **macro-step 与 sim-step 必须彻底分开计。** COOP² 的 `D_i / L_tot / P_*` 全部按「决策」算，指标分母用 macro-step；`Timeout(max_steps)` 数的是 sim step，要设得很大（或改用 wall-clock deadline，runner 里本来就有）。
→ 建议把 `apply_ref(attempts=1)` 或 2，让失败尽快回到 LLM，而不是在仿真里烧 5 分钟。

**(3) 启动成本线性叠加，无法共享。** 每个 robot 一个 `CuRoboMotionGenerator`，内含 `robot.curobo_path` 的每个 embodiment（R1 约 3 个）一个 `MotionGen`，每个都要 `warmup()` + CUDA graph capture；每个 generator 还有独立的 2048-mesh 碰撞缓存。障碍物位姿是在**各自 root_link 坐标系**下表达的，所以**结构上无法共享**。

→ N=2 起步，先测单 robot 的启动时间与显存再乘 N。别一上来跑 4 个。
→ `curobo_batch_size` 在 CUDA graph 捕获后**不可改**。

---

## 1. BEHAVIOR-1K 侧能力盘点（针对 D2+D3 修订）

| 需求 | 现状 | 结论 |
|---|---|---|
| 多机器人共存 | `config["robots"]` 是 list，逐个按 pose 落位 | ✅ 纯配置。**必须 `scene.include_robots: false`**，否则你的列表被静默忽略 |
| 动作空间 | `Dict{robot.name: Box}`；`_pre_step` 要求**每个 robot 每 tick 都有 key**，缺一个 KeyError | ⚠️ idle action 必需，用 `robot.q_to_action(robot.get_joint_positions())` |
| 观测空间 | `{robot_name: {...}, "task": ..., "external": {...}}` | ✅ 天然 per-agent |
| 原语实例 | `StarterSemanticActionPrimitives(env, robot, ...)`，无跨实例单例 | ✅ 每 robot 一份安全，但构造要在 robot 已到 reset pose 之后（`_arm_targets`/`_reset_eef_pose` 在 `__init__` 冻结） |
| 原语调用 | `apply_ref(primitive, *args, attempts=5)` 是生成器，yield 单 robot action tensor；**异常在迭代时抛出**（`ActionPrimitiveErrorGroup`） | ⚠️ try/except 要包 `next()`，不是包调用 |
| 失败反馈 | `ActionPrimitiveError(reason, message, metadata)`，5 种 Reason，消息本身带补救建议 | ✅ `str(e)` 直接喂回 LLM |
| 导航 | cuRobo 规划 base（`emb_sel=BASE`），**不用 traversability map**；`_sample_pose_near_object` 在目标周围 U(0,1.5)m 拒绝采样，**用 `scene._seg_map` 按房间过滤** | ⚠️ 必须用 `InteractiveTraversableScene`（如 `Rs_int`），空场景会挂 |
| 可达性 | `_navigate_if_needed` → `_target_in_reach_of_robot`（cuRobo IK + 碰撞检查） | ✅ **真实空间约束，协作压力天然存在**（这正是 D3 的目的） |
| 抓取仲裁 | **完全没有**。两个 robot 抓同一物体，输的那个在执行完之后才报 `POST_CONDITION_ERROR: An unexpected object was detected in hand` | ⚠️ 仲裁要在 L1c/L2 自己做（这恰好是 centralized/chain 协调器该管的事） |
| 部分进度 | 跳过 BDDL 后由自定义任务提供 | 见 §4 |
| eval harness | `evaluator.py` 硬编码单 robot | ❌ 不改它，用 COOP² 自带 runner |

### 仍需自己造

1. 多机器人原语并发调度器（+ idle action + **周期性重规划**，§3.3）
2. per-agent 部分可观测（`object_registry` 全局无可见性过滤）
3. agent 间通信（原语显式禁止 Robot 作参数 → 无 in-sim 交接，share/handover 只能符号层模拟）
4. 轮次调度（`env.step` 是单一全局 tick）
5. 目标物体锁 / 任务仲裁
6. 自定义协作任务与成功判据（D4）

---

## 2. 抽象层设计（七层）

目录与 `ma_crafter/` 同构，方便 diff 与回灌：

```
coop2/
├── behavior_env/          ← 全新（L1）
│   ├── coop_env.py            CooperativeBehaviorEnv         (L1)
│   ├── world_state.py         BehaviorWorldState              (L1a)
│   ├── symbolic_view.py       render_symbolic_view/target_hints (L1b)
│   ├── primitive_engine.py    并发原语调度器 + idle + 重规划   (L1c) ★核心
│   ├── cooperative_tasks.py   CoopTaskTracker（自定义任务）    (L1d)
│   └── coop_config.yaml       协作约束/计分/任务定义
├── cognitive/             ← 复制 ma_crafter/cognitive，改 3 处词表
├── comm_topology/         ← 原样复制（+ 后加 decentralized）
├── experiment/            ← 复制 runner，改 env 构造 + 抽公共 _run_episode
└── _repair_shim/          ← ~30 行桩，让 plan_env_wrapper 不用改
```

```
L6  experiment/     runner · run_grid 扇出 · llm_usage · build_results_table
                    ↑ 环境无关（改：env 构造、task_tracker 取用路径、viz）
────────────────────────────────────────────────────────────────
L5  comm_topology/  「谁跟谁说话、什么顺序」individual/chain/centralized
                    (+centralized_flow) ↑ 100% 环境无关，只改 role prose
────────────────────────────────────────────────────────────────
L4  cognitive/agent/ Agent FSM(R/W/X/I) · AgentMemory · MessageBroker
                     · prompts · LLMClient(结构化输出)
                    ↑ 环境无关，除：① 5 原语 Pydantic 词表 ② ENV_DESCRIPTION
                      ③ format_agent_status（health/food → 持物/房间/目标进度）
────────────────────────────────────────────────────────────────
L3  cognitive/plan/  SymbolicPlan/Executor/Logger
                     PlanningEnvWrapper ← ready 屏障 · W→X 同步时间戳
                                          · plan 成败判定 · 8 类日志
                    ↑ 环境无关，除 _plan_goal_failure_reason（换成任务谓词差分）
────────────────────────────────────────────────────────────────
L2  cognitive/action/ 符号动作 → 环境原语（动作接地层）
                      SymbolicActionExecutor（8 controller → 5 原语 + wait/share）
                      SymbolicEnvWrapper（名字映射 · intent 侧信道 · 终止判定）
                    ★ 保留 ActionRecord + check_termination_condition 的
                      pending/success/failed/terminate_plan 契约不动
                    ★ navigation.py（grid A*）整个丢掉，导航交给 cuRobo
────────────────────────────────────────────────────────────────
L1  behavior_env/   CooperativeBehaviorEnv ← 本次移植主体
    ├ L1a WorldState      实体注册表 + type-local ID + 位置/房间/状态/持物
    ├ L1b symbolic_view   场景→文本 + target_hints（LLM 唯一的合法 ID 来源）
    ├ L1c PrimitiveEngine ★ per-robot 原语集 · 生成器交织 · idle 填充
    │                       · 目标物体锁 · 周期性 update_obstacles/重规划
    ├ L1d CoopTaskTracker 自定义协作任务 → TaskState(spatial/temporal/
    │                       dependency/participation) + StepMetrics 差分
    └ L1e action_outcome  {status, reason_code, reason, effects{...}}
────────────────────────────────────────────────────────────────
L0  OmniGibson      og.Environment(N×R1) · DummyTask + objects ·
                    StarterSemanticActionPrimitives · InteractiveTraversableScene
                    ↑ 只写配置 + 调 macros，不改 OmniGibson 源码
                      （唯一例外：cuRobo mesh_cache_size / 基座关节限位，见 §6）
```

### 工作量估计（D1 生效后上调）

| 层 | 改动性质 | 估计 |
|---|---|---|
| L0 | 配置 + macros 调参 + 启动仪式 | 1 天 |
| **L1c** | **并发调度器（最难，含重规划与仲裁）** | **4–6 天** |
| L1a/L1b | 世界模型 + 文本观测（从零，无 symbolic_llm 可抄） | 4–5 天 |
| L1d/L1e | 自定义任务 + 度量底座 | 2–3 天 |
| L2 | 重写 controller 词表，保留架构 | 2–3 天 |
| L3 | 复制 + 改 1 方法 + 去 repair 依赖 | 1 天 |
| L4 | 复制 + 改 3 处 | 1–2 天 |
| L5 | 原样复制 | 0.5 天 |
| L6 | 复制 + 抽公共 loop | 1 天 |
| 新增 | `coop2_attempt_events.py` + repair shim + viz stub | 1 天 |

---

## 3. L1 子层设计要点

### 3.1 L1a/L1b：世界模型与文本观测（无现成实现，从零写）

按反编译出的 `observation.py` 结构写（那是你自己的设计，最省心）：

```
EntityObservation:  name · category · rooms · abilities · observable_states
                    · position · held_by
PredicateFact:      谓词 + 参数 + 当前真值
SymbolicObservation: step / max_steps / entities / facts / goal_status
                     / last_action_id / last_error
```

- **不要 dense matrix**（crafter 是 64×64 网格扫描），用稀疏实体列表 + 房间分组。
- **type-local ID 机制必须保留**：LLM 处理 `cup#1 / cup#2` 远好于 `cup_xkqmzp_0`。`world_state.py` 里维护 `type_local_positions` / `type_local_stable_ids` 的等价物。
- `target_hints` = 当前可交互目标 + 每个目标可用的原语（即反编译里的 `SymbolicActionCatalog.enumerate(observation, cooldowns)`）。**这是 LLM 学到合法 `item_id` 的唯一途径**，优先做对。
- per-agent 可见性：用 `robot.get_position_orientation()` + `scene._seg_map.get_room_instance_by_point()` 得到所在房间，只暴露同房间 + 曾访问过的房间的实体（维护 per-agent `seen` 集合）。

### 3.2 L1c：并发调度器（D2）—— **已实现**

`coop2/behavior_env/primitive_engine.py`。核心决定：**engine 是 stepper 而不是 driver**，主循环归调用方（L3），这样才能和 COOP² 的分层对齐。

⚠️ **屏障必须在 plan 边界，不能在原语边界。** COOP² 的 `PlanningEnvWrapper.step` 里：

```python
all_ready = all(self.agents[aid].ready for aid in self.agent_names)
if managed_agents and not all_ready:
    return self._idle_step_return({"waiting_for_agents": waiting_for})   # 环境完全不推进
for agent_id in self.agent_names: ...                 # W→X，共享时间戳
for agent_id in self.agent_names:                     # 各取自己 plan 的当前动作
    actions[agent_id] = self.plan_executors[agent_id].step(...)
```

`ready` 的含义是「我有可执行的 plan」；只有 plan 完成/失败才 `set_unready('plan_terminated')`。**plan 执行期间各 agent 完全不同步。** 早期版本的 `macro_step()`（所有 agent 各一个原语、全跑完才返回）在每个原语边界强制对齐，与论文语义冲突，已降级为 demo/测试用的便利封装。

API：

```python
engine.assign(agent_id, primitive, target) -> Optional[PrimitiveOutcome]
    # 解析目标 → 创建 apply_ref 生成器 → 记入 _active。不推进仿真。
    # None = 已接受；非 None = 一步未走就失败（INVALID_TARGET）
engine.tick() -> Dict[agent_id, PrimitiveOutcome]
    # 全 idle 的 action dict → 每个 active agent 调一次 next() 覆盖它的 action
    # → env.step() 推进一格。返回值只含【本 tick 终止的】，正常是 {}
engine.has_active(agent_id) -> bool          # 上一个原语还在飞吗
engine.abort(agent_id, retract=False)        # X→I 消息中断；retract 会把
                                             # controller._reset_robot() 挂成 cleanup run
```

L3 的循环因此与 COOP² 同构：

```python
while not done:
    if not all_ready:                                  # plan 层屏障
        yield _idle_step_return(...); continue         # 不调 tick()，物理冻结
    for agent in W态: agent.start_execution(shared_ts)
    for agent in X态:
        if not engine.has_active(agent):               # 上一个原语刚结束
            engine.assign(agent, *to_primitive(plan_executors[agent].step(...)))
    for agent, outcome in engine.tick().items():
        ... plan.advance_action() / complete_failed() / set_unready('plan_terminated')
```

要点：
- **idle action 用 `robot.q_to_action(robot.get_joint_positions())`**（`_settle_robot` 内部就是这么做的），**不要用 `_empty_action()`**——它会主动把手臂伺服到 `__init__` 时冻结的 `_arm_targets`，等待中的机器人会莫名收臂。
- **屏障阻塞时冻结物理是安全的**：不调 `env.step`，PhysX 不推进；Python 生成器天然挂起/恢复，cuRobo 的轨迹是关节位置目标序列，暂停不影响。
- `assign` 是**惰性**的：engine 永不预取 plan 的下一个原语，plan 推进权完全在 L3，这样 `interrupts_execution` 打断时不会有脏状态。
- 原语终止的那一 tick 仍会消耗一个 `env.step`（终止的 agent 该 tick 走 idle），所以每次原语切换有 1 个 idle tick——相对 1e3–1e4 可忽略。
- `MotionMode.EXCLUSIVE` 的语义随之变成「只有最早 assign 的 active agent 前进，其余 hold」，仍是 stale-obstacle 的可靠对照。
- **两个计数器分开**：`env_step`（tick 数，`Timeout(max_steps)` 数的是这个）与 `decision_count`（发出的原语数，**COOP² 指标的分母**）。
- 全局 RNG 共享（`random.choice` 在 `_sample_grasp_pose`、`th.rand` 在 `_sample_pose_near_object`）→ **交织顺序会改变采样结果**，复现性只能做到「固定 seed + 固定交织顺序」。

**目标仲裁：故意不做在这一层。** 两个 agent 可以被派到同一个物体，engine 不阻止；输的那个跑完整段运动后吃 `POST_CONDITION_ERROR`（"An unexpected object was detected in hand"）。理由：竞争是**协作问题**，该由拓扑层解决（centralized 的 leader 分配、chain 的提案序列），而不是底层加一把机械锁——否则 LLM 永远学不到需要协商。engine 的职责是把失败如实上报。

已知漏洞（留待上层解决）：`_get_obj_in_hand()` 读的是 `robot._ag_obj_in_hand[arm]`，**每个 controller 只看得见自己的手**，所以 agent_1 不知道苹果在 agent_0 手里，sticky 抓取还可能把它抢过来。`engine.held_objects()` 提供了跨 agent 的持有视图，L1b 的 `symbolic_view` 应当把它写进观测，让 LLM 自己看见。

**测试**：`feasibility_verify/test_primitive_engine_stubbed.py` 把 OmniGibson 全部打桩，**纯 CPU、无 Isaac 可跑**，覆盖不对齐、失败杀 plan、idle 填充、abort/retract、记录格式共 8 项。

### 3.3 ⚠️ 最大风险：cuRobo 的碰撞世界是过期快照

`CuRoboMotionGenerator.update_obstacles()` 只在**规划时刻**扫一遍 `robot.scene.objects` 建世界；随后 `_execute_motion_plan` 跑上千 tick，中途**完全不重新检查**（`stop_on_contact` 默认关）。并发时，A 的规划从第一 tick 起就已经过期——B 已经动了。

三级缓解，建议**都做，用配置开关**：

1. **确认其他 robot 在碰撞世界里**（启动时 assert）。`update_obstacles` 遍历 `scene.objects` 且只跳过 `self.robot` / `visual_only` / `ignore_objects`；`get_action_space` 里存在 `isinstance(obj, Robot)` 过滤，说明 robots 在 registry 中——但**必须实测确认**。若不在，就在建世界时手动补 mesh。
2. **周期性重规划**：每 K 个 waypoint 重新 `update_obstacles()` 并重规划剩余段。这需要复制 `_execute_motion_plan` 的循环逻辑到 L1c（不改 OmniGibson 源码，而是自己驱动 `_plan_joint_motion` + 分段执行）。
3. **执行互斥兜底**（配置项 `motion_exclusive: true`）：同一时刻只允许一个 robot 执行**规划运动**，其余 idle。语义上仍是「并发决策 + 串行执行」，正好对应 chain/centralized 的协调语义；当 (2) 调不稳时用它保底出结果。

另外：`_execute_motion_plan` 在 10 步内跟踪不到 waypoint（`JOINT_POS_DIFF_THRESHOLD=0.005`）就抛 `EXECUTION_ERROR`。并发下机器人互相扰动会频繁触发 → 先把 `MAX_STEPS_FOR_JOINT_MOTION` 调大、`JOINT_POS_DIFF_THRESHOLD` 放松、`low_precision=True`。

### 3.4 L1e：`action_outcome` 契约

L2→L3 的成败契约。crafter 里 `reason` 是**字符串前缀匹配**（`reason.startswith("requires ") and "agent" in reason`），移植时改成结构化：

```python
{"status": "success" | "failed",
 "reason_code": "OK" | "PRE_CONDITION" | "SAMPLING" | "PLANNING" | "EXECUTION"
               | "POST_CONDITION" | "OBJECT_CLAIMED" | "NEEDS_MORE_AGENTS" | "TIMEOUT",
 "reason": str(ActionPrimitiveError),      # 自然语言，直接进提示词
 "metadata": e.metadata,
 "effects": {"held_delta": {...}, "state_delta": {...}, "task_delta": {...}}}
```

`reason_code` 直接由 `ActionPrimitiveError.Reason` 映射，外加 L1c 自己产生的三种。

---

## 4. L1d：跳过 BDDL 后的任务层（D4）

**任务侧配置**：`DummyTask`（`_load` 空操作、无终止无奖励、`valid_scene_types={Scene}`）+ 在 env config 的 `objects:` 里手写要用的物体（参考 `examples/wip/solve_simple_task.py`）+ **自己的成功判据**。
⚠️ 不要用 `GraspTask`：它没有成功条件（`GraspGoal`/`Falling` 被注释掉了）、硬编码 `env.robots[0]`，且非缓存 reset 路径已经腐坏（`_sample_pose_near_object(pose_on_obj=...)` 这个 kwarg 早就不存在了 → TypeError）。

**协作任务定义**（`coop_config.yaml`，与 crafter 的 `cooperative_collection` 一一对应）：

```yaml
cooperative_transport:
  enabled: true
  distance_threshold: 1.5          # 判定「在附近」
  tasks:
    - id: t1
      target_object: cup#1
      goal: {predicate: OnTop, reference: table#1}
      required_agents: 2           # ← spatial / temporal 约束的来源
      required_capability: null    # ← dependency 约束（如需先 open fridge）
```

`TaskState` 四元组的映射：

| COOP² | crafter | 本次实现 |
|---|---|---|
| task | 一个可采集资源实例 | 一条 `(target_object, goal_predicate)` |
| spatial | 附近 agent 数 ≥ required_agents | 距 target ≤ threshold 或同房间的 agent 数 |
| temporal | 同步发 collect 的 agent 数 | 同一 macro-step 对该 task 发出动作的 agent 数 |
| dependency | 持有 required_tool 的 agent 数 | 前置谓词已满足 / 持有必需物体 |
| participation | 参与者比例 | 同上 |

`StepMetrics` 的 improved/worsened 差分、`CapabilityChange`、`convert_to_serializable`、`plot_metrics_timeline` **原样搬** → `compute_constraint_metrics` 与结果表零改动跑通。

成功/部分进度：每 macro-step 评估所有 `goal.predicate` → `satisfied / total`，作为 `Y` 与 `score`。BDDL 接进来时，只需把这个评估函数换成 `task.compiled_task.check_goal(task._evaluate_predicate)`，**其余全部不动**（这也是先跳过 BDDL 的最大好处）。

---

## 4.5 观测设计：房间级 symbolic world graph，无 RGB（已核实可行）

### 形态：一个共享 builder + 每 agent 一个房间子图视图

```python
builder = SceneGraphBuilder(robot_names=[r.name for r in env.robots],   # ← N 个名字全传
                            full_obs=True,          # ← 必须！否则走相机 FOV
                            merge_parallel_edges=True,
                            only_true=True,
                            exclude_states=(Touching, NextTo))          # ← 保持默认
builder.start(env.scene)      # og.sim.play() 之后
# 每 macro-step：
builder.step(env.scene)
G = builder.get_scene_graph()
obs_i = G.subgraph(nodes_in_room(robot_i) | {robot_i} | {同房间其他 robot})
```

- **`full_obs=True` 是硬要求**：默认 `full_obs=False` 会调 `robot.states[ObjectsInFOVOfRobot].get_value()`（相机 `seg_instance`），而且多机器人时取的是**所有 FOV 的交集**（`objs_to_add &= objs_in_fov`，`graph_builder.py:182`）——对多智能体毫无用处。`full_obs=True` 时 `step()` 完全不碰 sensor/render。
- **`robot_names` 要传全部 N 个**：builder 会把不在列表里的机器人从图中剔除（`objs_to_add -= set(base_robots)`，`graph_builder.py:186-187`），否则 agent 看不见队友。
- **不要给每个 robot 各建一个 builder**：无全局状态所以技术上可行，但 `full_obs=True` 下 N 个 builder 会把同一份全场景 O(N²) 谓词算 N 遍。共享一个再取子图。
- `egocentric=True` 只用 `self._robots[0]`，是真单机器人；且与多 robot 互斥（构造时 assert）。

图的结构：`MultiDiGraph`（`merge_parallel_edges=True` 则 `DiGraph`）。
**节点是活的 Python 对象实例，不是字符串** → 序列化前先 `nx.relabel_nodes(G, {o: o.name for o in G.nodes})`。
节点属性：`pose`(4×4) · `bbox_pose`(4×4) · `bbox_extent`(3,) · `states`(一元布尔 dict)。
边：关系名 → `{"value": bool}`（合并模式下 `{"states": [(name, value), ...]}`）。默认排除 `Touching`/`NextTo`，剩 `Inside/OnTop/Under/Overlaid/Draped/AttachedTo/Contains/Covered/Filled/Saturated` 等。

### 房间归属（有两个坑）

| 需求 | API |
|---|---|
| 位置 → 房间实例 | `scene._seg_map.get_room_instance_by_point(xy)`（公开属性 `scene.seg_map`） |
| 房间类型 | `room_inst.rsplit("_", 1)[0]`（照 `behavior_task.py:367`） |
| 房间 → 所有物体 | `scene.object_registry("in_rooms", room_inst, default_val=[])`（`in_rooms` 是注册的 group key） |
| 机器人在哪个房间 | **无 API**，用 `get_room_instance_by_point(robot.get_position_orientation()[0][:2])` |

- ⚠️ **`obj.in_rooms` 是场景加载时的静态元数据，物体移动后不更新。** 唯一的写入者是 BDDL 采样（写完还要手动 `object_registry.update(keys=["in_rooms"])`）。把杯子从厨房搬到卧室，它的 `in_rooms` 永远是 `["kitchen_0"]`。
  → **固定家具**（`scene.fixed_objects`）用 `in_rooms` 静态先验；**可移动小物体**每步用点查询现算。机器人根本没有 `in_rooms`。
- ⚠️ `get_room_type_by_point` 有 bug：少了 `.item()`，拿 0-dim tensor 当 dict key，非边界点必 `KeyError`。**别用**。
- `_seg_map` **只存在于 `InteractiveTraversableScene`**（`Rs_int` 等），且它是 `cv2.imread` 离线读 `layout/floor_insseg_0.png`，与渲染器无关。
- 多场景/`VectorEnvironment` 时 seg map 是**场景局部坐标**，要先 `scene.convert_world_pose_to_scene_relative(...)`；单场景（`scene.idx == 0`）可直接传世界坐标。

### 彻底关掉渲染

| 设置 | 位置 | 说明 |
|---|---|---|
| `obs_modalities: []` | robot config | 现成配置全是 `[rgb]`。空 list 让该 robot 整个从 obs space 与 `get_obs()` 里消失（`env_base.py:349-351,485-487` 按 `maxdim>0` 门控） |
| `external_sensors: null` | `env` | 默认值；同时让 `env.render()` 变 no-op |
| `gm.HEADLESS = True` | macros | 官方 headless 路径 |
| `gm.RENDER_VIEWER_CAMERA = False` | macros | 默认 True，否则 headless 下 viewer 相机仍每帧渲染 |
| `n_render_iterations=1` | `env.step` | 默认值 = 零次额外 render |
| **`env.reset(get_obs=False)`** | 调用处 | `reset()` 里有**无条件的 1 次 `sim.step()` + 3 次 `sim.render()`**（`env_base.py:691-694`），但整段被 `get_obs=True` 包着 → 符号观测下全部可跳过。N robot × 多 episode 时很可观 |

**没有任何东西需要视觉模态**：动作原语（cuRobo 读 USD stage 的碰撞网格）、seg map（离线图片）、`full_obs=True` 的 scene graph 全都不碰相机。唯一需要 RGB 的是 `visualize_scene_graph()`——别调。

### 为什么 `in_rooms` 过期不影响 BDDL（两套机制，别混）

| | `in_rooms` / `inroom` | 关系谓词 `ontop/inside/under/...` |
|---|---|---|
| 性质 | 场景 JSON 的静态标注，registry group index | **每次调用从几何/物理现算，无缓存** |
| 谁读 | 只有 `BDDLSampler`，episode **装配期**（`bddl_utils.py:629-657, 903`：「要一个厨房里的 ashcan，场景里哪些符合」） | `check_goal` / `_evaluate_predicate`，**运行期** |
| 能进 `:goal` 吗 | **不能**。`InRoom` 类在 `predicates.py` 有定义、`TOKEN_TO_PREDICATE` 有注册，但 **`PREDICATE_TO_STATE` 里没有** → `bddl_utils.py:224` 无保护字典查找直接 `KeyError`。同样待遇：`Grasped`、`Broken` | 能 |
| 物体移动后 | 过期不更新（唯一写入者是采样器让新物体继承父物体房间，`:1413`，写完还要手动 `object_registry.update(keys=["in_rooms"])`） | 自动正确 |

关系谓词的求值路径（`bddl_utils.py:227-233`）：
```python
state_class = PREDICATE_TO_STATE[predicate_cls]
return obj1.states[state_class].get_value(obj2)      # ← 那一刻从 AABB/邻接射线/接触算
```
BDDL 侧只存**符号引用**（`Predicate.inputs` 是实例名字符串），`evaluate(evaluate_fn)` 把「谓词类 + 名字」回调出去，`BehaviorTask._evaluate_predicate` 再用 `object_scope` 换成活对象。**没有状态缓存、没有事件监听、没有增量维护** —— 这既解释了为什么 `check_goal` 不 step 也能随时调，也解释了为什么 `in_rooms` 过期无害。

→ 所以 §4.5 需要的**动态房间归属是 BDDL 本身不需要、我们额外要的能力**（给 LLM 看「你这个房间里有什么」），不是在绕 BDDL 的 bug。
→ `nextto` 理论上可当动态「同区域」判据，但底下是邻接射线投射、很贵（`SceneGraphBuilder` 默认排除它就是这个原因），seg map 点查询便宜得多。

### 开销与语义

- 一元布尔状态很便宜。二元 kinematic 谓词底下是 `Horizontal/VerticalAdjacency` 的射线投射（这也正是默认排除 `Touching`/`NextTo` 的原因），房间级 30–80 物体可接受；先用 `bbox_pose`/`bbox_extent` 按距离预筛候选对。
- `_get_boolean_binary_states` 用裸 `try/except: pass` 吞掉所有异常（`graph_builder.py:113-120`），调试时要注意。
- `step()` 里 `remove_edges_from(list(itertools.product(objs, objs)))` 每步物化一个 N² 元组列表 —— 又一个房间级裁剪的理由。
- **语义声明**：这是**特权的房间级观测**（无遮挡、无 FOV）。crafter 的 `symbolic_view` 在视窗内同样是特权的，所以是一致的设计，论文里注明即可。要更严格的部分可观测，再叠 per-agent "seen set" 记忆，或退回 per-robot FOV builder（代价：需要 `seg_instance` 相机 + 渲染，与「无 RGB」冲突）。

---

## 5. L2–L6 对环境的全部依赖（适配器 checklist）

### 5.1 生命周期
- `CooperativeBehaviorEnv(scene_model, n_agents, seed, length, robot_model, coop_config_path, ...)`
- `reset(seed=None) → (obs_dict, info_dict)`；`step(actions: Dict[agent_id, Any]) → (obs, rewards, terminated, truncated, info)` **五元组全是 dict**
- `step` 额外接受 `share_requests / place_requests / collect_requests`（`SymbolicEnvWrapper` 用 `inspect.signature` 探测，签名带 `**kwargs` 就算支持）→ 建议**统一成一个 `intents` 通道**
- `possible_agents: List[str]`（与 `agent_names` 按位置 zip 成 name_map）、`agents: List[str]`、`render()`、`close()`、可选 `set_team_score_time_limit(sec)`

### 5.2 `info[agent_id]` —— 真正的观测通道
⚠️ **Agent 从不看 `obs`**（crafter 里 obs 是 RGB，`observe()` 只存不读）。所有 LLM 可见状态走 `info`：

| key | 类型 | 消费方 |
|---|---|---|
| `symbolic_world_state` | 对象 | L2 的 `handler.execute` / 终止判定 |
| `symbolic_view` | **str** | 提示词 `## Symbolic View` |
| `target_hints` | **str** | 提示词 `## Current Reachable Targets`；follower 发言也用 |
| `action_outcome` | dict | **L2→L3 成败契约**（§3.4） |
| `task_states` | dict | 过程日志（来自 L1d） |

---

## 6. L0 配置与调参清单（D3 必做）

**环境配置**
- `scene.type: InteractiveTraversableScene`，`scene_model: Rs_int`，`seg_map_resolution: 1.0`，**`include_robots: false`**
- `robots`: N 个 `{model: "r1", name: "robot_0"/..., position: [...], grasping_mode: "sticky", action_normalize: false, self_collisions: true, reset_joint_pos: [...]}`——**必须显式给 `name`**（name 就是 action/obs 的 key），用 `model` 而非已废弃的 `type`
- controllers 必须是：`base: HolonomicBaseJointController(motor_type: position, command_input_limits: null, use_impedances: false)`；`trunk/arm_*/gripper_*: JointController(motor_type: position, command_input_limits: null, use_delta_commands: false, use_impedances: false)`。原语按**名字**索引 `controller_action_idx`，名字必须齐
- `task: {type: DummyTask}` + `objects: [...]`
- `gm.ENABLE_OBJECT_STATES = True`（`_place_with_predicate` 要读 `OnTop/Inside`）；`gm.ENABLE_TRANSITION_RULES = False`（搬运任务纯开销）；`gm.USE_GPU_DYNAMICS = False`

**原语构造**
- `StarterSemanticActionPrimitives(env, robot, enable_head_tracking=False, curobo_batch_size=3)`
  - ⚠️ **`enable_head_tracking=False` 必填**：默认为 True，而 `_overwrite_head_action` 里有 `assert robot.model == "tiago"`，`_grasp` 一设 `_tracking_object` 就会崩
- 构造前 robot 必须已在目标 reset pose（`_arm_targets` / `_reset_eef_pose` 在 `__init__` 冻结）
- **启动仪式**（照 `examples/wip/rs_int_primitives_example.py`）：把所有 gripper 开到归一化 1.0 → `robot.keep_still()` → step 5 次 → `scene.update_initial_file()` → `scene.reset()`。cuRobo 的 locked-joint / retract 配置假设的就是这个张开状态，跳过会立刻规划失败

**macros 调参**（`with macros.unlocked():`，且必须在构造任何 controller **之前**——macro 被读过一次后写入会抛异常）
| macro | 默认 | 建议 |
|---|---|---|
| `MAX_STEPS_FOR_JOINT_MOTION` | 10 | **调大**（并发扰动下 10 步常跟不上 → 假 EXECUTION_ERROR） |
| `JOINT_POS_DIFF_THRESHOLD` | 0.005 | 放松；或 `_move_hand(low_precision=True)` |
| `MAX_STEPS_FOR_SETTLING` | 500 | 调小以省 tick |
| `DEFAULT_COLLISION_ACTIVATION_DISTANCE` | 0.02 | 调大以留出队友余量 |
| `BASE_POSE_SAMPLING_UPPER_BOUND` | 1.5 | 多机器人挤同一物体时调大 |
| `HOLONOMIC_BASE_PRISMATIC_JOINT_LIMIT`（curobo.py） | **5.0 m** | ⚠️ 规划器的虚拟基座关节限位只有 ±5m，**跨房间搬运会静默规划失败**——必须调大 |
| `mesh_cache_size`（`create_world_mesh_collision`） | 2048 | 完整场景 + 3 个额外铰接机器人可能溢出，调大 |

**硬件注意**：cuRobo 在 cuda capability (12,0)（RTX-50 系）上会裁掉 embodiment；对 `r1pro` 会删掉 DEFAULT 键，而 `update_obstacles` / `check_collisions` 无条件索引 `self.mg[DEFAULT]` → **KeyError**。**用普通 `R1` 而非 `R1Pro`**，或打补丁。

---

## 7. 执行顺序（里程碑，按 D1–D4 重排）

| # | 里程碑 | 验收标准 |
|---|---|---|
| **M1** | L0 冒烟：N=2 个 R1 + `DummyTask` + 手写 objects，各构造一个原语集 | 启动成功，记录**启动耗时与显存**；assert 其他 robot 在 cuRobo 碰撞世界里 |
| **M2** | **L1c 并发调度器**（代码已就绪，待上机验证） | `feasibility_verify/multiagent_concurrent_primitives.py` 跑通：两个 robot 同时 `NAVIGATE_TO` 不同物体再各 `GRASP`，`--plan asymmetric` 证明不在原语边界对齐，`--mode exclusive` 作对照；无假 EXECUTION_ERROR |
| M3 | 骨架搬迁：复制 L3–L6，写 repair shim + viz stub | `import coop2.experiment.run_individual` 不报错 |
| M4 | L1a+L1b 世界模型与文本观测 | 打印的 `symbolic_view`+`target_hints` 足以让人类照着写出合法动作 |
| M5 | L2 + LLM 5 原语词表（三件套同时改：Pydantic models ↔ controllers ↔ `action_outcome.effects`） | 单 agent 走 COOP² 的 plan 通道完成一次「把 cup 放到 table 上」 |
| M6 | L1d 任务追踪 + `coop2_attempt_events.py` | `coop2_metrics.json` 全字段有值，`build_results_table` 出表 |
| M7 | 三拓扑验证（只改 role prose） | individual / chain / centralized 各 ≥3 seed 跑通 |
| M8 | 抽公共 `_run_episode()`，加 decentralized flow + runner，注册进 `run_grid` / `build_results_table` | 四拓扑网格实验 |
| M9（可与 M6 并行） | 接回 BDDL：把 L1d 的谓词评估换成 `compiled_task.check_goal`，其余不动。可选：按 §11 做原生 2-agent BDDL | 一个真实 BEHAVIOR activity 跑通 |

**M1→M2 是全部风险所在**，建议这两步做完再动 L3 以上（那些是复制粘贴，风险低）。

---

## 8. 已删除的 `symbolic_llm/` 结构（从 .pyc 反编译，作为 L1/L2 蓝图）

源码找不回了，但模块划分与签名可以照抄：

```
types.py          SymbolicObservation / EntityObservation / PredicateFact / GoalStatus
                  ActionCandidate(.id, .to_dict) / AgentAction(.id, .from_mapping)
                  ExecutionResult / DecisionRecord / PlannerUsage / candidate_ids()
observation.py    PrivilegedSymbolicObservationAdapter.build(env, step, max_steps,
                      last_action_id, last_error)
                  _abilities / _rooms / _observable_states / _held_entities
                  _flatten_predicate（含否定的 BDDL atom → 字符串）
                  _goal_predicates / _predicate_facts(options, evaluate_fn)
action_catalog.py SymbolicActionCatalog.enumerate(observation, cooldowns)
                  _relevance / _relation_targets / _is_grasp_candidate / _candidate
executor.py       SymbolicPrimitiveExecutor(env, robot, attempts)
                      .execute(action, remaining_steps, metric, step_callback)
planner.py        ConstrainedActionPlanner(client).plan(observation, candidates, history)
openai_client.py  OpenAIPlannerClient(config, client, io_logger).choose(messages, candidates)
                  ._schema(candidates)  ← 把候选集约束进 structured output
                  ScriptedPlannerClient(action_ids, io_logger)   ← 无 LLM 的回归测试
agent.py          SymbolicLLMAgent(planner, config).decide/record/stopping_reason
                  ._tick_cooldowns
runner.py         SymbolicLLMRollout(env, agent, config, logger, video_recorder, seed).run()
logging.py        RolloutLogger.log_event / log_model_io / write_summary
video.py          AgentViewVideoRecorder(robot, output_path, sensor_name, fps)
```

三个值得保留的设计（COOP² 里没有，比 crafter 那套更好）：
1. **候选集约束生成**：`_schema(candidates)` 把当前可行动作枚举进 structured output schema，LLM 只能选合法项 —— 比 crafter 的「自由生成 + 事后校验」更稳。可以叠加在 `LLMClient` 之上。
2. **`cooldowns`**：动作冷却，防止 LLM 死循环重试同一失败动作。
3. **`log_model_io`**：逐次记录模型请求与原始响应，做 prompt 调试必需。

---

## 9. 移植坑清单（已核实）

**ma_crafter 侧**
- `PlanningEnvWrapper(..., log_file=, interrupt_on_message=)` 这两个参数**是死的**（只在 `env` 传类而非实例时生效）。真正的中断策略在 `MessageBroker.send_message`：**W 态收消息一定中断；X 态只有 `interrupts_execution=True` / `leader_broadcast` 才中断**。
- `env.agents` 二义：`SymbolicEnvWrapper.agents` 是 `List[str]`，`PlanningEnvWrapper.agents` 是 `Dict[str, Agent]` 并遮蔽前者；runner 依赖 dict。
- `plan.specification` 字符串 `collect_wood(tree#123)` 是**承重的序列化格式**，被 `_task_name_from_spec` 正则和 `RESOURCE_TARGET_PATTERN` 二次解析。保留 `task(type#id)` 约定，或三个消费方一起改。
- 协议自旋（`wait_for_messages_from` + 50ms sleep）**没有绝对超时**，只有 leader 的 30s 预算和 runner 的 wall-clock deadline；超时 break 后线程不 join → 给每个 `_execute_flow` 加硬超时。
- `broadcast_history` 整个 episode 累积、只在 reset 清空 → 提示词无界增长，顺手修掉。
- 写盘顺序：`save_logs` → `llm_usage.json` → `compute_all_metrics`（后者读前者算 scaling_efficiency）。
- 即使 `--repair off`，`PlanningEnvWrapper` 也会无条件构造 `Coop2TraceLogger` / `Coop2RepairController(enabled=False)` / adapter / dispatcher，且 `coop2_messages.py` 硬 import `coop2_repair.message_protocol` 的三个常量 → shim 里保留一个「list + `.save(path)`」的 TraceLogger、一个 `before_execution() -> None` 的空控制器，常量内联，`plan_env_wrapper.py` 就能一行不改。

**BEHAVIOR-1K 侧**
- `scene.include_robots: true`（`r1pro_behavior.yaml` 就是）会让 `robots:` 列表**被静默忽略**。
- `og.sim` 是**进程级单例**（dt/device/viewer 有 assert）→ 一个进程一个 env，并行实验必须 subprocess 扇出（`run_grid.py` 本来就是这么干的）。
- `apply_ref` 的重试**不幂等**（`_place_with_predicate` 失败时物体已 release，第二次必然前置失败）→ `attempts=1`。
- `RELEASE` 释放**所有**手臂；symbolic/starter 都是单臂（`self.arm`），双臂 R1 也一样。
- 原语显式禁止把 `Robot` 当参数——但 starter 版的 `_grasp` 只检查 `isinstance(obj, USDObject)`，而 Robot 传递地满足它，**过滤要自己加**。
- `Environment.reset(get_obs=False)` 返回 `None`；`EnvironmentWrapper.reset()` 不能转发该 kwarg。
- `_navigate_to_pose_direct` / `_rotate_in_place` 是真闭环速度控制器但**无人调用**，且 `assert motor_type == "velocity"`，与 `position` 配置冲突——想用来做反应式避障就得改基座控制模式，那会破坏规划路径。

---

## 11. BDDL 的多智能体支持度（已核实：不是根本不支持）

分四层，**语言层和求值引擎完全 agent 无关，卡点只在 OmniGibson 绑定层的硬编码**。

| 层 | 状态 | 证据 |
|---|---|---|
| (a) 语言 / 解析器 | ✅ **完全 agent 无关** | `parsing.py / config.py / activity.py / predicates.py / logic_base.py / condition_evaluation.py` 六个文件里 grep `agent` **零命中**。`:objects` 的解析（`parsing.py:263-276`）对 `agent.n.01_1 agent.n.01_2 - agent.n.01` 与 `bowl.n.01_1 bowl.n.01_2` 处理完全相同，**没有任何 arity/cardinality assert**。量词 `Universal/Existential/ForPairs` 也 agent 无关，`ForPairs` 还自动排除自配对（正好用于 agent 间目标） |
| (b) 求值引擎 | ✅ **完全 agent 无关** | `Predicate.evaluate(evaluate_fn)` 就是 `evaluate_fn(type(self), *self.inputs)`；`evaluate_state` 是一个平坦循环，返回 `(all_satisfied, {"satisfied":[idx], "unsatisfied":[idx]})`。没有任何 `scope["agent"]` 特例。`CompiledTask.check_goal` 直接委托给它（`knowledge_base/models.py:964`） |
| (c) OmniGibson 绑定层 | ⚠️ **唯一的实际阻碍，约 40 行** | 见下表 |
| (d) 谓词映射 | ⚠️ **缺一行字典** | BDDL3 **已删除所有 agent-implicit 谓词**（`holding/inhandofrobot/inreachofrobot/insameroomasrobot` 全部不存在）→ agent 只能作显式参数出现，这是好消息。`Grasped(agent, obj)` 二元谓词语言层已定义（`predicates.py:226,262`），但 `PREDICATE_TO_STATE`（`bddl_utils.py:180-201`）里**没有它** → goal 里写 `grasped` 会在 `bddl_utils.py:224` 抛 `KeyError`。而 `nextto`/`touching` 已有映射，`(nextto agent.n.01_1 agent.n.01_2)` **现在就能求值** |
| (e) 现成任务定义 | ❌ 1000 个基本全是单 agent | 实测 `picking_up_trash/problem0.bddl`：agent 就是 `:objects` 里的普通声明 `agent.n.01_1 - agent.n.01` + `:init` 里一条 `(ontop agent.n.01_1 floor.n.01_2)` |
| (f) 离线校验器 | ⚠️ 有单 agent 假设，但**任务加载时不执行** | `bddl_verification.agent_present(init)`（`:844-849`）要求字面 `(ontop agent.n.01_1 ...)`；`check_synset_predicate_alignment`（`:307`）只对 `ontop agent.n.01_1` 早退，第二个 agent 的 init 会走进 `syns_to_props` 查表挂掉。只有数据集作者离线跑，不阻塞运行 |

### (c) 层硬编码清单

| 位置 | 内容 |
|---|---|
| `behavior_task.py:303,321` | `object_scope` 种子写死 `{"agent.n.01_1": None}` |
| `bddl_utils.py:473` | `self._agent = self._env.robots[0]` |
| `bddl_utils.py:553` | `self._object_scope["agent.n.01_1"] = self._agent` |
| `bddl_utils.py:1148-1151` | 只把 `robots[0]` 传送到 `[300,300,300]` 让位 |
| `bddl_utils.py:802-803` | `remaining_kinematic_entities -= {"agent.n.01_1"}` |
| `behavior_task.py:511-522` | `get_agent()` 写死 `env.robots[0]` |
| `behavior_task.py:229-247` | `reset()` 只给 robot 0 摆 presampled pose |
| `behavior_task.py:594-615` | `_get_obs` 的 `in_gripper` 只算 robot 0（eval 里 `include_obs=False`，无害） |

**已经替你写好的部分**：`assign_object_scope_with_cache`（`behavior_task.py:546-550`）已支持
```python
if "agent.n." in obj_inst:
    idx = int(obj_inst.split("_")[-1].lstrip("0")) - 1
    entity = env.robots[idx]
```
即 `agent.n.01_2 → env.robots[1]`。另外这版**已经没有 `BDDLEntity` / `OmniGibsonBDDLBackend` 包装类**，`evaluate_bddl_predicate` 直接吃 sim object，scope 就是普通 `{str: obj}` dict，改动面因此很小。`bddl_utils.py:1418,1486` 的两处 agent 分支用的是**子串**判断（`"agent" in obj_inst`），已经天然支持 `agent.n.01_2`。

### 做一个原生 2-agent BDDL goal 的改动清单（易→难）

1. 自写 problem 文件：`:objects` 里 `agent.n.01_1 agent.n.01_2 - agent.n.01`，`:init` 里各给一条 kinematic 条件。**零代码改动**
2. `PREDICATE_TO_STATE` 加一行 `bddl_predicates.Grasped: object_states.IsGrasping`（`bddl_utils.py:180-201`），并把 `Grasped` 加进 `UNSAMPLEABLE_PREDICATES`（`:204`）。**这一行让 `(and (grasped agent.n.01_1 handle_1) (grasped agent.n.01_2 handle_2))` 变得可表达且可求值**
3. `behavior_task.py:303,321` 的 scope 种子改成按 `parsed_objects.get("agent.n.01", [])` 循环
4. `bddl_utils.py` 四处 de-hardcode（`:473,553,1148,802`），并加 `assert len(env.robots) >= 声明的 agent 数`（否则 scope 里留 `None`，最后表现为莫名的 `False`）
5. `get_agent(env, idx=0)`、per-agent presampled pose 列表
6. config 加第二个 robot（缓存路径已经好了）

**不需要改** `condition_evaluation.py / logic_base.py / parsing.py / activity.py / config.py` 中的任何一行。

### 对 COOP² 的两个含义

- 「先跳过 BDDL」的决定依然正确（M1–M8 不该被任务定义拖住），但**接回成本远低于原估计**，可与 M6 并行。
- ⚠️ `get_potential`（`behavior_task.py:408-413`）是**全场景全 literal 的单一标量**，没有 per-agent credit assignment。decentralized 方法大概需要自己做目标分解（按 `goal_condition.get_relevant_objects()` 把 literal 归属到 agent）——这是**增量**，不是对现有代码的改动。

---

## 10. 建议下一步先读的文件（本次未 stage）
- `omnigibson/robots/` — `q_to_action`、`controller_action_idx`、`base_footprint_link`、`curobo_path`
- `omnigibson/scenes/scene_base.py` — `objects` 是否含 robots（§3.3 第 1 条要 assert 的那点）、`_seg_map`、`reset` 语义
- `omnigibson/utils/` — `detect_robot_collision_in_sim`、`ControllerView`
- `macrafter/coop_env.py`（coop2 侧，本次未 stage 但存在）— 对照 §5 契约的原始实现
