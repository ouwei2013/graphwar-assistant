# 本科毕业论文扩展方案：基于 PPO 与 MLP 的 Graphwar 全局战术决策

> **状态：研究与实施提案（不是已实现功能）**。目标是最大化整局游戏胜率，不以单发命中率作为最终目标。本提案不替代仓库现有的 PROJECT_PLAN_V2.md；它将复杂的原计划收缩为可以独立验证的本科论文课题。

## 1. 题目与研究问题

建议题目：《基于深度强化学习的 Graphwar 多回合战术决策方法研究》（Deep Reinforcement Learning for Multi-turn Tactical Decision-Making in Graphwar）。

研究问题：在**固定相同函数求解器与候选动作集合**的前提下，PPO+MLP 是否比随机、即时击杀优先以及人工规则策略取得更高的**整局胜率**？在未见过的地图和对手上是否仍然有效？

- **主实验**：固定候选生成器，比较不同决策策略的胜率。
- **消融实验**：只用终局奖励 vs 加入中间奖励；候选数量与多样性变化；训练对手组合。
- **扩展（非必需）**：self-play、感知噪声鲁棒性、MLP vs Deep Sets 编码。
- **不作为本次毕设范围**：从截图端到端生成数学表达式；直接由 PPO 输出函数族和所有连续参数；完整多模式微分方程支持；多种高级搜索、Transformer 同时开发。

## 2. 与现有仓库的关系 / 依赖

截至仓库 README 所记录的 v0.1（2026-10-25），已验证屏幕截图、士兵及障碍检测；**坐标校准、完整仿真器、函数求解器、决策层和评估实验仍未完成**。因此 RL 训练不能立即开始。

前置条件按依赖顺序：
1. 确认游戏中的射手相对坐标、函数语义、参数范围与合法函数表达式，完成真实游戏试射校准。
2. 实现**单次攻击仿真**：曲线、碰撞、友伤、爆炸、障碍变化。
3. 实现**完整回合转换**：当前行动者、双方轮流、士兵死亡、终局判定、游戏重置。
4. 用 GA/CMA-ES 或解析种子+搜索产生合法的候选函数，支持不同目标与战术效果。训练与评估均复用**同一个冻结版本**的候选生成器。
5. 在固定随机种子、固定初始地图下，基于仿真器能够完成大量可复现的自动对局。

**重要**：模拟器不准确会造成 sim-to-real 差距。训练只在经校准的模拟器内大批量开展；真实客户端用于规则验证和少量独立验收，而不是用 GUI 大量训练。

## 3. 强化学习环境设计

采用 Gymnasium 风格接口：
- `reset(seed)`：生成训练地图，返回规范化状态和附加信息。
- `step(action_index)`：执行选中的函数，按预定的对手策略完成回应（或者显式逐步轮换到下一个行动者），返回 `obs, reward, terminated, truncated, info`。
- 决定采用哪种回合语义之后**全程保持一致**；不能把对手回应隐含于训练数据、又在评估时当作单独一步。
- `terminated` 用于胜败等自然终局；`truncated` 用于最大回合数等时间限制。超时平局按预先冻结的规则记分。
- 严格记录当前行动队伍、行动士兵、随机种子和对手版本；状态必须满足马尔可夫性，若敌方能力有隐藏因素，须明确将其视为部分可观察情形。

### 状态 / Observation

第一版使用结构化向量，不使用 CNN。建议由以下组成：
- 当前射手与其他士兵的**相对坐标**、归一化坐标、队伍、存活、行动状态。
- 障碍中心、半径和是否有效；采用固定最大数量 + padding mask，并以确定性顺序排序。
- 回合与当前行动者、双方剩余士兵数以及实际规则所需的附加变量。
- 固定场景规格（例如 3v3、至多 12 个圆形障碍）；输入维度由最终状态 schema 生成，不将“76 维”当作协议。

输入标准化到稳定区间；不要无意泄漏敌人的未来动作或仿真器内部答案。每次动作后根据新状态重新生成候选。

### 候选动作 / Action

优先使用**离散候选选择**：每回合候选生成器返回至多 K 条合法且不重复的函数。例如 K=10 或 16。动作是 `0..K-1` 的一个索引，缺失候选必须被 mask 掉。

候选不应该只是同一种“最高命中率”的函数副本。需要覆盖：单杀、多杀、保守/低误伤、地形改造、不同目标与函数族；保证至少一个 fallback/合法动作。缓存候选只在当前状态有效，下一状态需重新生成。

每个候选的特征 `c_i` 可以包括：
- 函数族 one-hot，经过范围归一化的参数向量和参数 mask；
- 预估敌方击杀数、友伤数、地形改变等（来自同一冻结仿真器）；
- 计算成本、距离或安全裕量；**不得包含未来真实胜负**。

为了避免模型“第 3 个位置总是比较好”的伪规律，应在训练时随机打乱候选顺序并保留动作映射；评估时可以测试置换不变性。候选数不足 K 使用 mask。

## 4. MLP Actor-Critic 网络

推荐 PPO + 小型 MLP，而非固定 10 个与候选位置绑定、不能读取候选内容的简单分类器。

```text
scene features s ── MLP(128,128) ── state embedding h
                                  ├─ Critic MLP → scalar V(s)
candidate features c_1 ───────────┤
           ...                    └─ Shared Actor MLP([h,c_i]) → logit_i
candidate features c_K ──────────────────────────────────────→ logit_K
                                           masked softmax(logits) → π(a|s,C)
```

- **Actor**：对每个候选调用共享参数的打分网络，生成 K 个 logits，设置无效动作 logit=-∞ 后做 Softmax；采样行动并记录其 log_prob。
- **Critic**：输入状态并估计当前策略的期望折扣累计奖励 `V(s)`，**不是默认直接输出胜率**。
- 初期 MLP 可用两层 128 hidden units、ReLU/Tanh 等作为起点，通过验证曲线调整，不把这些超参数作为最佳结果预设。
- 若候选集合本身携带不同动作可行性信息，Critic 可以再加入候选集合的 permutation-invariant pooling；第一版先用状态编码即可。
- 训练必须在采样时与 PPO 更新时使用**同一批候选特征、动作 mask、动作映射**，否则 old/new log probability 不对应。
- 可用 PyTorch 自定义策略分布 + PPO；若使用 sb3-contrib MaskablePPO，默认 MlpPolicy 仅适合固定长度状态与动作遮罩，**不能自动实现上述共享候选评分网络**，需定制 policy/features extractor 和正确的 rollout 存储。

## 5. 奖励设计：最终结果优先

主实验先用纯终局奖励：
- 胜：+1
- 负：-1
- 中间步骤：0
- 平局：0（如果游戏允许，提前固定判定规则）

用 PPO 的折扣回报与 GAE 将终局结果传播到前面的决策；推荐初始 `gamma` 例如 0.99，具体以实验为准。

对照组可使用 shaping，例如击杀敌人 +0.1、误伤队友 -0.15，但**不可只报告 shaping 后总奖励**：最终评价仍看真实终局胜率。需要警惕“刷击杀奖励而输掉游戏”，最好同时报告纯终局奖励训练结果。中间奖励系数必须预先登记或在独立验证集调参，避免用测试集选择最优系数。

## 6. 数据如何收集：在线交互而不是预先标注

强化学习不需要人工标注“最佳攻击函数”。核心数据通过**仿真器 + 当前策略 + 对手**自动生成：

```text
for episode in episodes:
    obs = env.reset(seed=...)
    while not done:
        candidates = frozen_solver.generate(obs)
        mask = valid_candidates(candidates)
        action, old_logp, value = actor_critic.sample(obs, candidates, mask)
        next_obs, reward, terminated, truncated, info = env.step(action)
        rollout.add(obs, candidates, mask, action, old_logp,
                    value, reward, terminated, truncated, info)
        obs = next_obs
    # 按 PPO 的 on-policy rollout batches 计算 advantage/returns 并更新
```

上面是数据流示意，不是可直接执行的训练程序。**PPO 为 on-policy**：不应将全部历史局数无差别混合、重复当作监督学习数据。历史数据可存储用于调试、分析和离线评估，PPO 更新以当前策略的 rollout 为主。

建议保存两个层次：
1. **episode metadata**：游戏规则版本、模拟器/求解器版本、对手 ID、map ID、seed、双方起始配置、最终结果、训练/验证/测试划分；
2. **transition/rollout**：episode_id、t、状态向量、所有候选表达式与特征、action mask、选择动作索引/表达式、old_log_prob、value、reward、next_state、terminated/truncated、最终赢家、动作实际效果、必要时随机数状态。

训练日志中的玩家昵称/截图按仓库现有隐私要求脱敏；大体积数据和模型权重不直接提交 Git。

### 训练数据规模（规划值，不是保证收敛的数据需求）

- 100–500 局：检查环境与日志正确性；
- 1,000–5,000 局：验证 PPO loss、value 和回报是否可学习；
- 10,000–100,000+ 局：视仿真速度和稳定性逐步增加；
- 测试：对每类固定对手分别执行足够多局（例如约 1,000 局），汇报胜率及 95% 置信区间。

优化对象是整局胜负，所以“一个函数命中率高”不是训练标签。不能将监督学习 `scene → GA 最优参数` 当作 PPO 数据集。

## 7. 对手与训练课程

1. 首先由 Random legal-action agent 验证环境和 Agent 能否学习；
2. 再加入 Greedy（本回合击杀数优先，友伤作为约束）；
3. 再加入 Rule-based（威胁、友伤、下一回合暴露）；
4. 可选使用冻结的历史 PPO 快照进行 self-play 或 opponent pool。

为避免只记住某一个对手，训练中采样多类对手；**测试时固定一组训练未使用的地图随机种子和策略版本**，禁止根据测试胜率反复调参。双方交换先手和阵营以消除偏差；记录对手版本和初始局面。

## 8. 实验矩阵及指标

固定同一版本 solver、候选集规则、模拟器和地图划分。对照：
- Random candidate selection；
- Greedy immediate kills（同分时友伤更少）；
- Rule-based heuristic（预先冻结手工权重）；
- PPO + MLP（纯终局奖励）；
- PPO + MLP（shaped reward）；
- 可选 PPO + opponent pool/self-play。

**主指标**：独立测试对局胜率，含每个对手的统计量及 95% CI；成对初始局面 + 交换阵营和先手。

**辅助指标**：平均每局回合数、友伤次数、平均击杀、不同地图难度的胜率、策略推理耗时与**完整决策耗时（含 GA 候选生成）**。报告 GA 候选池缺乏某类动作导致策略上限受限的失败案例。

**必要消融**：
- terminal-only vs reward shaping；
- K=5/10/20（保持公平的求解预算，特别注意 K 增大意味着计算开销变化）；
- seen vs held-out maps；
- 随机/贪心/规则对手及对手组合；
- 可选 state-only fixed logits vs candidate-aware shared scorer。

不仅评估 PPO 训练奖励；必须冻结模型用独立测试地图做真实胜负评估，不宣称未经测量的胜率改进。

## 9. 实现里程碑与验收

| 阶段 | 任务 | 最小验收 |
|---|---|---|
| M1 | 坐标/真实游戏规则校准 | 多张试射记录可复现，函数坐标与碰撞语义明确 |
| M2 | 最小可用仿真 | 对照真实客户端的弹道、命中、死亡和地形一致性报告 |
| M3 | 冻结 GA 候选器 | 不同目标/函数族的有效候选；合法性、去重、耗时测试 |
| M4 | 完整自动对局环境 | Random vs Greedy 可多局无人工参与运行并复现 |
| M5 | PPO/MLP 训练与日志 | masked actions 正确、rollout 一致，训练曲线可生成 |
| M6 | 消融与独立评估 | 胜率表、置信区间、失败案例、复现命令 |

如果 M1–M4 延迟，应收缩场景为**固定三对三、单一普通函数模式、有限障碍**，不要优先添加 CNN/Transformer/端到端函数生成。若模拟器校准无法达到可用程度，不应将模拟胜率作为真实游戏有效性的证据。

## 10. 建议实现目录（以后分 PR 落地）

```text
botmain/src/gwbot/rl/
    environment.py       # Gymnasium-compatible env + transition semantics
    observation.py       # state encoder, padding/masks, normalization
    candidates.py        # frozen solver adapter and candidate features
    policies.py          # shared MLP actor and critic
    opponents.py         # random, greedy, rule-based
    rewards.py           # terminal reward and shaping ablations
    rollout.py           # data schema and rollout persistence
botmain/scripts/
    train_ppo.py         # train only in calibrated simulator
    evaluate_agents.py   # held-out maps/opponents, paired comparisons
botmain/tests/
    test_rl_environment.py
    test_rl_action_mask.py
    test_rl_reproducibility.py
    test_rl_policy_shapes.py
```

以上为架构建议，**不宣称这些文件或训练功能当前存在**。此 PR 只引入研究计划，等待确认后将 M1–M6 分为小型代码 PR。

## 11. 面向论文的结论与边界

论文的可证伪主张是：**在同一冻结求解器生成的动作集合下，PPO 通过学习多回合后果，可以取得高于 Greedy 和 Rule-based 的胜率**。如果结果未超过规则算法，也应分析动作覆盖、奖励稀疏、仿真偏差、训练样本与对手多样性，而不是改写实验定义。

仅在离线/自己创建的房间/允许的内置电脑模式内测试；不自动进入公开服务器或排位，不分发游戏 jar；保留用户确认开火的辅助器运行形态。

