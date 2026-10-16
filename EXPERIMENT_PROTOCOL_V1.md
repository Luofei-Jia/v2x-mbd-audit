# V2X MBD Reassessment — Experiment Protocol v1（预注册风格）

日期：2026-10-13 | 状态：v1 骨架（文献对齐与序列级实验待补）
原则：所有论文数字必须来自 `results/` 机器可读 JSON；预注册后禁止事后调参。

## 1. 目标

量化三件事：① 随机划分的记忆化抬分幅度（审计）；② 诚实协议（时间窗+sender-disjoint）
下的逐攻击难度地图；③ 跨场景/跨数据集转移退化。负结果如实报告（身份泄漏已证伪为次要
效应，必须写入）。

## 2. 数据与溯源

- 数据：VeReMi NextGen，Zenodo record **19665762**，DOI 10.5281/zenodo.19665762，
  出版日期 2026-06-08，CC-BY 4.0。已用子集：highway_2 + urban_2 全部 38 文件（38 个
  md5 校验和见 `/tmp/vnx.json`，原始 zip 已删除）。
- 布局：`data/nextgen/<Name>/<Name>/{Train,Validation,Test}/<scenario>/veh_<id>.json`
  （每接收方一个文件）。
- **去重规则**：广播消息按 `(sender_id, messageID)` 去重；副本标签不一致时取
  `max(attacker)`（dataReplay 重放会制造标签冲突）。去重后 unique 数必须等于官方
  groundTruth 文件行数（实测吻合）。
- 标签：攻击子集消息内 `attacker ∈ {0,1}`；base 子集全正常。

## 3. 划分协议（固定构造，禁止变体）

- **P1 官方时间窗**：Train/Test 直接用官方划分。注意：官方划分 Train∩Test 共享
  ~37% 发送方（时间窗式，同一批车跨窗出现）。
- **P2 sender-disjoint**：train = Train 中发送方 ∉ Test 发送方集合的消息；test =
  Test 中发送方 ∉ Train 发送方集合的消息。
- **P3 随机**：Train+Test 合并后消息级随机 70/30（RandomState(42) 置换，一次性生成）。

## 4. 特征集（固定定义）

- **F1 瞬时运动学**：delay_ms、s/r_profile、s_posx/s_posy、s_edge、
  s/r{spd,spd_noise,acl,acl_noise,hed,hed_noise}。
- **F3 = F1 + 物理/速率**：dt_s、rate(=1/dt_s)、consistency(=|Δpos/Δt − s_spd|)、
  move(=Δpos 距离)。per-sender 差分，split 内部计算（因果，仅用过去）。
- **F4 = F3 + 跨域一致性**：dt_rcv_s、dt_mismatch(=|dt_send−dt_rcv|)、
  dir_consistency(=|atan2(Δpos) − s_hed| 圆差)。
- NaN 填充 −1（首条消息无差分）。

## 5. 模型与基线（2026-10-13 按前沿调研扩充）

- 已跑：RF 200 树 × 3 种子（0/1/2），n_jobs=-1，消息级。
- **SOTA 基线清单（单机 RTX 5090 可跑，逐条落实）**：
  ① RF+物理特征（已完成，统计锚点）
  ② VeReMi 参考统计检测器复现（VeReMi-Dataset GitHub 有参考实现：acceptance-range/
     速度一致性等）
  ③ LSTM 序列模型（per-sender 序列；TVT 2025 用 LSTM 层但 DRL 太重，只复现 LSTM 部分）
  ④ Transformer encoder（SwanLab 上有 VeReMi Transformer 实验先例，无代码，自实现）
  ⑤ ST-GNN（用 NextGen 位置/收发关系自建邻接；不直接用 VeReMi-Graph 的 PyG 对象——
     那是另一个数据集）
- 引证不实跑：DRL-MDS（TVT 2025）、元学习 few-shot（IET ITS）。
- 单机约束：全部本地跑（RTX 5090），不用服务器。

## 6. 指标与阈值

- 主指标：消息级 AUROC + AUPRC（3 种子均值±std）。
- 事件级指标（已完成，§9i）：连续攻击片段、1%/0.1% FPR 下 TPR、告警负担。
  注意：事件 AUROC 受 max 聚合 artifact 污染，论文只报 FPR 工作点。
- 阈值冻结：任何阈值只能在 Validation 上选，Test 只碰一次。

## 7. 结果台账

`results/*.json`（pilot_leakage.json、pilot_leakage_v2.json、sweep_attacks.json、
sweep_f4.json、transfer_cross_scenario.json）为唯一数字源；脚本与 JSON 一一对应。

## 8. 已定结论（勿推翻）

1. 审计：15 攻击中 13 个 P3_F1 > P1_F1，8 个差 >0.25，最大 +0.52（dos）。
   timeDelay 冒烟枪：attacker/normal delay_ms 边际分布一致 + P3=0.997 vs P1=0.49。
   修订（2026-10-16 JSON 复核）：严格为正实为 15/15；13/15 的差 ≥0.03（最小
   +0.0321，次小 +0.0179 与 +0.0001）；论文采用「≥+0.03 for 13 of 15」口径。
2. 难度地图（P2 + F3/F4）：平凡 6（dos/zeroSpeed/constSpeedOff/randSpeedOff/
   randPosOff/Sybil ≥0.99）；容易 5（suddenConstSpeed/feignedBraking/suddenStop/
   accelMult/reversedHeading 0.95-0.98）；中 1（constPosOff 0.83）；难 3 =
   **dataReplay 0.66、posMirroring 0.70、timeDelay 0.50**（真战场，需跨发送方推理）。
3. 身份泄漏（H2）次要：P2 vs P1 ≤3 点。如实写。

## 9. 已定结论补充（2026-10-13 转移实验）

- 跨场景转移：12/15 攻击 highway↔urban 双向近乎无损（物理特征场景不变）；
  难 3 攻击双向依旧难（timeDelay 0.47/0.47、dataReplay 0.68/0.73、posMirroring
  0.66/0.49）。→ 第四支柱叙事：迁移学习在解决对多数攻击不存在的问题。
- **F5 转移补充（results/transfer_dataReplay_f5.json）**：dataReplay 加碰撞特征后
  highway→urban 0.907 / urban→highway 0.911——接收方局部特征场景不变，第四支柱
  对 dataReplay 同样成立（"正确特征免费转移；迁移学习解决不存在的问题"）。

## 9b. Pending

- 70% 渗透率轴：暂缓（磁盘约束）。
- 已完成并关闭：序列级基线（§9f）、ST-GNN 基线（§9h）、事件级指标（§9i）、
  跨数据集轴（§9j）、文献补读（§9g）。

## 9g. 文献补读结论（2026-10-13，task #3 完成）

- **IET ITS 2026（10.1049/itr2.70182）**：VeReMi+Extension，域感知 Reptile 式
  元学习 + adapter few-shot（适配 <1s），**平均 AUC 0.70 / F1 0.62**——按域划分
  的诚实协议下 SOTA 也只有 0.70，佐证"未见攻击≈随机"叙事（TVT 2025 tabula rasa
  0.37-0.49 同向）。
- **两篇综述**：IEEE OJCOMS（Lone et al. 2025/26，10.1109/OJCOMS.2026.3679935）：
  "system-level metrics required for deployment claims"、图检测器须评估"unseen
  scenarios/trajectories"泛化；TETT 2025（Collective Perception，
  10.1002/ett.70267）；Bißmeyer 经典综述："static datasets 会过拟合已知攻击实现"。
- **关键空白确认**：检索确认**没有任何综述明确点名随机划分泄漏**——最接近的批评
  是"过拟合静态数据集/预处理不一致/场景迁移"→ 我们的泄漏审计（P1/P2/P3 协议化
  量化 + 冒烟枪）是该空白的第一次系统测量，论文定位成立。
- 前沿方法补充：PAMPOS（因果 Transformer，无监督只训正常轨迹，19 类 VeReMi++
  攻击 AUC 至 0.98）——引证 + 对照（我们证伪"随机划分下的高分"）。

## 9c. 难 3 攻击源码审计（2026-10-13，VeReMi-NextGen GitHub attackGenerator.py）

攻击在**后处理**阶段注入（非仿真内），逐接收方文件独立改写。难 3 攻击实现语义：

1. **timeDelayAttack**：`sendTime += δ; rcvTime += δ`（同一发送方恒定 δ =
   randint(2,4) 秒）→ **所有相对时序特征（delay_ms/dt_s/dt_rcv_s/dt_mismatch/rate）
   构造上不变**；且 receiver 字段冻结在真实接收时刻 → 真实世界的距离膨胀信号
   （sender 位置过期）被后处理**抹除**。唯一可分离信号 = receiver_pos 相对 rcvTime
   的**滞后伪迹**（≈v_r·δ ≈ 28-120m），属数据集构造产物，非现实检测能力 → 须
   如实标注 artifact。
2. **dataReplay**：攻击者消息被替换为受害车 ≤5s 前、≤400m 内的完整消息内容
   （sender_id/alias/时间戳换成攻击者的，receiver 字段逐接收方重插）。per-sender
   位置跳变只在几何条件成立时触发（0.66 的来源）；**干净信号 = 同一接收方文件内
   不同 sender_id 的内容精确碰撞**（重放内容与受害车早前消息逐字段相同）——
   现实中 MDS 可存近期消息做碰撞检测，属合法特征。
3. **positionMirroring**：位置沿道路法向镜像（偏移量 = 真实位置到最近道路距离），
   **航向不变** → 短时 Δpos 方向不受影响（dir_consistency 构造上失效）；设计目标
   即"看起来像对向车道的逆行车辆"（论文原话）。合法信号 = 接收方视角的
   **相对距离变化率**（close rate ≈ v_s+v_r ≈ 25-60 m/s vs 同向 ≈|v_s−v_r|）。
4. **审计预言（跑前注册）**：F5 接收方视角特征后，dataReplay/posMirroring 应
   大幅抬升（内容碰撞/close rate 是干净信号）；timeDelay 在合法特征下应保持
   ~0.5（构造性无信号），仅 artifact 特征（rcv_lag）可分 — 二者都如实报告。

**F5 特征（接收方视角，逐接收方文件计算后按 (sender_id,messageID) max 聚合）**：
r_posx/r_posy、dist_sr=‖s_pos−r_pos‖、close_rate=per-(receiver,sender) 对的
d(dist_sr)/dt、collision=与 ≤6s 内不同 sender 的内容精确碰撞（pos 串+spd+acl+hed）、
rcv_lag=receiver_pos 相对邻域插值的残差（**仅 timeDelay，标注 artifact**）。

## 9d. NextGen 官方 CaTch 基线（论文 Table II，highway_2，作对照）

官方参考检测器 CaTch 在 highway_2 的 F1：timeDelay 0.13、posMirroring 0.19、
dataReplay 0.28、dos 0.50、suddenStop 0.18、feignedBraking 0.07、accelMult 0.12、
Sybil 0.37、constPosOff 0.65、randPosOff 0.79、constSpeedOff 0.55、randSpeedOff 0.67、
zeroSpeed 0.77、reversedHeading 0.51、suddenConstSpeed 0.30（~待复核表值）。
→ 我们的 RF+F3/F4 已全面超过官方基线；难 3 攻击官方基线同样失败（0.13-0.28）。

## 9e. F5 结果（2026-10-13，results/sweep_hard3_f5.json，跑前已注册预言）

RF 200×3 种子，P1/P2，F34 vs F34+F5legit（+F5art 仅 timeDelay）：

- **dataReplay：P2 0.657 → 0.908（P1 0.713 → 0.911）**——内容碰撞是干净、接收方
  局部、因果的特征（0 假阳、48.5% 召回 + RF 联合其余特征补回）。难度从"难"修正
  为"容易"（0.91）。
- **positionMirroring：0.706 → 0.708**——F5 合法特征无增益，静态横向位移需地图，
  接收方视角构造性难。难度保持"难"（0.70）。
- **timeDelayAttack：F34 0.485-0.515（随机）→ +F5legit 0.474-0.508（仍随机）
  → +rcv_lag(artifact) 0.765-0.774**。合法特征下构造性不可检测；artifact 特征
  仅部分可分（稀疏流量下插值噪声）。二者如实报告。

**难度地图定稿 v2（P2，逐攻击最优特征=F3+F4+F5legit；timeDelay 另注 artifact）**：
平凡 6（dos/zeroSpeed/constSpeedOff/randSpeedOff/randPosOff/Sybil ≥0.99）；
容易 6（suddenConstSpeed/feignedBraking/suddenStop/accelMult/reversedHeading 0.95-0.98、
dataReplay 0.908）；中 1（constPosOff 0.83）；**难 2 = 真开放问题**（posMirroring
0.71 需地图/邻居共识；timeDelay 合法特征 0.51=构造性无信号，artifact 0.77 不计入
检测能力）。15 攻击中 13 个可消息级检测，2 个为构造性开放问题——与随机划分下
"15/15 解决（0.94-1.00）"形成审计对照。

## 9f. 序列级基线结果（2026-10-13，results/seq_baselines.json，P2 诚实协议）

（注：本节 LSTM/TF 数字为修正前旧值；P2 过滤缺陷由 §9o 修正并重跑，
修正后新值见 §9o 对照表，论文引用以新值为准。）

per-sender 序列（32 窗口、BCE、3 种子、z-score 标准化），F3+F4 同特征：

- **timeDelay：LSTM 0.481 / Transformer 0.488（随机）**——预言兑现：特征构造上
  不变，任何模型类都无信号。审计铁证链闭合：RF-F1 0.51 = RF-F34 0.51 = LSTM
  0.48 = Transformer 0.49，而 P3 随机划分 F1=0.997。
- **posMirroring：LSTM 0.610 / Transformer 0.713**——预言兑现（~0.7 上限）。
- **dataReplay：LSTM 0.768 / Transformer 0.911**——预言部分修正：per-sender
  序列本身携带足够信号（轨迹不连续），Transformer 0.911 ≈ RF+F5 碰撞 0.908，
  无需跨发送方特征。→ 结论改写为"dataReplay 可由序列模型或碰撞特征解决"；
  "难 3 需跨发送方推理"仅对 posMirroring/timeDelay 成立。

**难度地图定稿 v3**：13/15 可检测（dataReplay 0.908-0.911）；难 2 保持
（posMirroring 0.71、timeDelay 0.49）。序列模型对比表 = 论文基线表核心：
RF 0.657/0.706/0.515 → RF+F5 0.908/0.708/0.508 → LSTM 0.768/0.610/0.481 →
Transformer 0.911/0.713/0.488（dataReplay/posMirroring/timeDelay 序）。

## 9h. ST-GNN 基线（2026-10-13 跑前注册）

（注：本节 SAGE/MLP 数字为修正前旧值；P2 过滤缺陷由 §9o 修正并重跑，
修正后新值见 §9o 对照表，论文引用以新值为准。）

自建邻接（协议 9b）：节点=去重后消息；边=① 时间 ±2s 且距离 <300m 的空间边
② 同发送方连续消息的时间边。2 层 GraphSAGE（mean agg，hidden 64）→ 逐消息
logit，BCE 全批量，z-score 标准化，F34 节点特征（合法，不含 artifact）。
P2 协议、highway_2、难 2 + dataReplay、3 种子。

**预言**：posMirroring 或可小幅抬升（消息传递=邻居共识，静态横向位移 5-13m
对邻域聚合可见，预估 0.70→0.75-0.85）；timeDelay 应保持 ~0.5（节点特征无信号、
时间边构造上不变）；dataReplay 应与 Transformer 同水平（0.9 附近，序列不连续
信号在图上传导）。若 timeDelay 意外抬升须审计是否经 artifact 泄漏。

**9h 结果（2026-10-13，results/stgnn_baseline.json + gnn_tune_*.json）**：

GraphSAGE 60ep×3 种子：dataReplay **0.574** / posMirroring **0.667** / timeDelay
**0.476**。200ep 调参对照（seed 0，weight_decay 1e-4，MLP 无图对照）：
dataReplay MLP 0.641 / SAGE200 0.699；posMirroring 0.602 / 0.618；timeDelay
0.504 / 0.490。逐条对照预言：

- **timeDelay ✓（0.49，随机）**——第三次独立模型类确认构造性无信号（RF 0.51 =
  LSTM 0.48 = Transformer 0.49 = ST-GNN 0.49）。
- **posMirroring ✗**——预言 0.75-0.85（邻居共识）未兑现：SAGE 0.62-0.67 不超
  RF 0.706，200ep 反而低于 60ep（训练图过拟合）。静态横向位移对 mean-aggregation
  邻域不构成可聚合信号：位移与攻击者身份绑定，邻居（正常车）特征无对比基准，
  且 F34 已含 per-sender 差分。
- **dataReplay ✗**——预言 ~0.9 未兑现：SAGE200 0.699 仅平 RF 0.657 的水平，
  远低于 Transformer 0.911。时间边（同发送方连续消息）在 mean 聚合下未把
  轨迹不连续信号传导为节点级可分性（被正常邻居消息稀释）。

**结论（如实）**：自建邻接的 ST-GNN 对难攻击无增益，不构成新支柱；基线表新增
末行 ST-GNN 0.574-0.699 / 0.617-0.667 / 0.476-0.490（dataReplay/posMirroring/
timeDelay 序）。"图消息传递=邻居共识救 posMirroring"假设证伪；难度地图 v3 不变
（难 2 保持）。这与文献综述中"图检测器须评估 unseen 场景"的批评同向：图结构
本身不自动带来泛化信号。

## 9j. 跨数据集轴：VeReMi Extension（2026-10-13 预注册）

**数据**：Mendeley 10.17632/k62n4z9gdz.1，`data/extension_mixalldata_clean.csv`，
3,194,808 条 BSM（LuST 场景，24h，1Hz，CC-BY 4.0）。30 列：type(恒为 4)、
sendTime(秒)、sender、senderPseudo、messageID、class + pos/spd/acl/hed 各 3 维
clean+_n。**实测发现（预注册笔记）**：① `_n` 列全为 0 填充（垃圾），攻击值在
clean 列；② **无接收方字段、无 rcvTime** → 跨数据集特征只能对齐到发送方视角
F3-lite（dt/rate/consistency/move + dir_consistency）；③ senderPseudo=可观测身份
（假名随时间轮换，118,909 个），sender=真实 ID（24,663 个）——序列特征必须按
senderPseudo 分组（用 sender 会构成身份泄漏）；④ class 0=正常（1,900,539=59.49%，
与论文描述精确吻合），1-19=19 类攻击/故障。

**类名映射（经验签名裁决，网上论文表互相矛盾、与数据不符，不采信）**：
1 constPos（位置冻结）、2 constPosOffset（24m 偏移，差分正常）、3 randomPos
（步均 780m 跳跃）、4 randomPosOffset（37m）、5 constSpeed（**组内速度方差
0.000**）、6 constSpeedOffset（一致性 7.2）、7 randomSpeed（组内方差 11.4）、
8 randomSpeedOffset（一致性 6.8）、9 eventualStop（82% 零速、位置速度自洽）、
10/11 dataReplay 族（内容碰撞 0.16/0.32 + 位置跳跃，受害车远近两种配置）、
12 disruptive 族（57% 零速 + 固定点混排）、13 DoS（3× 洪泛、内容真实）、
14 DoSRandom（洪泛+生成式随机内容）、15 DoSDisruptive（洪泛+重放式随机内容）、
16 DoSDisruptiveSybil（洪泛+4× churn+启停）、17 dataReplaySybil（57× churn+
碰撞 0.72）、18 DoSRandomSybil（92× churn+洪泛+随机）、19 gridSybil（92× churn+
多点伪造+碰撞 0.43）。**delayedMessages 构造性不可定位**（与 NextGen timeDelay
同构：发送方视角无信号，候选 {6,10,11,13}——见预言 E4）。

**实验（RF 200×3 种子，消息级，特征 = F3-lite 对齐集）**：
- **E1 Extension 内部基线**：时间划分（sendTime 前 70%/后 30%，假名跨窗，≈P1），
  per-class 二分类（正常 vs 该类），正常类训练子采样 150k。→ Extension 版难度地图。
- **E2 NextGen highway_2 (P2) 训练 → Extension 测试**：重叠族对齐
  dosAttack→{13,14,15}、dataReplay→{10,11,17}、constPosOff→2、constSpeedOff→6、
  randPosOff→4、randSpeedOff→8、zeroSpeed→9、Sybil→{19,12}；另有 7 个 NextGen
  攻击在 Extension 无对应（unseen，负转移预期）。
- **E3 反向**：Extension 训练 → NextGen highway_2 测试（同对齐表）。
- **E4 预言**：timeDelay→delayedMessages 双向应保持 ~0.5（发送方视角构造性无信号
  跨数据集成立）；dataReplay 族在发送方 F3-lite 下应显著低于 NextGen 的碰撞特征
  水平（碰撞特征需接收方文件，Extension 无）——若 E2 dataReplay 族 ≈0.5 即预言
  兑现（NextGen 0.91 的来源是接收方内容碰撞，Extension 不可算）。
- 阈值/指标：AUROC+AUPRC，Extension 测试侧正常类子采样 150k（AUC 对负类子采样
  不变，协议注明）。

**9j 结果（2026-10-13，results/cross_dataset_extension.json）**：

- **E1 内部基线（AUROC）**：cls 2 constPosOffset = **0.498 机会水平**（AUPRC
  0.021），17 类 ≥0.86（13/14/15/18/19 ≈1.0；17:0.863、11:0.870、16:0.892、
  9:0.925），12 disruptive 0.788 中等可分离。**位置偏移攻击在自家数据集内部即
  构造性不可见**，与 NextGen 难度地图一致。
- **E2 NextGen→Extension**：DoS 族 1.000/1.000/1.000 无损转移 ✓。dataReplay 族
  **预言部分被证伪**：→cls 10 = 0.982（AUPRC 0.950）——重放内容在发送方视角留有
  运动学不连续痕迹，是真实信号而非 NextGen 捷径；→cls 11 0.842、→cls 17 0.560。
  constPosOff→2 = 0.493 机会 ✓；constSpeedOff 0.968、randPosOff 0.988、
  randSpeedOff 0.944、zeroSpeed 0.755、Sybil→19 0.969 / →12 0.478 机会。
  timeDelay→{6,10,11,13} = 0.73-0.82（AUPRC 仅 0.04-0.17，见 E4 讨论）。
- **E3 Extension→NextGen**：DoS 0.998/0.985/0.993 无损 ✓。dataReplay **0.573/
  0.579/0.568 机会** ✓（预言兑现方向）——**转移不对称**：E2 单向外溢 0.982 而
  E3 全机会，NextGen 重放比 Extension cls 10 更难的签名 Extension 训练学不到。
  constPosOff 0.520 机会 ✓；constSpeedOff 0.951、randPosOff 0.892（较内部 0.988
  退化）、randSpeedOff 0.953、zeroSpeed 0.721、Sybil→19 0.986 / →12 0.479 机会。
- **E4 timeDelay→全部 19 类：预言被证伪**。并非全机会：15/19 类 0.70-0.87 AUROC，
  但 AUPRC 0.04-0.18 ≈ 流行率（低精度）。解释（保守）：通用"时间异常先验"在
  失校准状态下外溢（NextGen 10Hz CAM vs Extension 1Hz BSM 速率尺度完全不同），
  非可转移的延迟签名。**4 类保持机会**恰好是 per-identity 时序正常的类：cls 2
  constPosOffset（0.489）+ 三个 Sybil 类 17/18/19（0.515/0.471/0.463）——假名级
  特征使跨身份洪泛不可见。
- **E4b 定位器**：Extension {6,10,11,13} → NextGen timeDelay **全部机会**
  （0.505/0.509/0.503/0.492）。delayedMessages 在候选集中不可定位——与 NextGen
  内部 timeDelay 的机会水平同构：**纯时序操纵在发送方特征中两个数据集、两个方向
  均无信号**。
- 结论：跨数据集轴确认难度地图的两个锚点（位置偏移双向不可见；DoS 双向无损），
  证伪一个预言（E4 非全机会，低精度外溢）和一个方向（E2 dataReplay cls 10
  0.982），并揭示转移不对称性（重放：E2 单向溢出 / E3 机会）。

## 9i. 事件级指标 + 告警负担（2026-10-13，results/event_metrics.json，P2）

（注：事件指标随 §9o 重跑更新；本节旧数字仅作历史记录，论文以重跑值为准。）

方法：RF 200×3（F34；dataReplay +F5legit），事件 = per-sender 连续攻击消息最大
游程（间隔 ≤2s），事件打分 = max(消息分数)，FPR 工作点 {10%, 1%, 0.1%}（阈值
取正常消息分位数），误报负担 = fp×正常消息数/(发送方数×2h)。

**关键 artifact（如实注册）**：事件 AUROC 全部 15 攻击 = 1.000，但 timeDelay
消息 AUROC 仅 0.485——**max 聚合在随机分数下结构性抬分**（k 个随机抽取的最大值
随机占优于单次抽取，≈7 条/事件时理论上限已 ~0.88）。因此事件 AUROC 不得作为
任何检测能力声称；FPR 工作点才是诚实指标。

- **timeDelay：事件 TPR 0.29 / 0.00 / 0.00 @ 10%/1%/0.1% 消息 FPR**——冒烟枪
  延伸至事件级：序列+事件聚合均不可救。
- dataReplay 0.98/0.91/0.79；posMirroring 0.89/0.79/0.58；dos/zeroSpeed
  0.1% FPR 下事件 TPR 1.00。
- 严格工作点下部分攻击事件 TPR 低（accelMult 0.36、feignedBraking 0.24、
  Sybil 0.06 @0.1% FPR）：短事件/分数分布重叠，如实报告。
- 误报负担：0.1% 消息 FPR 下多数攻击 ~0.02 次/车·小时。

## 9k. ΔF3 记忆化差距分析（2026-10-13，事后探索性，数字源自既有 JSON）

方法：对 15 攻击计算 ΔF3 = P3_F3 − P1_F3（sweep_attacks.json 既有字段，无新训练）。

结果：10/15 攻击 ΔF3 ≤ 0.035；恰好 5 攻击 ΔF3 ≥ 0.16（即 F3 下诚实特征仍不足
者）：timeDelay +0.513、revHead +0.409、posMirror +0.281、dataReplay +0.216、
constPosOff +0.161。P3-F3 下 15 攻击全部 ≥0.937（随机划分"全解"假象）。

解释（写入论文 fact (iv) + Table III 特征阶梯）：ΔF3 大 ⇔ 该攻击在诚实特征下
仍有缺口——泄漏差同时是特征充足性探针。F4 解决 revHead (0.971)、F5 解决
dataReplay (0.908)；其余 3 攻击（timeDelay/posMirror/constPosOff）在所有合法
特征层保持缺口，与 §9c 构造性结论一致。

论文新增表格均只用既有 JSON + 已核验来源：特征阶梯表（本节点）、
CaTch 全 15 攻击对照表（数值取自 NextGen 论文 Table II，/tmp/vnx_paper.pdf
已核验）、模型超参表（脚本源码常量）。无任何调参或新实验。

## 9l. 跨攻击迁移矩阵 + LOAO（2026-10-14 跑前注册；LOVO 支柱的落地执行）

**动机**：论文目前只引用 TVT 2025 的 tabula-rasa 数字论证"未见攻击≈随机"，
无自有测量。本实验补上：① 15×15 跨攻击迁移矩阵（训攻击 i、测攻击 j，画热图，
文献无人画过）② LOAO（训练=正常+14 攻击合并，测第 15 攻击）。

**方法**：15 子集各官方 Train/Test，去重与 F3/F4/F5 计算复用 §9c/9e 机制
（load_rows+dedup，F5legit 逐副本 max 聚合）。**统一特征集 F34+F5legit（29 维，
全攻击一致）**——矩阵测"签名迁移"而非特征配置差异；对角线与难度地图存在
系统差（地图 P2 行=各攻击最优特征；且矩阵给所有攻击都加了 F5），登记为预期
偏差而非错误。发送方不相交规则（对称 P2 的成对推广）：TR_i=子集 i 的 P2 训练
发送方集（tr2_i=tr_i 去 te_i 发送方）、TE_j=te2_j；评估集=(i,j) 时
test_ij=te2_j 去掉 TR_i；LOAO_j 的 test 去掉 ∪_{i≠j}TR_i。RF 200×3 种子，
种子概率平均。输出 results/cross_attack_matrix.json（含 n_pos/n_neg）。

**跑前预言**：
1. 对角线 ≈ 地图 P2 值（±0.05；dataReplay 带 F5 应落 0.9 附近）。
2. **LOAO**：平凡/运动学签名攻击（dos/zeroSpeed/各 speedOffset/sudden*）
   ≥0.9——签名可从其他攻击泛化学到；reversedHeading 靠 F4 dir_consistency
   物理特征，预期高（≥0.85）。**难 2 保持 ~0.5**（posMirroring/timeDelay 无
   共享签名，构造性结论的跨攻击再现）。constPosOff 0.7-0.85（与 randomPosOff
   签名重叠）。dataReplay 开放（F5 碰撞特征训练侧全零 → 只能靠 F34 序列痕迹
   迁移，预期 0.6-0.9 不确定区间，如实报告）。
3. 矩阵 off-diagonal 出现签名簇（位置类互迁、速度类互迁、突变类互迁）；
   不对称不做过分解读。
4. LOAO_j ≥ max_i M[i][j]（多数 j）——多源联合不劣于最优单源；若否如实报。

## 9m. 运行指标轴：检测时延 + 在线稳定性 + 推理成本（2026-10-14 跑前注册）

（注：运行指标随 §9o 重跑更新；本节旧数字仅作历史记录，论文以重跑值为准。）

**方法**：错误率配置同 §9i（P2，RF 200×3，F34；dataReplay +F5legit），重跑取
逐消息分数。① **检测时延**：阈值=正常（非事件）消息分数分位数 @ 消息 FPR
{1%, 0.1%}；事件检出=事件内按 sendTime 首条 ≥ 阈值；报检出率、检出事件中位
消息数/秒数。② **在线稳定性**：测试按 rcvTime 600s 分窗（12 窗），逐窗 AUROC
（<50 条或单类跳过并记录）与固定全局阈值下的逐窗 FPR，报均值/极差。③ **推理
成本**：RF predict_proba 计时（20k×29，3 次重复，n_jobs=1 与 -1 两档），报
μs/条。输出 results/operational_metrics.json。

**跑前预言**：平凡攻击（dos/zeroSpeed）检出中位 0-1 条消息 @1% FPR；难 2 检出
率 ≈ FPR 巧合水平（§9i 事件 TPR@1% 为 0.00/0.79 的延续，timeDelay 预期
≤0.05）@1%；高 AUROC 攻击逐窗 AUROC 极差 <0.05；稀疏攻击有效窗数 <12（记录
而非填补）。推理成本为常数级（特征为流式 O(1)/条）。**诚实边界**：②是单一
2h 模拟内的稳定性检查，非真实概念漂移证据，论文措辞不得拔高。

**结果（2026-10-14 跑，results/cross_attack_matrix.json）**：

- 预言 1 兑现：对角线 15 格全部落带内，|Δ| ≤0.018（最大 constPosOff
  0.845 vs 地图 0.827；dataReplay 0.908 落在预期 0.9 附近），矩阵池构造与
  统一特征集忠实于主实验。
- 预言 2 部分兑现、含一处重大证伪：
  - LOAO ≥0.95 共 7 攻击：zeroSpeed 0.997、randPosOff 0.998、
    randSpeedOff 0.996、constSpeedOff 0.994、DoS 0.983、suddConstSpd
    0.980、Sybil 0.950。
  - suddenStop 0.872（略低于 ≥0.9 预期）；constPosOff 0.817（落在
    0.7-0.85 区间 ✓）；dataReplay 0.590（略低于开放区间 0.6-0.9 下沿，
    如实报告，不再下调）。
  - 难 2：timeDelay 0.481 ✓（≈0.5）；posMirror 0.642（高于预期 0.5，
    弱可迁移，按注册不做过度解读）。
  - **reversedHeading 证伪**：预期 ≥0.85，实际 0.475（三种子
    [0.4922, 0.4495, 0.4823]，两种子低于随机）——dir_consistency 物理
    签名无法从其他攻击学到，且 LOAO 池化训练系统性压低其分数。
- 预言 3 兑现：off-diagonal 签名簇清晰——运动学不可信族（speedOffset
  家族 + zeroSpeed + DoS + randPosOff）30 个有序方向全部 ≥0.888 互迁；
  accelMult↔feignBrake 对称 0.907（纵向加速度共享签名，不外延）；
  posMirror 出迁移强（向静止族 0.96-0.99）、入迁移弱（最高 0.62 vs 自身
  0.708），与该攻击的身份绑定偏移构造一致。
- 预言 4 **证伪**：LOAO < 最优单源（i≠j）出现于 12/15 攻击；差距 >0.05
  者 5 个：feignBrake 0.750 vs 0.907（accelMult）、revHead 0.475 vs 0.616
  （randPosOff）、suddenStop 0.872 vs 0.947（zeroSpeed）、timeDelay 0.481
  vs 0.548（revHead）、dataReplay 0.590 vs 0.647（randSpeedOff）。池化
  14 源稀释而非覆盖攻击专属签名，论文如实报告。

论文落点：sec:unseen（fig4_matrix.png 热图 + tab:loao 表 + 签名簇与
证伪叙述）。

**结果（2026-10-14 跑，results/operational_metrics.json）**：n_events 与
msg_auroc 与 §9i 全部 15 攻击一致——配置复现核对通过。

- 平凡攻击 @1% FPR 中位 0-1 条 ✓（DoS 1.0 条、zeroSpeed 0.0 条、速度/
  位置偏移家族 0-1 条，检出率 0.98-1.00）。
- 难 2：timeDelay 检出 0.00（142 事件 @1% 与 @0.1%，≤0.05 预期 ✓，烟枪
  结论从排序延伸到运行）；posMirror 0.79 = §9i 事件 TPR@1% 0.79 的再现 ✓。
- 逐窗 AUROC 极差 <0.05：✓ 仅对 ≥0.98 近满分族（DoS 0.001、zeroSpeed
  0.000、constSpeedOff 0.001、randSpeedOff 0.003、randPosOff 0.006、
  Sybil 0.008）；**部分证伪**：中等难度/稀疏攻击超界——suddConstSpd
  0.347（0.652-0.999，59 事件）、constPosOff 0.283、posMirror 0.258、
  dataReplay 0.136、suddenStop 0.102、revHead 0.057。
- 稀疏攻击有效窗 <12：**证伪**——14/15 攻击 12/12 全有效，仅 Sybil
  12/13（跳 1 窗）；消息级样本量即使对 59 事件攻击也足够分窗。
- 推理成本常数级 ✓：2.0-8.9 μs/条（单核 n_jobs=1；随树深/特征数同量级
  浮动；即使 1000 条/s 也只占单核 <1%）。
- 诚实边界按要求保留：单次 2h 模拟内稳定性检查；逐窗最大 FPR 1.4-4.8%
  （标称 1%，最坏窗可达 5 倍），论文措辞为"阈值需周期复标或留余量"，
  未拔高为真实漂移证据。

论文落点：sec:operational（tab:operational 表 + 时延/稳定性/成本三段）。

## 9n. 深基线全攻击覆盖（2026-10-14 跑前注册；防御性完备）

**动机**：Table IV 只报 3 个难攻击的 LSTM/Transformer/ST-GNN；预审指出
"深模型在其余 12 攻击上是否退化"是显然会被问的问题。本轴把 LSTM/Transformer
补齐全 15 攻击（P2，F34，32 窗，与 §9f 完全同管线），论文用一句话报实测范围
（作为完备性声明，不改变难度地图结论）。

**方法**：seq_baselines.py 同款管线与超参（2 层 LSTM / 2 层 Transformer，
30 ep，3 种子，train 窗 32/步长 8、test 窗 32，sender-disjoint）跑 12 个剩余
攻击（dos/zeroSpeed/constSpeedOff/randSpeedOff/randPosOff/Sybil/suddConstSpd/
revHead/feignBrake/suddenStop/accelMult/constPosOff）；输出
results/seq_baselines_ext_seeds.json。scripts: seq_baselines_ext.py。

**跑前预言**：（a）平凡 6 攻击两模型均 ≥0.99；（b）易族 5 攻击两模型 ≥0.90
（revHead 依赖 F34 内 dir_consistency，预期延续 RF-F34 的 0.977 水平）；（c）
constPosOff（中档、per-identity 构造）开放区间 0.6–0.9；（d）**不预设"全员
≥0.98"**——低于者照写，论文句子按实测最低值措辞。

**诚实边界**：覆盖补测（reviewer-proofing），不改变难度地图；结果只用于
Table IV 伴随的一句话范围报告。

**9n 结果（2026-10-14，results/seq_baselines_ext_seeds.json，3 种子均值；
RF-P2F3 对照列来自 sweep_attacks.json）**：

| 攻击 | LSTM | TF | RF-P2F3 |
|---|---|---|---|
| dos | 0.513（种子 0.34/0.49/0.71） | 0.995 | 1.000 |
| zeroSpeed | 0.917 | 0.991 | 1.000 |
| constSpeedOff | 0.905 | 0.998 | 0.997 |
| randSpeedOff | 0.976 | 0.985 | 0.997 |
| randPosOff | 0.946 | 1.000 | 0.995 |
| Sybil | nan | nan | 0.992 |
| suddConstSpd | 0.712 | 0.793 | 0.979 |
| revHead | 0.474 | 0.460 | 0.573 |
| feignBrake | 0.761 | 0.918 | 0.970 |
| suddenStop | 0.797 | 0.890 | 0.960 |
| accelMult | 0.558 | 0.851 | 0.958 |
| constPosOff | 0.861 | 0.876 | 0.827 |

逐条对照预言：
- **(a) 平凡 6 攻两模型 ≥0.99 → 证伪**：仅 TF 在 dos/zeroSpeed/constSpeedOff
  达 0.99+；randSpeedOff 双低（0.976/0.985）；randPosOff LSTM 0.946；LSTM dos
  均值 0.513 且种子极不稳（0.34–0.71）。
- **(b) 易族 5 攻两模型 ≥0.90 → 部分证伪**：revHead 最重（0.474/0.460，两模型
  低于随机，远低于"延续 RF-F34 0.977"的预期）；suddConstSpd 双低（0.712/0.793）；
  accelMult LSTM 0.558；TF 侧 feignBrake 0.918 / suddenStop 0.890 / accelMult
  0.851 达成，LSTM 侧均未达成。
- **(c) constPosOff 0.6–0.9 → 兑现**：0.861/0.876。
- **(d) 不预设 ≥0.98，最低值照写 → 遵守**：revHead 0.460 已如实写入论文
  §VII 新增 (iv) 覆盖段。
- **Sybil = 结构性不可表示**：Test 中 16,782 个幽灵发送方（每人 1–2 条消息），
  低于 make_windows 的 8 条/发送方门槛与 seq_len=32，正窗口为 0，AUROC 数学上
  无定义（sklearn 单类异常）。非缺陷，是 per-sender 序列公式的表示边界。
- **总览**：TF ≥0.98 于 5 攻；两模型 ≥0.86 于 5 个 offset/flood 攻；序列模型
  超过同协议 RF-P2F3 仅 **4/22 可计算格**（TF constSpeedOff 0.998 vs 0.997、
  TF randPosOff 1.000 vs 0.995、LSTM+TF constPosOff 0.861/0.876 vs 0.827）
  → "特征边界、非容量边界"论点获独立证据。

论文落点：§VII 预言段新增 (iv) Full-attack coverage 段。

## 9o. P2 协议缺陷修正与全链重跑（2026-10-14）

**发现**：审读 §3 划分构造时发现 5 个脚本的 P2 实现存在 in-place 过滤缺陷
（第 2 行用已过滤的 tr 去滤 te，成为 no-op）：

```python
tr = tr[~tr.sender_id.isin(set(te.sender_id))]
te = te[~te.sender_id.isin(set(tr.sender_id))]   # no-op：tr 已被覆盖
```

实际效果：te 保留了大量"在 tr 中出现过的发送方"的消息，违反论文 §III-B 的
P2 定义（"messages of senders seen in the other split are removed from both
sides"）。量化：Train∩Test 重叠 180 个发送方，修正前 Test 中约 36.4% 消息
来自 seen 发送方（Sybil 达 19.4%）。

**修正**（双向过滤均基于原始帧的快照模式）：

```python
tr_senders = set(tr.sender_id)
tr = tr[~tr.sender_id.isin(set(te.sender_id))]
te = te[~te.sender_id.isin(tr_senders)]
```

**受影响脚本（5+1）**：seq_baselines.py、seq_baselines_ext.py、
stgnn_baseline.py、event_metrics.py、operational_metrics.py；gnn_tune.py
（2026-10-14 恢复重建时同步修正）。**不受影响**：sweep_attacks.py
（本就是 tr2/te2 快照引用实现）、sweep_f4.py、sweep_hard3_f5.py、
cross_attack_matrix.py、各 transfer 脚本、pilot/hard3_signatures——重跑确认
其 JSON 与备份逐位一致。

**重跑与备份**：旧结果全部备份至 results/pre_p2fix/（22 个 JSON）；修复后
全链重跑 5 脚本 + gnn_tune，全部 exit 0；均值文件（seq_baselines.json /
stgnn_baseline.json）由 seeds 文件重新同步。

**old → new 对照（论文引用面）**：

| 出处 | 项 | 旧 | 新 |
|---|---|---|---|
| §9f | timeDelay LSTM/TF | .481/.488 | .445/.470 |
| §9f | posMirror LSTM/TF | .610/.713 | .616/.796 |
| §9f | dataReplay LSTM/TF | .768/.911 | .771/.900 |
| §9f | 铁证链（P2 修正后） | RF .51 = LSTM .48 = TF .49 | RF-F1 .49 / RF-F34 .51 = LSTM .445 = TF .470 |
| §9h | SAGE60 dataReplay/posMirror/timeDelay | .574/.667/.476 | .631/.667/.472 |
| §9h | gnn_tune MLP/SAGE200 三攻 | .641/.699; .602/.618; .504/.490 | .647/.699; .600/.619; .499/.483 |
| §9i | timeDelay msg_auroc | .4855 | .516 |
| §9i | dataReplay/posMirror 事件召回@0.1% | .79/.58 | .86/.70 |
| §9i | timeDelay 事件数 / 10% 召回 | 142 / .29 | 85 / .31 |
| §9m | 事件数、检出率、窗口域、max FPR、成本 | 见 pre_p2fix/ | 见 results/operational_metrics.json（论文 tab:operational 已同步） |

**论文同步面**：Table IV 全部行、§VI/VII 文本（0.445–0.515 链、Transformer
0.900、0.667→0.619）、tab:events 全表、tab:operational 全表、§VII 事件/运行
段落、fig3 左面板（带域 0.45–0.51）与图注、附录 tab:hyper MLP 行——均已按
新结果更新。未受影响面（sweep 系、转移系、LOAO、cross_attack_matrix、fig2、
fig4、tab:catch、tab:ladder）经比对确认不变。

## 10. 禁令

- 论文数字手写/来自屏幕输出 → 一律走 results/*.json。
- P3 随机划分的数字不得作为任何"检测性能"声称。
- 难攻击的 0.5 级别结果不得通过调特征/调模型硬刷高后隐瞒过程（任何提升须以
  F1→F3→F4 消融形式如实呈现）。
- 不得因"身份泄漏次要"而隐瞒 H2 曾作为主假设的事实（负结果写进论文）。
