# GW-YOLO v2 自动化基线运行报告（2026-07-19）

## 结论

自动化链路已经在远端 RTX 4090 D 上完整运行，并通过初始质量门槛。最终模型是 YOLO26m-seg：无同源组泄漏验证集 `mask mAP50=0.7471`，锁定测试集 `mask mAP50=0.7651`、`mask mAP50-95=0.4047`。这是一条可信的图像分割基线，但还不是能够宣称超过搜索方法或 AMPLFI/DINGO 的论文结果。

## 运行边界与可复现信息

- 原项目 `/root/GW-YOLO` 全程只读，没有覆盖权重或历史运行。
- 新代码：`/root/GW-YOLO-v2`。
- 新产物：`/root/GW-YOLO-v2-artifacts`。
- 环境：Python 3.11.15、PyTorch 2.13.0+cu130、Ultralytics 8.4.101、RTX 4090 D。
- 配置哈希：`ff200f8279b40073`。
- 数据 manifest SHA256：`d5391f428be2ec8d21dceb8e390ea1e1ec9c7f4392a9d931a9f21b7570298b0d`。
- 最终 checkpoint SHA256：`5014fbd3ef816231082504d15e213cc961b86e022c4e4ac3bd6ef5d3f78fd9ba`。
- 随机种子：`20260719`。
- 完整机器可读状态：`/root/GW-YOLO-v2-artifacts/pipeline_state.json`。

## 数据审计和重划分

源数据共有 414 张图、300 个物理来源组、894 个实例，其中 chirp 360、noise 534、空标注 7。旧 train/validation 之间发现 31 个同源组交叉。

新划分按物理来源组完成，交叉组为零：

| split | 图像 | 来源组 | chirp 实例 | noise 实例 | 空标注 |
|---|---:|---:|---:|---:|---:|
| train | 295 | 251 | 247 | 362 | 5 |
| validation | 60 | 25 | 56 | 86 | 1 |
| locked test | 59 | 24 | 57 | 86 | 1 |

因此，本报告不能与旧随机划分或论文中的 0.95 mAP 作无条件横向比较；它测量的是更严格的同源组外泛化。

## 自动候选训练结果

| 模型 | 训练 | val box mAP50 | val mask mAP50 | val mask mAP50-95 | 结论 |
|---|---:|---:|---:|---:|---|
| YOLO26n-seg | 100 epochs | 0.7017 | 0.6971 | 0.4090 | 未过 0.72 门槛 |
| YOLO26m-seg | 118 epochs，epoch 88 最优 | 0.7332 | 0.7471 | 0.4278 | 通过门槛 |

YOLO26m 的锁定测试结果：

| 指标 | all | chirp | noise |
|---|---:|---:|---:|
| box mAP50 | 0.7487 | 0.772 | 0.725 |
| mask mAP50 | 0.7651 | 0.806 | 0.725 |
| mask precision | 0.8582 | 0.873 | 0.843 |
| mask recall | 0.7465 | 0.807 | 0.686 |
| mask mAP50-95 | 0.4047 | 0.394 | 0.415 |

推理速度约为 11.6 ms/图（不含完整应变预处理、Q-transform 和网络传输）。

## GWTC-4 目录诊断

在置信度 0.25 下，85 张目录图中 43 张含 chirp 预测，目录图命中率为 50.6%，Wilson 95% 区间为 40.2%–61.0%；共产生 58 个检测，58 个实例均保留分割 mask。

这个数值低于旧模型的 47/85，说明无泄漏分割基线仍存在明显的 O4 域偏移。它不是搜索 recall：样本没有连续背景分母，主要是单探测器视图，也没有固定 FAR。高 SNR 漏检进一步说明单一 Q 图和单 IFO 表示不是可靠的网络事件统计量。

## 本轮发现并修复的工程问题

训练框架默认使用综合 fitness 保存 `best.pt`，而论文主指标是 mask mAP50；YOLO26n 的训练日志曾达到约 0.741，但默认 `best.pt` 复验只有 0.697。代码现已增加 `best_target.pt`：每个 epoch 按配置中的主指标独立保存 checkpoint，后续运行优先使用该权重。该修复已增加单元测试。

当前本地测试为 27/27 通过，远端模块也已编译和导入验证。

## 数据工厂增量（同日）

数据不足结论已经转化为可运行代码，而不再只是规划：新增确定性的物理 `SceneRecipe`、
waveform/injection、glitch 和 GPS 四轴泄漏审计、三 IFO × 三 Q 数值张量以及 chirp/glitch
双掩膜。远端完整 pilot 生成 104 个场景，67/67 个含 chirp 场景和 67/67 个含 glitch
场景均有非空目标掩膜，跨 split 物理 ID 重叠为零，产物位于
`/root/GW-YOLO-v2-artifacts/data_factory_pilot`。

服务器仅余约 14 GB，而 full-debug pilot 已占 42 MB；线性外推 20 万场景约 82 GB。因此
研究配置已改为 16 万训练、1 万验证、3 万锁定测试的 `recipe_only` 模式，训练在线确定性
生成，只有冻结评估集采用 float16 分片。真实 O4 接口使用 GWOSC API v2/HDF5，并默认硬锁
O4b。详细设计和命令见 `docs/DATA_FACTORY.md`。

20 万条配方已经在远端完整生成并审计，manifest 为 93,093,283 bytes，SHA256 为
`ac36fc3732fc8583b1903b78cccb50048b8f2680d36d1c483ee576569c5b9505`，四个物理轴的跨 split
重叠为零。这个结果验证了目标规模的 provenance/I/O 路径，但不等价于已经拥有 20 万条真实
波形和真实噪声锚点。

## 下一阶段执行布局

### P0：把图像基线做成可发表的负责任基线（1–2 周）

1. 用新 checkpoint 规则完成 5 个种子；报告均值、标准差和配对 bootstrap 区间。
2. 固定同一组划分，完成 YOLOv8/YOLO26、n/m、旧增强/物理保守增强消融。
3. 做 confidence calibration、PR 曲线、错误图库和质量/质量比/SNR 分层。
4. 主结论只限于“时频实例分割”；不使用 GWTC 目录命中率替代搜索效率。

退出门槛：无泄漏、5 seeds、结果差异的置信区间、完整 manifest 和 checkpoint 哈希。

### P1：数值多 Q、多时间尺度数据（2–6 周）

1. 从应变生成数值张量而非渲染截图，保留 PSD、GPS、IFO、DQ 和 provenance ID。
2. 使用 1/4/16/64 s 窗和多 Q 平面；训练 O1–O3，使用 O4a 开发和校准，锁定 O4b/GWTC-5。
3. 在时域组合 CBC 注入和真实 glitch，再重新变换；禁止把 mosaic/mixup 当作物理叠加。
4. 建立 BBH/NSBH/BNS、质量比、自旋、进动、偏心、透镜/重叠事件和单 IFO 缺失分层。

退出门槛：数值输入在 O3→O4 域迁移上显著优于截图基线。

### P2：多探测器场景模型（5–10 周）

主模型采用共享 IFO 编码器和跨探测器注意力，输入 H1/L1/V1 多 Q 张量与有效性 mask；输出 chirp/glitch 实例、到达时间差/coherence、OOD 和校准置信度。保留早融合模型作为低延迟基线。

关键消融：单 Q→多 Q、单 IFO→多 IFO、无 coherence→有 coherence、分类→实例 mask、无 OOD→有 OOD。

退出门槛：固定验证 FAR 下，多 IFO 对单 IFO 的注入恢复提升具有不跨零的配对置信区间。

### P3：搜索级评价和去噪优势（6–14 周）

代码已经提供 `gwyolo search-eval`，用验证背景校准阈值并冻结到测试，输出 FAR、IFAR、Poisson 零计数上限、加权效率和 recovered `<VT>`。下一步接入真实 trigger 流、软件注入和 time slide。

必须报告：

- 连续背景和等效背景时长；
- efficiency–SNR、FAR/IFAR、敏感距离和 `<VT>`；
- clean 与 glitch-overlap 两个主层；
- mask gating/inpainting 前后的恢复 SNR、误 veto 率和偏差；
- 若声称 IFAR 100 年量级，零背景计数时至少需要约 230 年等效背景才能使 90% FAR 上限达到 0.01/年。

论文主门槛建议预注册为：固定 FAR 下 overlap recovery 提升至少 10 个百分点，或 `<VT>` 提升 5%–10%，同时 clean injection 损失低于 1 个百分点。

### P4：与 AMPLFI/DINGO 的公平联合对表（10–16 周）

AMPLFI 和 DINGO 是参数估计系统，不是截图检测器。共同事件集应运行四条链：原始应变→AMPLFI、原始应变→DINGO、GW-YOLO 去噪→AMPLFI、GW-YOLO 去噪→DINGO。

共同指标为端到端延迟、PP/SBC coverage、Jensen–Shannon/Wasserstein 距离、参数偏差、90% 天区面积/体积和 glitch-overlap 失败率。GW-YOLO 的独立优势应落在可解释 chirp/glitch mask、失效检测和去噪稳健性，而不是宣称 mAP 高于 posterior estimator。

## 论文落点

- **PRD**：以固定 FAR 的 `<VT>`、重叠恢复和 PE 偏差降低为主结果。
- **JCAP**：需要进一步展示对选择效应、并合率、标准汽笛或重叠事件天体物理推断的实际影响。
- **ApJS**：发布 O4 多 IFO/multi-Q/mask benchmark、软件、权重和大规模验证表，最适合作为首篇资源论文。

当前最接近的是 ApJS 数据/基准路线；达到 PRD 还缺连续背景、注入和固定 FAR 的显著提升。仅凭本次 mAP 不能承诺录用，也不能宣称已与 AMPLFI/DINGO 持平。
