# GW-YOLO 项目审计、论文对表与 GWTC-4/5 研究路线

审计日期：2026-07-19（UTC）
审计范围：本地 `模型报告.pdf`、远端 `/root/GW-YOLO`（只读检查）、GitHub 仓库、arXiv:2508.17399、GWOSC/LVK 的 GWTC-4.0 与 GWTC-5.0 官方资料。

## 1. 结论先行

当前项目已经完成了一次可运行的实例分割训练，也把模型应用到了 85 张 GWTC-4.0 事件 Q-scan 上；但它目前仍是“图像实验原型”，不是可以和 GWTC 搜索管线公平比较的引力波搜索系统。

最紧迫的问题不是继续换更大的 YOLO 或多训几个 epoch，而是：

1. **评估有不一致与算术错误。** PDF 报告中的 Recall 和 F1 计算有误，所谓验证集与独立集“完全一致”不成立；PDF 指标也不对应服务器上当前曲线。
2. **验证切分可能泄漏。** 远端 414 张图像就是论文公开的完整 Zenodo 数据集；当前 331/83 切分中，43.4% 的验证图像与训练集共享疑似底层 waveform/glitch 标识。必须按物理源、注入、glitch ID 和 GPS 时间分组重切。
3. **当前 GWTC-4.0 命中并不理想。** 85 个官方事件名全部可在 GWOSC API 中匹配，但模型只对 47 个输出 chirp。这个 55.3% 只能称为“单张候选图的朴素命中率”，不能称为搜索效率；其中还有 4 个 network SNR > 20 的事件未输出 chirp。
4. **没有能支持“超过其他方法”的指标。** 目前没有连续背景、时间滑移、FAR/IFAR、`p_astro`、灵敏体积时间 `<VT>`、注入恢复曲线或低延迟测量。mAP、F1 和已知目录事件上的分类准确率不能与 PyCBC、GstLAL、MBTA、cWB 等正式搜索管线直接排名。
5. **最值得深入的方向是混合系统。** 让 GW-YOLO 做多探测器、多 Q 尺度的 morphology/glitch 识别和掩膜，再用其改进候选重排序、数据质量 veto 或 deglitch 后的匹配滤波。与单独用 YOLO“发现所有事件”相比，这条路线更可能在固定 FAR 下获得显著 `<VT>` 增益，也更接近论文尚未完成的下游工作。

## 2. 机器和工程现状

### 2.1 服务器

| 项目 | 观察结果 |
|---|---|
| 主机 | Ubuntu 20.04.5，`autodl-container-74204da04f-3722bacf` |
| GPU | NVIDIA GeForce RTX 4090 D，报告显存 49140 MiB |
| 项目路径 | `/root/GW-YOLO` |
| Git 状态 | 该目录不是 Git 仓库 |
| 磁盘 | 30 GB，总占用约 17 GB，剩余约 14 GB |
| GPU 任务 | 审计时无 GPU 计算进程 |
| 环境状态 | `gwyolo` 环境为 Python 3.11.15、Ultralytics 8.4.101；审计时另有两个已持续约 75 分钟的 Conda/Pip 安装进程，环境处于被并发修改状态 |

这台机器足以做中等规模消融和推理，但当前环境不可复现，且安装进程长时间并发修改 PyTorch。应先停止“边装边跑”的工作方式，建立锁定环境；本次审计没有终止进程，也没有修改远端任何文件。

### 2.2 代码和产物

远端顶层只有 6 个 Python 文件、数据、预训练权重和一次训练产物，没有 README、requirements/lockfile、测试、数据生成代码或实验配置版本控制。

关键文件状态：

- `trian.py`（文件名拼写错误）从 `yolo26m-seg.pt` 开始训练，而不是 PDF 所称的 `yolo26n-set`。
- 当前 `best.pt` 的 SHA-256 是 `e5e0b03ed0106cbac9b7e4c904a5abe2a17a65ca19e007884fd96cc773fba884`，`last.pt` 是 `f2705e91a95c039f575cfed712c24fa987590915530def089a5e951b3c791e27`；后续报告应始终用该类哈希绑定具体权重。
- `gw_data.yaml` 使用 Windows 风格的 `\root\GW-YOLO` 和反斜杠路径；`args.yaml` 的 `save_dir` 也是 `F:\python\...`。这说明训练产物很可能先在 Windows 生成后上传，当前 Linux 项目不能直接视为可复现实验。
- `predict.py` 虽然加载 segmentation 权重，却只读取 `result.boxes`，完全丢弃 `result.masks`。
- `predict.py` 每张图只保留最高置信度的一个 chirp，天然不支持论文强调的 multi-transient/multi-instance 信号场景。
- 输出标签是 `class x_center y_center width height confidence`，不是 segmentation polygon。
- `analysis.py`、`color.py` 使用硬编码 Windows 路径和 `cv2.imshow()`，无法作为无头服务器上的批处理管线。
- `analysis.py` 依靠 Hough 直线从渲染后的坐标轴反推时间/频率；它假设检测到足够且顺序稳定的轴线，并把一个边界位置打印为“发生时间”，鲁棒性和物理定义都不足。
- 项目没有从 strain 生成 Q-transform 的代码，没有注入生成、背景采样、事件级聚合、多探测器对齐或指标计算代码。

### 2.3 GitHub

[lanhung/gw-yolo-update](https://github.com/lanhung/gw-yolo-update) 在审计时是公开但空的仓库：GitHub API `size=0`，`git ls-remote` 没有任何 ref。当前服务器代码、数据配置、训练日志和权重都没有进入版本控制。这是复现、协作和论文可信度的首要工程阻塞项。

## 3. 数据审计

### 3.1 当前数据规模

| 切分 | 图像 | 标签文件 | 空标签 | chirp 实例 | noise 实例 |
|---|---:|---:|---:|---:|---:|
| Train | 331 | 331 | 6 | 282 | 415 |
| Validation | 83 | 83 | 1 | 78 | 119 |
| 合计 | 414 | 414 | 7 | 360 | 534 |

所有标签行都满足当前 YOLO polygon 的基本格式，坐标在 `[0,1]` 内；图像与标签一一配对。

### 3.2 与论文公开数据的关系

论文发布的 [GW-YOLO Training Dataset](https://zenodo.org/records/17211276) 声明共有 414 张图像。下载其 `Train.zip` 后：

- ZIP 的 MD5 与 Zenodo 元数据一致：`36df84387abd4f2d261f537d6d97e3e1`；
- 远端 train 与 val 的图像并集，和 ZIP 中 414 个图像文件名完全一致；
- 标签文件名也完全一致。

因此，远端不是一套新的扩展训练数据，而是把论文公开的 414 张完整数据重新分成 331/83，即 80/20。论文正文写的是 80/10/10，当前项目没有可追溯的独立 test 切分。

### 3.3 分组泄漏风险

文件名中存在疑似表示底层 waveform、glitch 或注入来源的 32 位标识。例如同一个标识会出现在不同 GPS 时间的多张图片中。审计结果：

- Train 中有 183 个此类源标识；Validation 中有 63 个；两者交集 31 个；
- 55 张训练图、36 张验证图含有跨 split 的共同源标识；
- 即 36/83 = **43.4% 的验证图**可能与训练图共享底层形态来源。

这不是像素级完全重复；两边没有发现 SHA-256 完全相同的图像。但如果共同标识确实代表相同 waveform、glitch 模板或其变体，那么随机图片切分会显著高估泛化能力。正式实验必须按以下键进行 group split：

`waveform/injection_id + glitch_id + GPS segment + detector + observing run`

同一个底层信号或 glitch 的所有增强版本必须只属于一个 split。

### 3.4 输入表示的域偏移

训练图是带坐标轴、文字、色条和固定 colormap 的 640×640 截图；模型可能学习渲染风格而不只是物理形态。抽查显示：

- 训练图多为 3 秒窗口、色条上限可到 45；
- 当前 GWTC-4.0 图多为约 4 秒窗口、色条上限常为 25；
- 坐标轴范围、字体、图像边距和动态范围均可能变化。

这些 nuisance feature 足以造成明显 domain shift。长期方案应直接输入数值 Q-map/小波张量，不把坐标轴、标题、色条渲染进模型输入。

## 4. 训练产物的真实指标

### 4.1 训练配置

- 模型：`yolo26m-seg.pt`，约 54.5 MB；
- 计划 300 epochs，实际记录 257 epochs；
- batch 8，imgsz 640，seed 0，deterministic true；
- `cls=1.5`；
- `mosaic=0.5, mixup=0.2, translate=0.1, scale=0.3, erasing=0.3`；
- 禁止旋转、上下/左右翻转和 HSV 变换；
- patience 50。

按 segmentation fitness（Box 与 Mask mAP50-95 的组合）计算，最佳点是 **epoch 207**；训练到 257 后早停，正好相隔 50 epochs。

Epoch 207 的 aggregate 指标：

| 指标 | Box | Mask |
|---|---:|---:|
| Precision | 0.781 | 0.846 |
| Recall | 0.786 | 0.729 |
| mAP50 | 0.771 | 0.773 |
| mAP50-95 | 0.465 | 0.430 |

当前 `best.pt` 对应 PR/F1 图给出的类别指标：

| 分支 | chirp mAP50 | noise mAP50 | all mAP50 | 最大总体 F1 / 阈值 |
|---|---:|---:|---:|---:|
| Box | 0.800 | 0.742 | 0.771 | 0.78 @ 0.252 |
| Mask | 0.827 | 0.721 | 0.774 | 0.78 @ 0.284 |

验证混淆矩阵的对象级计数为：chirp 正确 65、chirp 漏检 13、noise 正确 91、noise 漏检 27、noise→chirp 错分 1，另有 13 个 chirp 背景误报和 28 个 noise 背景误报。

### 4.2 过拟合判断

PDF 说“未观察到明显过拟合”，这个判断过强。当前 `results.png` 中：

- train segmentation loss 持续下降；
- validation segmentation loss 约在 epoch 50 附近达到最低，此后明显回升；
- mAP50 很早进入平台期，后续主要是 mAP50-95 缓慢提升且波动。

这至少是 segmentation 分支的泛化 gap，需要通过真正独立、group-aware 的验证集确认。仅凭训练 loss 平滑不能排除过拟合。

### 4.3 图像增强的物理问题

关闭翻转、旋转和颜色扰动是合理改动；但 mosaic 和 image-space mixup 并不等价于在 strain 时域中叠加两个物理信号。它们会拼接坐标轴或线性混合已经做过非线性归一化/着色的图像。`scale` 也可能无意改变 chirp 的时频斜率和质量信息。

下一版应在时域完成：信号注入、glitch 叠加、时间偏移、幅度/SNR 调节、PSD 漂移和 calibration 扰动，然后重新计算 Q-transform。图像空间只保留不改变物理语义的轻量增强。

## 5. 对《模型报告.pdf》的核查

### 5.1 算术错误

报告给出 TP=61、FN=22、FP=7、TN=35。由此应得到：

- Recall = `61/83 = 0.73494`，即 **73.5%**，不是 73.9%；
- Precision = `61/68 = 89.7%`；
- Accuracy = `96/125 = 76.8%`；
- F1 = `2TP/(2TP+FP+FN) = 122/151 = 80.8%`，不是 81.2%。

Wilson 95% 区间也很宽：

- Recall 63.1%–81.8%；
- Precision 80.2%–94.9%；
- Accuracy 68.7%–83.3%。

所以 125 张样本不足以支撑小幅度改进结论，更不能据此说明可部署。

### 5.2 “完全一致”结论不成立

报告选择 epoch 237 的 Box Recall 0.73862，与误算后的 0.739 比较并称“完全一致”。但：

- 独立集精确 Recall 是 0.73494；
- 绝对差为约 0.0037，并非完全相同；
- 最佳 checkpoint 是 epoch 207，其 Box Recall 为 0.78641；
- 从 257 个 epoch 中事后挑一个数值接近的点，不能证明泛化；
- 两个集合的类别口径也不同：CSV 是两类对象级 aggregate，独立集是图像级 chirp presence。

### 5.3 PDF 与当前产物不一致

| 项目 | PDF | 服务器当前曲线 |
|---|---:|---:|
| 模型描述 | `yolo26n-set` | `yolo26m-seg.pt` |
| Box chirp/noise/all mAP50 | .779/.772/.775 | .800/.742/.771 |
| Mask chirp/noise/all mAP50 | .810/.718/.764 | .827/.721/.774 |
| Box 最优阈值 | 约 .300–.500 | .252 |
| Mask 最优阈值 | .386 | .284 |

这意味着 PDF 评估的是另一 checkpoint、另一运行或旧曲线。没有模型哈希、数据 manifest、代码 commit 和评估命令，无法确定其来源。报告所述“独立 125 张测试集”也不在当前服务器：`imgs/gw_4.0` 只有 85 张，且当前输出是 47 个 chirp，而不是 61 个。

## 6. 与 arXiv:2508.17399 逐项对表

论文：[GW-YOLO: Multi-transient segmentation in LIGO using computer vision](https://arxiv.org/abs/2508.17399)，v2，2025-10-09。

| 维度 | 论文 | 当前远端项目 | 判断 |
|---|---|---|---|
| 基础模型 | YOLOv8 instance segmentation | YOLO26m segmentation | 架构升级，但没有同 split 消融，不能归因性能变化 |
| 训练数据 | O3 GravitySpy glitch + PyCBC BBH/BNS + O3 injections | 与公开 414 张数据文件名完全相同 | 当前没有扩展训练样本 |
| 数据切分 | 声称 80/10/10 | 331/83，即 80/20，无 test | 不可直接复现论文 test |
| 论文 validation | mAP50 .947，P .890，R .900 | 当前 mask mAP50 .774 | 低约 17.3 个百分点，需先解释/复现 |
| 论文 test | mAP50 .953，P .915，R .920 | 当前无可追溯 test | 无法比较 |
| 推理集合 | BBH、BNS、BBH+glitch、BNS+glitch | 85 张 GWTC-4.0 Q-scan | 当前增加真实 O4a 目录应用，但缺对照和真值 |
| 推理规模 | BBH 每套约 1400；BNS 每套约 1100，总计约 5000 | 85 | 统计功效明显不足 |
| SNR 分层 | BBH 6–48；BNS 12–48；约每 bin 100 | 无 | 无法画效率曲线 |
| 阈值 | 0.48（论文 F1 最优） | 0.25；当前曲线最优约 .252/.284 | 阈值未校准 |
| overlap 结果 | 50% efficiency：BBH SNR≈15，BNS SNR≈30 | 无相同实验 | 无法判断是否改进 |
| 掩膜输出 | pixel mask 是核心产出 | 批量预测丢弃 mask，只存 box | 功能倒退 |
| multi-transient | 支持多个实例 | 每图仅保留一个 chirp | 与目标冲突 |
| 探测器 | 论文 inference 使用 Livingston | 82 张 H1、3 张 L1，仍为单探测器 | 有跨站点样本，但没有融合 |
| Q-transform | 论文说明 Q-map，并提出 adaptive Q future work | 没有 Q-transform 生成代码 | 尚未落实论文最重要改进方向 |
| 下游 | 背景抑制、事件验证、deglitch 留作 future work | 未实现 | 最有价值的研究空白仍在 |

当前确实存在的正向改动包括：换用更大的新模型、提高分类损失权重、关闭明显破坏 chirp 方向性的翻转/旋转/HSV、开始在真实 GWTC-4.0 事件图上推理，并同时记录 Box/Mask 的 mAP50-95。但这些改动没有经过可归因消融，且当前 validation 指标远低于论文报告，因此还不能称为对论文方法的性能改进。

## 7. GWTC-4.0 实测审计

当前 `imgs/gw_4.0` 有 85 张图，事件名全部能在 [GWTC-4.0 官方 API/事件列表](https://gwosc.org/eventapi/html/GWTC-4.0/) 中匹配。目录完整发布包含 128 个新的 O4a 候选；当前 85 张只是其中一个高显著性子集，而不是全部目录。

在 `conf=0.25` 下：

- chirp 对象：47；
- noise 对象：20；
- 有任意输出的图：57；
- 完全无输出：28；
- 未输出 chirp 的事件图：38；
- 每图最多一个 chirp 是代码硬限制。

若暂时把这 85 张都视为“应命中 chirp”的正例，则朴素命中率为 `47/85 = 55.3%`。但它不能作为正式 recall，原因包括：只取一个探测器视图、没有逐事件可见性真值、没有负样本、没有事件级多探测器融合。

按 GWOSC 提供的 network matched-filter SNR 粗分层：

| Network SNR | 输出 chirp / 总数 | 朴素命中率 |
|---|---:|---:|
| 8–10 | 13/29 | 44.8% |
| 10–12 | 18/31 | 58.1% |
| 12–15 | 10/14 | 71.4% |
| 15–20 | 3/4 | 75.0% |
| ≥20 | 3/7 | 42.9% |

高 SNR 段反而下降，漏掉的 network SNR > 20 事件包括 GW240105_151143、GW231028_153006、GW231206_233901 和 GW231123_135430。这不一定单独归咎于模型：network SNR 不等于所选 H1 图的单站 SNR，而当前 85 张中 82 张只用了 H1。它恰好说明“任取单站 Q-scan”不是可靠事件表示，必须做 H1/L1/V1 融合。

## 8. “在 GWTC-4/5 超过其他方法”应如何定义

GWTC 不是普通分类 benchmark，而是多个搜索算法对连续 strain 数据进行搜索、背景估计、事件验证和参数推断后的目录。GWTC-4.0 使用/汇总了 cWB、GstLAL、MBTA、PyCBC、SPIIR 等管线；官方方法用模拟注入测量灵敏度，并以 FAR、`p_astro` 和 `<VT>` 评估搜索能力，详见 [GWTC-4.0 Methods](https://dcc.ligo.org/public/0195/P2400300/011/GWTC-4.0_methods_v11.pdf)。

[GWTC-5.0](https://gwosc.org/GWTC-5.0/) 已于 2026-05-26 发布，覆盖 O4b；其新增 161 个 `p_astro ≥ 0.5` 且通过事件验证的候选，累计目录达到 390。官方还发布了 O4a+O4b search sensitivity estimates。因此今天已经可以做真正的 O4a/O4b 跨运行评估，而不是只在少量目录截图上算准确率。

建议把“超过”拆成三个层次：

### A. 超过原 GW-YOLO 论文（近期、可实现）

在完全相同、无泄漏的 O3 注入协议上：

- 降低 glitch-overlap 下 50% efficiency 的 SNR：BBH 从约 15 降到 ≤12，BNS 从约 30 降到 ≤22；
- 同时报告 95% bootstrap 区间；
- mask IoU/mAP、事件召回与背景误报都不能恶化；
- 用固定阈值或预先锁定的 calibration，不能每个测试集重新找最优阈值。

### B. 超过单一 ML/Q-scan 基线（中期、最适合发论文）

在 O4a 开发集、O4b锁定测试集上，与 YOLOv8、当前 YOLO26m、单 Q、单探测器等基线比较：

- 固定 FAR 下的 injection efficiency；
- glitch-overlap 子集的恢复率；
- 事件验证/噪声框 mask IoU；
- 推理延迟和校准误差。

### C. 对正式搜索管线产生增益（困难但最有价值）

不要只把 YOLO 与 matched filtering 二选一。把 GW-YOLO 输出作为额外 ranking/veto/deglitch 特征，证明在相同搜索背景上：

- 固定 FAR（如 1/yr 或 1/100 yr）时，`<VT>` 相对最佳单管线提高至少 5%–10%，且置信区间不跨 0；或
- glitch-overlap 注入的效率提高至少 10 个百分点，同时普通干净注入损失 <1 个百分点；或
- 在不降低高可信事件召回的情况下，显著降低高 ranking-statistic 的 glitch 背景。

只有 C 层可以严谨地说对 GWTC 搜索方法有竞争性；只在已知目录事件上达到高 recall 不等价于发现能力。

## 9. 推荐技术路线

### 9.1 P0：先重建可信 benchmark

1. 将远端代码真正提交到 GitHub，加入 README、license、环境锁、配置、数据 manifest、模型 SHA-256 和一键复现实验命令。
2. 从公开 strain 重新生成数据，不依赖截图。保存数值 Q-map、PSD、GPS、IFO、窗口、Q/frequency 参数、注入参数和 glitch ID。
3. Group split：O1–O3 训练，O4a 开发/校准，O4b 一次性锁定测试；同一 GPS 段、waveform、glitch 或其增强版本不可跨 split。
4. 建立三类评估集：
   - 正例：真实目录事件与软件注入；
   - 难例：chirp+真实 glitch，按时间/频率重叠度分层；
   - 背景：连续 analysis-ready 数据、目录事件移除后的 off-source 数据、时间滑移及公开 retraction/glitch。
5. 指标改为 efficiency-vs-SNR、FAR/IFAR、`<VT>`、calibration、mask IoU 和 latency；mAP 只作为辅助指标。

背景时长必须足够。若零误报，Poisson 90% 上限约为 `2.3/T`；要证明 FAR < 1/yr，等效背景至少需要约 2.3 年；要证明 <1/100 yr，需要约 230 年，可通过多探测器 time slide 获得。

### 9.2 P1：修正输入表示

1. **多 Q、多时窗、多频段。** 建议至少使用 1/4/16/64 秒窗口，以及覆盖高质量 BBH、低质量 BBH/NSBH、长时 BNS 的频段；把多个 Q plane 当作 channel 或 token，而不是只取最大能量 Q plane。
2. **自适应 Q。** 原论文明确指出最大能量 Q 可能偏向 glitch、压制低 SNR chirp。这应成为第一项核心消融。
3. **取消渲染依赖。** 输入 whitened log-energy 数组和 validity mask；坐标轴、文字、色条不进入网络。
4. **多探测器融合。** H1/L1/V1 以 GPS 对齐，保留每站 PSD、可用性和 antenna response；用 cross-attention 或 late fusion，并加入允许的传播时延约束。
5. **辅助通道 veto。** O4 已公开部分 auxiliary channel 数据，可增加一个轻量环境/仪器分支来识别非天体共因，数据入口见 [O4 Auxiliary Channel Data Release](https://gwosc.org/O4/auxiliary/)。

### 9.3 P1：修正数据和标签

1. 在时域把 waveform 注入真实 O3/O4 噪声，再做 Q-transform；禁止把 image mixup/mosaic 当成主要物理增强。
2. 覆盖 BBH、NSBH、BNS，并系统采样质量比、自旋、进动、高阶模、偏心、距离、倾角和天空位置。GWTC-5.0 新增事件按当前结果均与 BBH 一致，因此短时、高质量 BBH 和 glitch 混淆应优先。
3. 对 overlap 控制 `Δt`、频率交叠比例、glitch SNR、signal SNR 和 glitch 类别，形成二维/三维效率图，而不是只按 SNR 一维统计。
4. chirp mask 从无噪声注入的时频能量支持区自动生成；glitch mask用 BayesWave/Omicron/人工复核组合，记录标注不确定性。
5. 把 `noise` 更名为 `glitch/transient_noise`，与无目标的 stationary background 明确区分。
6. 做 hard-negative mining：blip、scattered light、extremely loud、tomte、线噪声，以及高排名但被事件验证否决的触发器。

### 9.4 P2：模型和决策层

1. 先做公平基线：YOLOv8n/m-seg、YOLO26n/m-seg 使用完全相同数据、split、seed 和输入；至少 5 个 seed，报告均值和区间。
2. 与 U-Net/Mask2Former 类分割模型、时频 transformer、原始时序 1D/2D 混合模型做同预算消融。模型大小不是首要变量。
3. 输出保持多实例，不要只保留一个 chirp；支持多信号、信号+glitch 和多个 glitch。
4. 将 frame-level box/mask 转成 event-level score：跨滑窗合并、跨探测器融合、time-frequency 一致性、到达时延一致性。
5. 在独立 calibration 集上做温度缩放或 isotonic calibration；阈值由目标 FAR 决定，而不是由 test F1 决定。
6. 使用不确定性拒识/OOD 检测，防止新 glitch 类别被高置信度误判为 chirp。

### 9.5 P2：最有论文价值的混合方案

建议主线题目为：**Multi-detector, multi-Q GW-YOLO for mask-informed ranking and deglitching in O4 data**。

流程：

`连续 strain → 多尺度数值 Q-map → 多探测器 GW-YOLO → chirp/glitch masks → 候选重排序或 mask-informed deglitch → PyCBC/GstLAL-like statistic → FAR/<VT>`

两个直接可测的贡献：

1. **背景抑制：** mask/coherence score 作为 ranking feature，降低 glitch 尾部，固定 FAR 提升 `<VT>`。
2. **重叠事件恢复：** 根据 glitch mask 做门控、inpainting 或 BayesWave 引导，重新匹配滤波；比较恢复 SNR、参数偏差和误删真实信号率。

这正好补上原论文只画在 workflow 中、尚未量化的 background reduction 和 deglitching。

## 10. 建议实验矩阵

| ID | 输入/模型 | 目的 | 必报指标 |
|---|---|---|---|
| B0 | 论文 YOLOv8 + 原始协议 | 复现论文 | mAP、P/R、SNR efficiency |
| B1 | YOLO26m + 同协议 | 隔离架构贡献 | 同 B0，加速度/显存 |
| A1 | 数值 Q-map，取消坐标轴/色条 | 测渲染域偏移 | O3→O4 泛化、ECE |
| A2 | Multi-Q + multi-window | 提升低 SNR/BNS | 分类型 efficiency |
| A3 | H1/L1/V1 fusion | 修复单站漏检 | network efficiency、FAR |
| A4 | + auxiliary veto | 压低 glitch 背景 | FAR 尾部、误 veto 率 |
| A5 | + mask-informed deglitch/rerank | 最终系统 | `<VT>`、恢复 SNR、PE bias、latency |

所有实验使用相同 injection set 和 background，并做成对 bootstrap；不允许为每个模型单独挑选有利测试子集。

## 11. 分阶段执行计划

### 第 1–2 周：让结果可相信

- 完成 Git、环境锁、数据 manifest 和配置系统；
- 修复 Linux 路径、模块化 train/predict/evaluate；
- 删除“只保留一个 chirp”和“丢弃 mask”的逻辑；
- 完成 group-aware 重切和数据审计；
- 复现 B0/B1，解释为什么当前 mAP50 只有约 .774 而论文报告 .947/.953。

Go/No-Go：若论文结果不能在无泄漏 split 上复现，则先公开说明差异，不继续用旧 mAP 做宣传。

### 第 3–6 周：建立 O4 基准

- 数值 Q-map 生成器；
- O1–O3 train、O4a validation、O4b locked test；
- 连续背景与 time-slide；
- SNR、质量、glitch 类别、overlap、IFO 分层；
- 完成 A1/A2/A3。

Go/No-Go：至少在相同 FAR 下显著优于单 Q、单探测器 B1；否则先优化数据表示，不急于换网络。

### 第 7–12 周：形成显著结果

- 加入 auxiliary veto、事件级校准和 OOD；
- 接入 mask-informed reranking/deglitch；
- 在 O4b locked test 上一次性评估；
- 报告 `<VT>`、FAR、效率、置信区间、延迟和失败案例。

目标：glitch-overlap recovery 提高 ≥10 个百分点，或固定 FAR 下 `<VT>` 提高 ≥5%–10%，且对干净注入效率损失 <1 个百分点。

## 12. 当前最值得立即做的五件事

1. 冻结远端安装，导出可复现环境；把服务器代码提交到现在为空的 GitHub 仓库。
2. 保存当前 checkpoint、`args.yaml`、`results.csv` 和数据列表的哈希，给 PDF 补充模型/数据版本；重新生成一份算术正确的报告。
3. 按底层标识和 GPS group 重切 414 张数据，重新跑 YOLOv8 与 YOLO26 的公平基线。
4. 为 85 张 GWTC-4.0 图补齐 H1/L1 两站输入、单站 SNR/可见性和人工/物理真值，先解释 38 个 chirp miss，特别是 4 个 network SNR > 20 的案例。
5. 不再把目录截图 recall 当“搜索性能”；立即建设 O4 continuous-background + injections 的 FAR/`<VT>` harness。

## 13. 证据边界

- 本次只读访问远端，没有启动/停止训练、修改环境、上传代码或改动远端文件。
- 权重没有反序列化执行；只检查了文件格式、哈希、训练日志和已有图表。
- 85 张事件的 55.3% 是基于现有输出的审计统计，不是正式 pipeline recall。
- 跨 split 的共同 32 位标识是强烈泄漏信号，但其确切语义仍需原始数据生成 metadata 证明；因此本报告称其为“潜在/疑似底层来源泄漏”。
- GWTC-5.0、O4b 和官方方法状态以 2026-07-19 可访问的 GWOSC/LVK 页面为准。

## 14. 主要公开来源

- [GW-YOLO arXiv v2](https://arxiv.org/abs/2508.17399)
- [GW-YOLO Training Dataset / Zenodo](https://zenodo.org/records/17211276)
- [GWTC-4.0 Data Release](https://gwosc.org/GWTC-4.0/)
- [GWTC-4.0 Event List/API](https://gwosc.org/eventapi/html/GWTC-4.0/)
- [GWTC-4.0 Methods](https://dcc.ligo.org/public/0195/P2400300/011/GWTC-4.0_methods_v11.pdf)
- [GWTC-5.0 Data Release](https://gwosc.org/GWTC-5.0/)
- [O4b Open Data Release](https://gwosc.org/news/o4b-open-data-release/)
- [O4 Auxiliary Channel Data Release](https://gwosc.org/O4/auxiliary/)
- [当前 GitHub 仓库](https://github.com/lanhung/gw-yolo-update)
