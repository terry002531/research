# SWIR OPD（1000–1300 nm）可行研究点调研

> 日期：2026-09-24
> 适用前提：少碰化学；重点在器件物理、成像/系统、模拟；有光电测试能力；器件由系里同事或合作方提供；阵列/读出平台还在搭；会自己写代码（Python/MATLAB）；目标是 6–12 个月内在中高档期刊（AFM / LPR / AOM / ACS Photonics / Adv. Sci. 等）发一篇。
>
> 说明：这次检索只能看到搜索引擎给出的摘要，出版社全文页面被当前网络环境拦截了。所以"是否已被做过"的判断只是初筛，动手之前请用第 6 节的关键词在 Web of Science / Google Scholar 上再核一遍。

---

## 1. 一句话结论

> **更新**：按实际仪器清单重新评估后，主线改成 **B + C**，D 作为扩展，A 降为可选并先做仿真。详见 [02-feasibility-with-instruments.md](02-feasibility-with-instruments.md)。下面是最初的排序，保留作记录。

**首选：A「考虑器件物理的 SWIR 单像素成像」**。它不需要等阵列，能直接用上你的测试能力和编程能力，故事也清楚：窄带隙带来陷阱，陷阱让响应变慢，慢响应又限制了成像速度，我们把这个过程建模并在重建时反演回去。
**同时做：B「从 D\* 到成像 SNR」**。它和 A 共用大部分测试数据，也正好给你们在搭的读出平台提供设计依据，可以写成第二篇，或者作为 A 的器件物理部分。
**阵列到位后接着做：D「共享有源层 SWIR 阵列的像素串扰」**。

---

## 2. 1000–1300 nm OPD 现状速览

| 维度 | 代表性数据（近期报道） | 来源 |
|---|---|---|
| 暗电流 | PDCBT-DTO2F 聚合物体系：−2 V 下 15.6 nA cm⁻²，−5 V 下 58.1 nA cm⁻²；1100 nm 处 D\* = 1.04×10¹² Jones（−2 V）；带偏压存放 500 h 后 D\* 仍 >10¹¹ Jones | [Liu, Adv. Mater. 2026](https://advanced.onlinelibrary.wiley.com/doi/10.1002/adma.73895) |
| 响应度 / D\* | 超窄带隙小分子受体，响应覆盖 0.3–1.3 µm，1110 nm 处 0.53 A/W、1.71×10¹³ Jones（零偏压）。**要注意 D\* 的噪声是怎么取的** | [Sci. Adv.](https://www.science.org/doi/10.1126/sciadv.adm9631) |
| 响应 >1200 nm 的 NFA | BZIC-2F / BZIC-2Cl 吸收边超过 1200 nm；1010 nm 处 0.49 A/W、1.30×10¹² Jones，并演示了 SWIR 成像 | [Qiao, Adv. Mater. 2026](https://advanced.onlinelibrary.wiley.com/doi/10.1002/adma.202522600) |
| 成像器 | P3HT:BZIC-2F 刮涂在 a-Si TFT 背板上，面积 32×32 mm，响应 400–1200 nm，在 1100 nm 下演示了穿硅成像；另有 256×256 OPD-TFT 成像器，1140 nm 处 0.36 A/W、LDR 82 dB（这组数据出自右边两篇中的哪一篇，搜索摘要没写清，需核对原文） | [Qiao 2026](https://advanced.onlinelibrary.wiley.com/doi/10.1002/adma.202522600)、[Chen, Angew.](https://onlinelibrary.wiley.com/doi/10.1002/anie.8903659) |
| 窄带 | 微腔耦合 CT 态（PTB7-Th:BTPSeV-4F），1208 nm 共振、FWHM 17 nm，通过腔长可在 1050–1350 nm 连续调节 | [Zhu, LPR 2026](https://onlinelibrary.wiley.com/doi/10.1002/lpor.71238) |
| 速度 | SWIR OPD 的上升/下降时间一般在 µs 量级（例如有三元体系报道 3.82/2.91 µs）；弱光下响应会明显变慢，原因是深陷阱 | [Joseph, Small 2026](https://onlinelibrary.wiley.com/doi/10.1002/smll.73959)、[Nat. Commun. 2022](https://www.nature.com/articles/s41467-022-31367-4)、[Saggar, AOM 2024](https://advanced.onlinelibrary.wiley.com/doi/10.1002/adom.202302916) |
| 暗电流的物理极限 | 反向暗电流主要来自陷阱辅助产生（[Kublitski, Nat. Commun. 2021](https://www.nature.com/articles/s41467-020-20856-z)）；饱和暗电流受中隙陷阱态限制，由此给出 D\* 的上限（[Sandberg, Nat. Photon. 2023](https://www.nature.com/articles/s41566-023-01173-5)） | — |
| 综述 | 材料与器件路线图 | [Chem. Sci. 2025 roadmap](https://pubs.rsc.org/sc/article/16/46/21705/911147/High-performance-organic-semiconductor-near) |

**核心矛盾**：带隙越窄，非辐射复合和陷阱越多，暗电流越高、响应越慢、1/f 噪声越大。目前大部分工作在材料层面解决这个问题，这恰好是你要避开的。**器件物理怎样影响系统性能**（噪声进入成像 SNR、慢响应限制帧率、串扰降低 MTF）目前讨论得很少，这是你的切入口。

---

## 3. 已经比较拥挤或已被做过的方向（建议避开，或只作为对照）

| 方向 | 已有工作 |
|---|---|
| 设计新的窄带隙受体/给体 | 路线最拥挤，而且是化学主导 |
| 微腔 + CT 态做 1050–1350 nm 可调窄带 | [Zhu, LPR 2026](https://onlinelibrary.wiley.com/doi/10.1002/lpor.71238)；早期原理见 [Siegmund, Nat. Commun. 2017](https://www.nature.com/articles/ncomms15421) |
| 用 OPD 做单像素压缩成像（到 1.3 µm） | [Yao…Ng, ACS AEM 2019](https://pubs.acs.org/doi/abs/10.1021/acsaelm.9b00009)。**所以 A 的新意不能落在"用 OPD 做单像素成像"本身，要落在器件动力学建模上** |
| 在 TFT 上做穿硅成像 | [Qiao, Adv. Mater. 2026](https://advanced.onlinelibrary.wiley.com/doi/10.1002/adma.202522600) |
| 偏压可调 OPD 计算光谱仪 | 400–1000 nm 已有 [Nat. Electron. 2024](https://www.nature.com/articles/s41928-024-01199-9) 和 [叠层单像素光谱仪 2025](https://www.sciencedirect.com/science/article/pii/S2666998625001796)；延伸到 SWIR 需要复杂的叠层工艺 |
| 角度无关的窄带（极化激元） | 965 nm 已有 [arXiv 2412.06741](https://arxiv.org/pdf/2412.06741) |
| PPG / 血压 / 光通信应用演示 | [Zeng, Adv. Mater. 2025](https://advanced.onlinelibrary.wiley.com/doi/10.1002/adma.202509923) |
| NIR-I / NIR-II 双模窄带 | [Nat. Commun. 2025](https://www.nature.com/articles/s41467-025-62394-6) |
| 通过界面层或三元策略"降低暗电流" | 文章非常多，只做这个很难进中高档期刊 |

---

## 4. 候选研究点

### A. 考虑器件物理的 SWIR 单像素成像 ⭐ 首选

**科学问题**
窄带隙 SWIR OPD 的陷阱密度高，瞬态响应有 µs–ms 级的慢拖尾，而且响应时间随光强变化（弱光更慢，见 [Nat. Commun. 2022](https://www.nature.com/articles/s41467-022-31367-4)、[Saggar AOM 2024](https://advanced.onlinelibrary.wiley.com/doi/10.1002/adom.202302916)）。
单像素成像（SPI）每个 DMD 图案对应一个采样。如果图案周期接近或短于器件的响应时间，前面图案的信号会漏进后面的采样（相当于通信里的码间串扰），图像就会模糊、出伪影。常规做法是把图案刷新率降到器件带宽以下，代价是帧率很低。

**核心想法**
1. 把 OPD 的时间响应当作前向模型的一部分：y = h(t, I) ⊛ (P·x) + n，其中 h 是实测的、随光强变化的冲激响应。
2. 重建时反演这个响应，可以用正则化最小二乘（TV / 稀疏先验），或者 unrolled 网络。
3. 目标是在图案刷新率达到器件 f₋₃dB 的 3–10 倍时仍能得到可用的图像，也就是用算法弥补器件本身的慢。
4. 附带结论：互补 Hadamard 图案做差分可以抑制 1/f 噪声和漂移。据此可以定量说明，决定成像 SNR 的是**图案调制频率处的噪声谱密度**，而不是 1 Hz 下的 D\*。这部分和 B 衔接。

**为什么现在还是空白**
OPD 做 SPI 已经有人做过（2019，到 1.3 µm），但检索里没看到有人把 OPD 的**陷阱动力学和强度相关的响应**写进 SPI 前向模型。一般光学领域有"慢探测器时域成像"的工作（例如 [arXiv 2305.13346](https://arxiv.org/html/2305.13346)），但出发点和器件不同。**这一点需要你再精确检索确认。**

**需要的东西**
- 器件：同事提供 2–4 种 1000–1300 nm 响应、**陷阱密度不同**的 OPD。同一个共混体系做不同退火或不同添加剂就行，这样"陷阱多 → 慢 → 可以被算法补偿"的因果链最清楚。
- 光学：NIR 窗口的 DMD（TI 有 NIR 版本的 DLP 评估板，刷新率和窗口透过范围要按具体型号确认），1050 / 1200 / 1300 nm 的 LED 或激光，NIR 镀膜透镜。
- 电学：TIA 加示波器或 DAQ，用于测阶跃/冲激响应和 SPI 采样。
- 代码：前向模型、重建算法、响应拟合（多指数或陷阱速率方程）。这部分都是你擅长的。

**6–9 个月计划**
| 月 | 内容 |
|---|---|
| 1–2 | 搭 SPI 光路；拿到器件后测 EQE、J–V、噪声谱 |
| 2–4 | 在不同光强和偏压下测阶跃/方波响应；建立 h(t, I) 模型（经验多指数模型，加上一个简单的陷阱俘获-释放速率方程模型） |
| 4–6 | 仿真：前向模型加噪声，得到重建质量与图案刷新率/f₋₃dB 的关系；把算法调好 |
| 6–8 | 实验：分辨率靶、穿硅检测（Si 在 >1100 nm 基本透明）、隐藏物体等；和"朴素重建"及 InGaAs 参考对比 |
| 8–9 | 写作。图示：器件动力学 → 前向模型 → 速度-质量曲线 → 应用演示 |

**风险和对策**
- DMD 在 SWIR 的透过率或刷新率不够：可以改用机械编码盘，或用 LED 阵列做结构光（这也是一个卖点）。
- 器件太快，看不出问题：在弱光、低偏压下工作，或者换陷阱更多的样品。反过来说，器件越慢，这个方法的价值越大。
- 审稿人质疑"只是算法"：用器件物理讲清楚 h(t, I) 从哪里来（陷阱能级、密度和响应拖尾对应起来），这是 OPD 期刊关心的。

**目标期刊**：Laser Photonics Rev. / ACS Photonics / Adv. Opt. Mater. / Adv. Sci.；保底可以投 Photonics Research / Opt. Express。

---

### B. 从 D\* 到成像 SNR：SWIR OPD 在积分型读出下的真实性能

**科学问题**
OPD 文献普遍用 D\* 排座次，很多 D\* 是用暗电流按散粒噪声估算的，会明显高估（见 [Kublitski 2021](https://www.nature.com/articles/s41467-020-20856-z) 以及 [噪声表征陷阱的讨论](https://pubmed.ncbi.nlm.nih.gov/40557714/)）。
而成像器工作在**电荷积分模式**（积分 1–30 ms，CDS 采样）。真正起作用的是：积分传递函数加权后的噪声，包括 1/f 噪声、暗电流占用的满阱容量、kTC 噪声和读出噪声（见 [imec TF-PPD, Nat. Electron. 2023](https://www.nature.com/articles/s41928-023-01016-9)）。
**D\* 高的 SWIR OPD，放进像素以后 SNR 不一定更好。**

**做法**
1. 拿 3–6 种 SWIR OPD（不同体系、界面层、厚度）。
2. 测 EQE、暗 J–V（有条件的话做变温）、1 Hz–100 kHz 噪声谱（多个偏压）、瞬态响应。
3. 写一个像素级模型：3T 或 CTIA，加 CDS 传递函数，积分实测噪声谱，得到输入等效噪声电子数、SNR 随光子通量的曲线、动态范围和噪声等效辐照度（NEI）随积分时间的变化。这些是 [EMVA 1288](https://www.emva.org/wp-content/uploads/EMVA1288Linear_4.0Release.pdf) 风格的相机指标，Release 4 已支持 SWIR。
4. 用你们在搭的分立积分电路（电荷放大器 / CTIA）实测时域噪声随积分时间的变化，验证模型。
5. 输出结论：散粒噪声 D\*、实测 D\* 和成像 SNR 三种排名为什么不一致；在 1000–1300 nm，哪个器件参数（1/f 拐点、暗电流、电容、EQE）最影响成像。给出设计规则。

**优点**：几乎不需要新器件；和 A 共用数据；直接服务于你们的读出平台。
**风险**：可能被当成"测试方法"类文章。对策是一定要有实测积分电路的验证，并给出可操作的设计规则或者开源工具。
**目标期刊**：Adv. Opt. Mater. / ACS Photonics / Adv. Electron. Mater.；保底 IEEE TED。

---

### C. 1000–1300 nm OPD 暗电流诊断：离中隙陷阱极限还有多远？

**科学问题**
[Sandberg 2023](https://www.nature.com/articles/s41566-023-01173-5) 给出了"中隙陷阱决定暗电流下限"的普适趋势。但对 1.0–1.2 eV 带隙的 SWIR 体系，实际器件的暗电流到底是**本征的陷阱产生**、**接触注入**，还是**分流/针孔**？每种机制离极限还有几个数量级？

**做法**：变温暗 J–V、偏压相关的噪声谱，有条件再加灵敏 EQE / FTPS。用你写的漂移-扩散代码拟合，把暗电流分解成几个来源，给出每个器件的"剩余优化空间"。

**评价**：新意中等，因为"低暗电流"和"陷阱"类文章很多。更适合**作为 A 或 B 的器件物理章节**。单独成文的话，需要覆盖多个体系，并开源诊断代码。
**需要确认**：你们有没有变温台（Peltier 或低温恒温器），能不能测灵敏 EQE。

---

### D. 共享有源层的 SWIR OPD 阵列：像素串扰与 MTF（阵列到位后做）

**科学问题**
在 TFT/CMOS 上做 OPD 时，有源层和界面层通常不图形化。SWIR 波段吸收弱，需要**更厚的有源层**，这会加重载流子横向扩散；PEDOT:PSS、ZnO 等界面层的横向电导也会带来电学串扰。OLED 里这个问题研究得很多（例如 [Sci. Rep. 2023](https://www.nature.com/articles/s41598-023-41033-4)），OPD 方面只看到指纹传感的零星工作（[叉指电极光电流扩散](https://www.researchgate.net/publication/362711458_Photocurrent_diffusion_of_organic_photodetector_between_pixels_for_fingerprint_sensor_by_using_interdigitated_electrodes)），**针对 SWIR 厚膜的系统研究和设计图谱基本空白**。

**做法**
1. 请同事在衬底上做不同间距的底电极测试结构。不需要完整阵列。
2. 用聚焦的 1064 / 1310 nm 光斑扫描，测线扩散函数，换算成 MTF。
3. 用 2D 漂移-扩散模型（自己写，或者简化成电阻网络加扩散）解释有源层厚度、界面层电导、偏压和像素间距之间的权衡。
4. 输出"EQE–MTF–像素间距"设计图谱。

**目标期刊**：Adv. Opt. Mater. / ACS Photonics / IEEE EDL/TED。
**时间**：拿到测试结构后大约 6 个月。适合作为第二篇。

---

### E.（备选，风险高）SWIR 计算光谱仪

把偏压可调叠层 OPD 光谱仪从 400–1000 nm 扩展到 900–1300 nm；也可以做"腔长渐变的微腔 OPD 阵列 + 重建算法"。算法部分你擅长，但器件工艺复杂，依赖合作方，不建议作为第一篇。

---

## 5. 各方向对比

| | 新意 | 与你的条件匹配 | 6–12 月出文概率 | 依赖外部 | 建议 |
|---|---|---|---|---|---|
| A 考虑器件物理的 SPI | 高 | 很高（测试 + 编程 + 系统） | 中高 | DMD、SWIR 光源、器件 | **首选** |
| B 从 D\* 到成像 SNR | 中高 | 很高（测试 + 读出 + 编程） | 高 | 器件、简单积分电路 | **并行** |
| C 暗电流诊断 | 中 | 高 | 高（中档期刊） | 变温台 | 作为 A/B 的一部分 |
| D 阵列串扰 | 中高 | 高（等阵列） | 中 | 图形化衬底 | 第二篇 |
| E SWIR 光谱仪 | 中高 | 中 | 中低 | 叠层工艺 | 备选 |

---

## 6. 查新检索用的关键词（建议在 Web of Science / Google Scholar 上跑一遍）

- A：`single-pixel imaging` AND (`photodetector response time` OR `detector bandwidth` OR `temporal response` OR `deconvolution`) ；`single-pixel` AND `organic photodetector` AND (`infrared` OR `SWIR`)；`ghost imaging` AND `slow detector`
- B：`organic photodiode` AND (`integration time` OR `charge integration`) AND `noise`；`organic image sensor` AND (`EMVA` OR `photon transfer` OR `input-referred noise`)
- C：`organic photodetector` AND `dark current` AND `temperature dependent` AND (`shortwave infrared` OR `SWIR` OR `1100 nm`)
- D：`organic photodetector` AND (`crosstalk` OR `MTF` OR `line spread`) AND (`array` OR `imager`)

---

## 7. 不管做哪个方向，测试都要守的规矩（审稿人常卡的地方）

1. **D\* 必须用实测噪声谱**（说明频率和带宽），不能只用暗电流按散粒噪声估算。
2. 1000–1300 nm 的 EQE 要用 InGaAs 或 Ge 参比标定，并说明光斑和器件面积（加光阑）。
3. LDR 用中性密度片覆盖到弱光端，写明噪声底。
4. 响应速度要测多个光强，尤其是弱光。器件在弱光下会变慢，这恰好也是 A 的切入点。
5. 所有指标都注明偏压；如果有稳定性数据，给出带偏压工作下的暗电流漂移。

---

## 8. 还需要你确认的问题

1. 能不能借到或买到 **NIR 窗口的 DMD**（或其他空间光调制器），以及 1050–1300 nm 的 LED/激光？这决定 A 能不能马上开始。
2. 同事/合作方现在手上有哪些**已经在 1000–1300 nm 有响应的体系**（例如 PTB7-Th:IEICO-4F、PTB7-Th:CO1-4Cl、BZIC 系列、PDCBT-DTO2F 类聚合物……）？能不能提供**同一体系、陷阱密度不同**的样品？
3. 你们的测试平台有没有**噪声谱**（低噪声前放 + 频谱仪 / 锁相）、**变温台**、**灵敏 EQE**？
4. 读出平台打算走 TFT、CMOS 还是分立电路？大概什么时候能用？
