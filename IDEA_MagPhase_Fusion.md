# MPFADet：面向宽带时频检测的幅度–相位融合

> 目标以本数据集为准：在现成幅度时频图上做检测/分类/时频定位，相位图从 I/Q 创新生成并与之融合。
> CMFADet 与 Phase-Aware 只提供原则，不作为必须照搬的网络骨架。

---

## 0. 设计立场

本任务不是航拍 RGB–IR 检测，也不是 PSK/QAM 调制识别竞赛。

**目标值（按优先级）：**

1. **框准**：每个信号的时间–频率矩形（DateTime × Freq）
2. **类准**：9 类调制分得开（尤其 2FSK/4FSK、8-Tone/16-Tone、GMSK/FM、PSK）
3. **可训练**：4000 张 few-shot，模型不能太重
4. **可解释**：相位图上能看出调制事件，而不是黑盒通道拼接

CMFADet 只借三条原则：双流中层融合优于早拼接；两路提取器可以异构；分类与回归可以吃不同特征。  
Phase-Aware 只借三条原则：raw phase / Re-Im 喂 CNN 无效；相位必须变成视觉图案；时频框又瘦又长，定位比自然图像难。  
**模块名、yaml、SFEM、IR-AFAB、CIF、ATAH 一律不当默认答案。**

---

## 1. 任务与数据（一切设计的约束）

### 1.1 任务定义

输入一对对齐的时频图 `(幅度 PNG, 相位图)`，输出若干水平框：

```
(cls, t_start, t_end, f_low, f_high)
```

对应标注字段 `Content, DateTimeStart, DateTimeEnd, FreqD, FreqU`。

### 1.2 已核对的数据事实

| 项 | 事实 |
|---|---|
| 样本 | 4000，四件套一一对应 |
| 文件 | `{id}.wav` / `{id}.mat` / `{id}_spectrogram.png` / `{id}.DR.txt` |
| 背景 | `background_2` 2666，`background_5` 1334 |
| WAV | 2ch 16-bit，50 kHz，5.0 s，250000 点（I/Q） |
| PNG | 875×656 RGB，**已渲染的幅度时频图（含轴/colorbar）** |
| 标注 | JSONL：`FreqD, FreqU, DateTimeStart, DateTimeEnd, Content` |
| 框数 | 每图 2–9 个，众数 5 |
| 时间 | `[0, 5e7]` ↔ 5 s |
| 频率 | 约 `[1 kHz, 50 kHz]` ↔ 全带宽 |
| 类别 | 4FSK, GMSK, Morse, 2FSK, 8-Tone, 16-Tone, FM, AM-DSB, PSK（约均衡） |

### 1.3 对网络的硬约束

- 幅度支路 **直接用现成 PNG**，不重画幅度图。
- 相位支路从 wav 的 I/Q 生成，与 PNG **同时频几何、同输入分辨率**。
- 两路共用同一套框标签。
- 时频图坐标轴有物理意义：**禁止上下翻转、禁止大角度旋转**；时间轴≠频率轴，不能当自然图像处理。
- 信号多为细长条（窄带长时 Morse/GMSK vs 宽带多音），定位难度在边界而不是中心点。
- 9 类可分性主要来自 **瞬时频率几何**，不是 constellation。PSK 只是一类，相位图不能只为 phase-hole 服务。

---

## 2. 问题拆解：幅度能做什么，相位必须补什么

### 2.1 幅度 PNG 的能力与上限

现成 spectrogram 已经把能量占据画得很清楚：

- 框的四条边 ≈ 能量从有到无的沿时间/频率突变 → **定位的主证据**
- FSK 多台阶、多音多脊、Morse 开关，在幅度上已有部分可分性
- 上限：GMSK vs FM（都是平滑扫频）、PSK vs 窄带数字、8-Tone vs 16-Tone（脊密度）在伪彩幅度图上容易糊

所以幅度流的任务是 **占据 + 粗结构**，不指望它单独把 9 类打满。

### 2.2 相位不能怎么用（Phase-Aware 的负面结论，直接当基线）

| 做法 | 对本任务 |
|---|---|
| `angle(STFT)` 伪彩 | wrapping 竖纹，CNN 当边，无效 |
| `Re/Im` 两通道 | 无局部事件，无效 |
| `∣X∣` 与 `φ` 早拼接 | 幅度主导，相位等于没融 |
| 原样 P-spectrogram 当第二路 | 本质仍是幅度图，与现成 PNG 冗余，且偏 PSK/QAM |

这些全部做消融，不当主方案。

### 2.3 本数据相位真正该编码的事件

不是 phase-hole 一种，而是五类：

| 调制 | 应在相位图上看到的稳定图案 |
|---|---|
| 2FSK / 4FSK | 沿时间的 IF 台阶，跳变处竖向突变 |
| GMSK / FM | 平滑弯曲的 IF 轨迹（GMSK 更受符号率约束） |
| 8-Tone / 16-Tone | 多条平行细脊，脊中心 IF≈0，脊数不同 |
| Morse | 相干开/关，有键时段 IF 稳定 |
| AM-DSB | 载频 IF≈0，双边带对称 |
| PSK | 符号处短 IF 脉冲 / 残余相位斑 |

**相位图创新的判据：这 9 类在图上肉眼可分，且与幅度 PNG 不冗余。**

---

## 3. 核心 Idea

> **幅度 PNG 负责“在哪、有多大”；从 I/Q 重建的相位事件图（PEM）负责“是什么”。**  
> 网络按这个分工来：相位只在能量占据区域内说话；分类头吃相位，回归头吃幅度；融合是门控不是对等相加。

方法名仍用 **MPFADet**（Magnitude–Phase Feature Adaptive Detector），指的是这个问题定义，不是 CMFADet 改名。

三个必须同时成立的点：

1. 相位是 **事件场 PEM**，不是 wrapping 相位。
2. 两路提取器按时频轴异性设计，而不是抄航拍模块。
3. 融合与检测头 **任务分工**，幅度保框、相位认类。

---

## 4. 相位图创新：Phase-Event Map（PEM）

这是本工作的主贡献之一，与用哪套检测器无关。

### 4.1 原则

从 I/Q 做 STFT 得 \(X[t,f]\in\mathbb{C}\)。PEM 必须：

1. 与幅度 PNG 几何对齐（同一时间轴、频率轴、最终输入尺寸）
2. 无 \(2\pi\) wrapping 伪影
3. 9 类事件可分、可解释
4. 输出 3 通道图，便于任意 CNN 当第二路输入

### 4.2 三通道定义

#### R：瞬时频率场 IF

\[
\Delta\phi_t[t,f]=\mathrm{atan2}\big(\Im(X[t,f]\,X^*[t-1,f]),\;\Re(X[t,f]\,X^*[t-1,f])\big)
\]
\[
\mathrm{IF}[t,f]=\frac{\Delta\phi_t[t,f]}{2\pi}\cdot\frac{f_s}{\mathrm{hop}}
\]

沿时间的相位差分，连续、无 wrapping。编码 FSK 台阶、FM/GMSK 轨迹、多音脊、PSK 跳变脉冲。

#### G：相位相干场 Coh

局部窗 \(t\pm W_t,\,f\pm W_f\) 内：

\[
\mathrm{Coh}[t,f]=\left\lvert\frac{1}{|N|}\sum_{(t',f')\in N} e^{j\phi[t',f']}\right\rvert\in[0,1]
\]

噪声≈0，稳态载波/FSK≈1。作用：掐掉无信号区的假相位；Morse 变成相干开关；给融合提供空间门。

#### B：残余相位事件 Residual

去掉线性频率相位后看残差，可视化用连续编码：

\[
B[t,f]=\sin(\phi_{\mathrm{res}}[t,f])
\]

把 Phase-Aware 的 phase hole **从幅度域搬到独立事件场**：PSK/GMSK 相移、FSK 跳频瞬间有斑纹，且不与 PNG 重复。

### 4.3 渲染

\[
\begin{aligned}
R &= \mathrm{clip}(\mathrm{IF}/(0.5 B_{\mathrm{bin}}),-1,1)\cdot 0.5+0.5 \\
G &= \mathrm{Coh} \\
B &= 0.5\sin(\phi_{\mathrm{res}})+0.5
\end{aligned}
\]

能量门控（用自己算的 \(\log\lvert X\rvert\)，不用 PNG 伪彩）：

\[
\mathrm{PEM}\leftarrow \mathrm{PEM}\cdot\sigma\big(\alpha(\log\lvert X\rvert-\tau)\big)
\]

可选：\(E=\lvert\partial_t \mathrm{IF}\rvert\cdot G\) 混入 R，锐化 FSK/PSK 跳变。

### 4.4 与 PNG 对齐（不做对，融合无意义）

PNG 含轴和 colorbar，875×656 不全是时频矩阵。

1. 从 PNG 裁出 spectrogram 画布（固定边距或找主色块）。
2. I/Q 的 STFT 覆盖 **0–5 s、0–50 kHz**，与 `DateTime` / `Freq` 一致（标注上限到 49995）。
3. IF/Coh/Residual resize 到画布尺寸。
4. 两路最终文件 **同高同宽**。优先：幅度、相位都只保留 crop 后的内容图，轴区不要进网络。若坚持幅度用整张 PNG，则 PEM 贴回同尺寸画布、轴区填 0。
5. 用若干样本把 DR 框画回 PNG，确认 y 是否翻转、频率轴方向。

STFT 起步（按本数据，不按 Phase-Aware 的 N=70）：

- \(f_s=50000\)，`n_fft=1024`（≈48.8 Hz，够 Morse/GMSK）
- `hop=256`（5.12 ms，5 s ≈ 976 帧），Hann
- 频率轴与标注一致（0–fs 或 fftshift 后映射到 0–50 kHz）

短窗（N=70 量级）只作为 B 通道旁路，增强 PSK 残差；主 IF/Coh 仍用 1024，否则 FSK 台阶被抹平。

### 4.5 生成后的人工验收（先于训练）

每类抽 5 张看 R/G/B：

- 看不出 FSK 台阶 → 查 I/Q 通道顺序、fftshift、IF 动态范围
- 噪声区很花 → 加强能量门控 / Coh
- 与幅度图几乎一样 → 通道定义失败，不要开训

---

## 5. 网络：按时频检测重设计（参考 CMFADet，不复刻）

检测器外形可以是 YOLO 式单阶段（工程熟、框任务匹配），但内部按本任务设计三个件：**轴异性 backbone、占据门控融合、任务分工头**。

### 5.1 为什么不直接用 SFEM / IR-AFAB / CIF / ATAH

| 模块 | 原用途 | 对本任务 |
|---|---|---|
| SFEM（Scharr + 图像 2D-FFT） | 航拍边缘 + 全局频谱 | 时频图再做 2D-FFT 物理意义弱；框边已是轴对齐的能量突变 |
| IR-AFAB | 热图弱纹理 | 相位事件沿时间轴局部、沿频率轴稀疏，需要轴分离而不是各向同性 mixer |
| 对称 CIF | RGB/IR 对等互补 | 幅度与相位不对等，对称交叉会让噪声相位污染框 |
| ATAH + OBB | 旋转框、任务对齐 | 本数据是轴对齐矩形，OBB 多余；任务分工的思想可留 |

借鉴的是 **异构双流 + 中层融合 + 分类/回归可解耦**，实现另写。

### 5.2 总体结构

```
幅度 PNG  ─►  MagNet（轴分离，偏边缘/占据）  ─► P3m,P4m,P5m
PEM 相位图 ─►  PhaseNet（轴分离，偏时间局部纹理） ─► P3p,P4p,P5p
                │
                ▼  Occupancy-Gated Fusion（每层）
                F_loc, F_cls
                │
          YOLO-style FPN + 解耦头
          回归 ← F_loc    分类 ← F_cls
```

轻量：MagNet / PhaseNet 用同一套 YOLO11n 量级的 stage 宽度，但 **不共享权重、卷积核各向异性不同**。4000 张用 n 档，不上 l/x。

### 5.3 MagNet：轴分离的占据提取

时频图上，框的左右边 = 时间突变，上下边 = 频率突变。用 **可分离条带卷积** 代替普通 3×3 为主：

- 水平核（1×k）：沿时间聚能量、找起止时刻
- 垂直核（k×1）：沿频率聚能量、找带宽
- 二者相加后再 1×1 混合

浅层加简单梯度（Sobel-x / Sobel-y 分开，不要 Scharr 混成各向同性边缘）。  
目的：把 PNG 伪彩里的色块变成稳定的占据特征，服务回归。

### 5.4 PhaseNet：沿时间看事件、沿频率看脊

PEM 的判别力在：

- 时间方向：IF 台阶、跳变、Morse 开关（要较长的时间感受野）
- 频率方向：多音脊是否分开（要较锐的频率局部性）

结构建议：每个 stage 内

```
X → DWConv(k×1)  → 频率局部
  → DWConv(1×k)  → 时间上下文
  → 1×1 + 残差
```

k 用 {3,7} 两档，覆盖 Morse 短点与 FSK 驻留。  
不要上大核各向同性注意力；相位纹理碎，全局 attention 容易把不同信号混在一起。

### 5.5 Occupancy-Gated Fusion（相对 CIF 的真正改动）

物理先验：**没有能量的地方，相位无定义。**

对每一层 \((F_m, F_p)\)：

\[
\begin{aligned}
g_{\mathrm{occ}} &= \sigma\big(\mathrm{Conv}_{1\times1}(F_m)\big) \\
F_p^{\mathrm{gated}} &= F_p \odot g_{\mathrm{occ}} \\
F_{\mathrm{loc}} &= F_m + \alpha\,\psi(F_p^{\mathrm{gated}}) \\
F_{\mathrm{cls}} &= F_p^{\mathrm{gated}} + \beta\,\phi(F_m)
\end{aligned}
\]

- \(g_{\mathrm{occ}}\) 由幅度预测占据门，相位只在门内有效
- \(\alpha\) 小（初值 0.2–0.3，可学）：相位仅轻微修幅度边界（FSK 跳变沿、多音脊更细）
- \(\beta\) 大（初值 1.0）：分类必须知道轮廓，否则 PEM 纹理没有空间锚点
- \(\psi,\phi\) 为轻量 1×1 或 3×3，不是对称通道对调相加

P3/P4/P5 各做一次。之后 FPN 可以两套（loc 一套、cls 一套）或一套后在头部分叉。推荐 **两套短 FPN**，避免 ADD 过早把分工打掉。

与 CMFADet-CIF 的差别：门的方向由物理定义（幅度→相位），权重不对等，输出直接拆成 loc/cls 两路，而不是融完再交给同一个 head 自己猜。

### 5.6 检测头

用解耦的 HBB 头即可（本数据无旋转）：

- 回归：DFL / 分布焦点框，输入 \(F_{\mathrm{loc}}\)。细长条用分布回归比直接 xywh 稳。
- 分类：输入 \(F_{\mathrm{cls}}\)。
- 置信度：\(\hat{s} = s_{\mathrm{cls}}\cdot s_{\mathrm{obj}}\)，\(s_{\mathrm{obj}}\) 来自 loc 分支的质量（例如预测 IoU 或 DFL 锐度），减少“类对但框飘”进 NMS。这是借鉴 Phase-Aware 的 localization-guided rescoring，实现上就是乘一个标量，不加新论文模块名。

第一版不要上 OBB、不要上 DGCL 离散化。若 val 上 mAP75 比 mAP50 掉很多，再把回归改成粗 bin + 细残差。

### 5.7 损失（对准目标值）

\[
L = \lambda_{\mathrm{cls}} L_{\mathrm{cls}} + \lambda_{\mathrm{box}} L_{\mathrm{box}} + \lambda_{\mathrm{dfl}} L_{\mathrm{dfl}} + \lambda_{\mathrm{occ}} L_{\mathrm{occ}}
\]

- \(L_{\mathrm{cls}}\)：BCE / Focal，**按类不均衡可加权**（PSK、16-Tone 若弱则提高权重）
- \(L_{\mathrm{box}}+L_{\mathrm{dfl}}\)：只监督 loc 头
- \(L_{\mathrm{occ}}\)（可选、弱权）：用 GT 框填成占据图监督 \(g_{\mathrm{occ}}\)，让门控先会找能量区

指标主看：**mAP50（检出）+ mAP75（框准）+ 分项 AP（类准）**。三者都是目标值，不能只报 mAP50。

---

## 6. 标签与数据管线

### 6.1 框换算

设有效画布宽高为 \((W_c, H_c)\)（crop 后），时间 \(T_0,T_1\in[0,5\times10^7]\)，频率 \(F_D,F_U\) 按 0–50 kHz：

\[
\begin{aligned}
x &= \frac{(T_0+T_1)/2}{5\times10^7},\quad
w = \frac{T_1-T_0}{5\times10^7} \\
y &= 1-\frac{(F_D+F_U)/2}{5\times10^4},\quad
h = \frac{F_U-F_D}{5\times10^4}
\end{aligned}
\]

YOLO：`cls x y w h` ∈ [0,1]。**y 是否翻转必须以画框回图为准。**

```
0:2FSK  1:4FSK  2:8-Tone  3:16-Tone  4:GMSK
5:FM    6:AM-DSB  7:Morse  8:PSK
```

### 6.2 划分

3200 / 400 / 400。按 `background_2` 与 `background_5` 分层，同一 id 的 mag/phase/label 同 split。避免把背景当性能。

### 6.3 增广

- 禁止垂直翻转、旋转
- 时间向平移可以；频率向大平移会破坏绝对频率，少用
- Mosaic 会打乱频率绝对位置，建议关
- 两路必须同一套几何变换

### 6.4 工程落点

可以用 CMFADet 的双路径 dataloader 当 **文件配对器**（`train` = 幅度，`train_ir` = PEM），但模型 yaml 自己写，不要加载 `Multi-SFEM-IRAFAB-CIF-*.yaml` 当主模型。检测头用 HBB。`imgsz` 512 或 640，scale=n。

---

## 7. 实验（证明目标值，而不是证明搬了 CMFADet）

### 7.1 主表：同一套轻量检测器，只换输入与融合

| ID | 输入 | 融合 | 回答什么 |
|---|---|---|---|
| A | 仅幅度 PNG | 单流 | 现成图下限：框好不好、类够不够 |
| B | 仅 PEM | 单流 | 相位能否独立框；预期定位弱、分类强 |
| C | mag + angle 伪彩 | 早拼接 / 对称加 | wrapping 是否真的无效 |
| D | mag + P-spectrogram | 中层融合 | 与 PNG 是否冗余 |
| E | mag + PEM | concat / ADD | 朴素融合上限 |
| **F** | **mag + PEM** | **占据门控 + 任务分工头** | **主模型** |

对照 A→F 必须同时报 mAP50 / mAP75 / 每类 AP。  
成功判据：

- 相对 A，mAP75 不掉或升（框没被相位污染）
- PSK、GMSK/FM、8/16-Tone 的 AP 明显升（相位真的在认类）
- C、D 低于 F（表征和融合都不是白做）

### 7.2 PEM 消融

仅 IF / 仅 Coh / 仅 Residual / 两两组合 / 全三通道；有无能量门控；长窗 vs 短窗 residual。  
预期：mAP75 吃 IF+Coh，PSK 的 AP 吃 Residual。

### 7.3 融合消融

- \(\alpha=\beta=1\) 且无占据门（接近对称 CIF 思想）
- \(\alpha=0\)（相位完全不进定位）
- 分类回归同特征（取消任务分工）
- 单套 FPN vs 双套

### 7.4 可视化（论文需要）

1. 9 类 PEM 的 R/G/B 图册
2. 同一例：PNG、PEM、\(g_{\mathrm{occ}}\)、检测框
3. A vs F 的分错样本（证明补的是类而不是刷 mAP50）

---

## 8. 预期贡献（按本任务写，不按 CMFADet 写）

1. **问题**：宽带时频检测里，幅度图能框不能认、相位又不能直接喂 CNN；mag/phase 是角色不对等的两路证据，不是多通道拼接问题。
2. **表征**：PEM 把瞬时频率、相干、残余相位编成与现成幅度 PNG 对齐的事件图，覆盖本数据 9 类（FSK/多音/FM/Morse/PSK），而不是把 phase-hole 烧回幅度谱。
3. **结构**：轴分离双流 + 幅度占据门控相位 + 回归/分类分特征，服务“框准 + 类准”两个目标值。

---

## 9. 实施顺序

1. 画框回 PNG，锁定 crop 与坐标（不训练）。
2. 生成 4000 张 PEM，9 类人工看图。
3. 转 YOLO 标签，跑通 **A**（仅幅度）。这是性能锚点。
4. 接双流，跑 **E** 再跑 **F**。先保证不掉 mAP75，再看分项 AP。
5. 做 C/D 与 PEM 通道消融。
6. 仅当 mAP75 仍差，再加粗细回归或加强 MagNet 条带卷积。

第一版改动量：PEM 脚本、标签脚本、MagNet/PhaseNet/Fusion/Head 四个模块、一份自己的 yaml。不要把贡献做成“在 CMFADet 上换了 CIF 名字”。

---

## 10. 风险

| 风险 | 以目标值为准的预案 |
|---|---|
| PNG 含轴，框错位 | 优先 crop 内容区；对齐重于“整图入网” |
| PEM 类间无差异 | 先修 STFT/I/Q/门控，不堆网络 |
| F 的 mAP75 < A | 降 \(\alpha\) 或 \(\alpha=0\)，保住框，再只让相位进分类 |
| 4000 张过拟合 | scale=n，关 Mosaic，按背景分层报指标 |
| PSK 仍弱 | 只加强 B 通道短窗残差，不改整体故事 |

---

## 11. 一句话

幅度用现成 PNG 管占据与框；相位用 PEM 管 9 类事件；融合是“有能量才看相位、分类吃相位、回归吃幅度”。CMFADet 证明双流中层融合可行，Phase-Aware 证明相位必须视觉化——模块按本任务的时频轴和 9 类目标重做。
