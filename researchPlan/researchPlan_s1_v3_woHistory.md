# 第一阶段研究计划：FDD 下行 CSI 幅度预测（无历史信息版）

> **课题名称**：《大规模 MIMO 中基于深度学习的信道状态信息反馈研究》
> **阶段定位**：第一阶段（零下行反馈条件下预测下行 CSI 幅度，为第二阶段相位恢复提供先验）
> **技术路线**：CNN + Transformer 级联架构（不引入 LLM）
> **核心约束**：第一阶段仅使用当前上行 CSI 与上下行共享的大尺度参数，**不引入历史 UL/DL CSI 信息**
> **适用目标**：国内通信/电子类学报（如《通信学报》《电子与信息学报》）
> **版本**：v4.0（无历史信息版，加入残差 log-幅度头、UL 几何扰动、混合场景训练与 log-ratio 损失）

---

## 一、研究目标与核心问题

### 1.1 研究背景

在 FDD 大规模 MIMO 系统中，上下行工作于不同载频，信道快衰落近似独立，因此不存在 TDD 系统中的强互易性。然而，上下行链路共享同一物理传播环境，其**大尺度参数**（路径延迟、离开角/到达角、簇功率、RMS 延迟扩展等）具有高度相关性。如何利用这一部分互易性，在**不依赖任何下行反馈**的前提下，由上行 CSI 推断下行信道的有用结构，是本阶段要解决的核心问题。

### 1.2 阶段目标

本阶段旨在建立从上行 CSI 与大尺度参数到下行 CSI **幅度**的统计映射：

$$
\log |\mathbf{H}_{\text{DL}}| = \log |\mathbf{H}_{\text{UL}}| + \Delta\big(\mathbf{H}_{\text{UL}}, \mathbf{z}_{\text{LSP}}\big)
$$

其中：

- $\mathbf{H}_{\text{UL}} \in \mathbb{C}^{N_{\text{bs}} \times N_{\text{ue}} \times M}$ 为当前时隙上行 CSI；
- $\mathbf{z}_{\text{LSP}}$ 为上下行共享的大尺度参数向量；
- 回归头不再直接输出 $|H_{\text{DL}}|$，而是输出相对于上行 log-幅度的**残差** $\Delta$，从而强制网络必须利用上行 CSI；
- 输出 $|H_{\text{DL}}|$ 为下行 CSI 幅度谱，作为第二阶段相位恢复的结构性先验。

### 1.3 核心科学问题

1. **FDD 不完全互易性下的可学习结构是什么？**
   上下行快衰落独立导致复数 CSI 不可预测；但角延迟域中的功率谱由共享的路径角度、延迟和功率决定，因此**幅度谱**成为可学习且有用的目标。

2. **大尺度参数如何辅助下行幅度预测？**
   RMS 延迟扩展、角度扩展、LOS/NLOS 指示等参数刻画了传播环境，能够为网络提供关于多径分布的先验约束。

3. **历史 UL/DL CSI 是否必要？**
   早期设计中引入了历史上下行对作为时序先验，但实验发现其带来的增益有限且显著增加数据生成与训练开销。因此本阶段最终采用**仅当前上行 CSI + 大尺度参数**的精简输入。

4. **如何防止网络忽略上行 CSI 而 memorizing 固定模板？**
   直接预测 $|H_{\text{DL}}|$ 时，网络容易输出数据集平均功率谱。为此引入：
   - **残差 log-幅度头**：输出 $\log |H_{\text{DL}}| - \log |H_{\text{UL}}|$；
   - **log-ratio 幅度损失**：在 log-ratio 空间衡量预测与真值的差异；
   - **UL 几何扰动**：在数据生成时扰动上行射线角度/延迟/功率；
   - **训练时 UL 损坏**：随机加噪与随机掩蔽上行 CSI；
   - **混合场景训练**：每个样本随机选自 UMa/UMi/RMa，避免过拟合单一传播模板。

---

## 二、整体技术路线

本课题采用**两阶段级联架构**，第一阶段专注于**零反馈幅度预测**：

```
阶段一（本阶段核心）：
  当前上行 CSI  ──→ 角延迟域变换 ──→ 3D CNN 编码
  大尺度参数   ──→ MLP 编码 ────────────┤
                                        ↓
                              轻量 Transformer 融合
                                        ↓
                    回归头：输出 Δ_log_amp（相对上行 log-幅度残差）
                                        ↓
                              |H_DL_pred| = |H_UL| · exp(Δ_log_amp)
                                        ↓
                              取幅度 |H_DL| 作为第二阶段先验

阶段二（后续工作）：
  稀疏下行导频观测 + |H_DL| ──→ 相位恢复网络 ──→ ∠H_DL
                                        ↓
                         融合：|H_DL| ⊙ e^{j∠H_DL} → 完整下行 CSI
```

**关键约束**：

- 第一阶段**完全不使用任何下行 CSI**（当前或历史）作为输入；
- 第一阶段回归头输出**相对上行 log-幅度的残差**，训练目标聚焦**log-ratio 幅度 MSE**；
- 历史 UL/DL CSI 分支在代码中保留为可配置消融接口，但本版本研究计划与训练配置中将其关闭。

---

## 三、数据生成方案

### 3.1 信道模型与参数

| 参数 | 配置 |
|------|------|
| **仿真平台** | Sionna 2.0 (TensorFlow) |
| **信道模型** | 3GPP TR 38.901 CDL (CDL-A, CDL-C, CDL-D) |
| **场景** | UMa / UMi / RMa，训练时混合抽取 |
| **天线配置** | BS: 32/64 阵元 ULA；UE: 4/8 阵元 ULA |
| **OFDM 参数** | 子载波间隔 15/30 kHz，有效子载波数 32/64/128 |
| **载频** | 上行约 2.0 GHz，下行约 2.1 GHz（FDD 双工间隔约 100 MHz） |

### 3.2 上下行信道生成策略

- 下行链路从 Sionna CDL 对象抽取**路径延迟 $\tau$**、**离开角 AoD**、**到达角 AoA**、**簇功率 powers**；
- 上行链路在下行射线参数基础上做**几何扰动**（见 3.4 节），并**独立采样快衰落复增益**，以体现 FDD 小尺度去相关特性；
- 显式提取 Sionna CDL 对象内部几何参数（`._aod`, `._aoa`, `._powers`, `._tau`）手动重构上下行信道；
- Sionna (TensorFlow) → NumPy → PyTorch Tensor，保存为 HDF5 格式。

### 3.3 混合场景训练

在 `config.yaml` 中将 `data.scenario` 设置为列表：

```yaml
scenario: ["UMa", "UMi", "RMa"]
```

数据生成器每次随机抽取一个场景，从而：

- 提高模型对 CDL-A/C/D 的泛化能力；
- 降低网络 memorizing 单一传播模板的概率；
- 为后续跨场景鲁棒性实验提供统一数据基础。

验证/测试集可与训练集采用同分布混合，也可单独生成单场景数据以评估泛化性。

### 3.4 UL 几何扰动

为防止上下行大尺度几何完全一致导致网络忽略上行 CSI，数据生成时加入可控扰动：

```yaml
ul_geometry_perturbation:
  enabled: true
  angle_std_deg: 3.0      # 方位角扰动标准差
  delay_std_s: 5.0e-9     # 路径延迟扰动标准差
  power_std_db: 3.0       # 每径功率扰动标准差（dB）
```

扰动后上行射线与下行射线仅近似共享几何，从而强制 Stage 1 网络从当前上行 CSI 中提取有用信息，而非依赖一个隐含的共享模板。

### 3.5 样本规模

| 数据集 | 样本数 | 说明 |
|--------|--------|------|
| 训练集 | 70,000 | 混合 UMa/UMi/RMa，覆盖多种场景、LOS/NLOS |
| 验证集 | 10,000 | 用于超参数调优和早停 |
| 测试集 | 20,000 | 严格隔离，仅用于最终评估 |

**每个样本包含**：

- 当前时隙上行 CSI：$\mathbf{H}_{\text{UL}} \in \mathbb{C}^{N_{\text{bs}} \times N_{\text{ue}} \times M}$
- 当前时隙下行 CSI（真值）：$\mathbf{H}_{\text{DL}} \in \mathbb{C}^{N_{\text{bs}} \times N_{\text{ue}} \times M}$
- 大尺度参数向量：路径功率、RMS 延迟扩展、AoA/AoD 角度扩展、Rician K-factor、BS-UE 距离
- 历史 UL/DL CSI（可选，本研究中不使用）

### 3.6 TDD Oracle 数据集

为度量 FDD 与 TDD 的性能差距，额外生成一个 TDD Oracle 测试集：上下行使用**完全相同的快衰落**，作为理论上界参考。

---

## 四、数据预处理

### 4.1 角延迟域变换

将空间–频域 CSI 变换到 Angle-Delay (AD) 域，以增强 UL/DL 之间的结构互易性：

```
空间-频域 H ──→ 沿 BS 天线维度 DFT ──→ 角度域
            ──→ 沿子载波维度 IFFT ──→ 延迟域
            ──→ 角延迟域 H_AD
```

- 角度域分辨率：与 BS 天线数相同（32/64）；
- 延迟域分辨率：与子载波数相同（32/64/128）；
- 变换通过 `torch.fft` 实现，梯度可反向传播。

### 4.2 逐样本归一化

对每个样本在角延迟域做零均值、单位能量归一化：

$$
\mathbf{H}_{\text{AD}}^{\text{norm}} = \frac{\mathbf{H}_{\text{AD}} - \mu}{\sigma}, \quad \sigma = \sqrt{\mathbb{E}\big[|\mathbf{H}_{\text{AD}}|^2\big]}
$$

该归一化使网络专注于学习功率谱结构，而非样本绝对能量。

### 4.3 复数转实部/虚部通道

为便于 3D CNN 处理，将复数 CSI 拆分为实部与虚部两个通道：

$$
\mathbf{H} \in \mathbb{C}^{N_{\text{bs}} \times N_{\text{ue}} \times M} \Rightarrow \mathbf{H}^{\text{RI}} \in \mathbb{R}^{2 \times N_{\text{bs}} \times N_{\text{ue}} \times M}
$$

回归头在 `amp_phase` 模式下分别输出幅度残差与相位，再重组为复数张量。

---

## 五、网络架构设计

第一阶段采用 **CNN + Transformer** 结构（不引入 LLM），整体模型为 `DlCsiPredictor`。

### 5.1 输入层

| 输入 | 维度 | 说明 |
|------|------|------|
| 当前上行 CSI（角延迟域） | $[B, 2, N_{\text{bs}}, N_{\text{ue}}, M]$ | 实部/虚部双通道 |
| 大尺度参数 | $[B, D_{\text{lsp}}]$ | 上下行共享的物理环境信息 |
| 历史 UL/DL CSI | 可选，本研究中关闭 | 代码中保留为消融接口 |

### 5.2 核心模块

#### 模块 A：当前上行 CSI 编码器（CsiEncoder）

- **结构**：2 层 3D 卷积 + GroupNorm/LayerNorm + GELU + MaxPool + Dropout；
- **池化策略**：沿 BS 天线维度与频率维度做下采样，**保留 UE 天线维度**（防止小 UE 阵列塌陷）；
- **输出**：全局平均池化后经线性投影为特征向量 $\mathbf{f}_{\text{UL}} \in \mathbb{R}^{D}$。

#### 模块 B：大尺度参数编码器（EnvironmentEncoder）

- **结构**：2 层 MLP（$128 \to 256$）+ SiLU + Dropout + 线性投影至 $D$ 维；
- **输出**：环境特征向量 $\mathbf{f}_{\text{Env}} \in \mathbb{R}^{D}$。

#### 模块 C：可选交叉注意力融合（CrossAttentionFusion）

- 对当前上行特征与大尺度特征先做一层 token 级自注意力；
- 增强模态间交互后再送入 Transformer Fusion。

#### 模块 D：Transformer 融合块（TransformerFusion）

- **结构**：4 层标准 `TransformerEncoder`，隐藏维度 $H$；
- **输入**：固定数量的模态 token（当前上行 + 大尺度，共 2 个）；
- **位置编码**：可学习的位置嵌入；
- **输出**：融合后的 token 序列 $[B, N_{\text{tokens}}, H]$。

#### 模块 E：残差 Log-幅度回归头（RegressionHead）

- **结构**：2 层 MLP（$H \to 1024 \to 2 \cdot N_{\text{bs}} N_{\text{ue}} M$）；
- **输出模式**：`"amp_phase"`；
- **残差 log-幅度**：

$$
\log |\hat{\mathbf{H}}_{\text{DL}}| = \log |\mathbf{H}_{\text{UL}}| + \text{tanh}(\mathbf{x}_{\text{amp}}) \cdot \rho_{\max}
$$

其中 $\rho_{\max}=5.0$（nats），用于限制动态范围并避免 `exp()` 溢出。

- **相位分支**：输出 $\cos/\sin$ 或直接预测相位角（训练时用于复数 MSE 辅助项，最终取幅度作为先验）；
- **显式上行依赖**：由于幅度是相对于上行 log-幅度构建的，网络无法输出一个与上行无关的固定模板。

### 5.3 架构示意图

```text
Input
├── 当前上行 CSI (角延迟域)
│   └── 3D CNN (CsiEncoder)               -->  [B, D]
├── 大尺度参数
│   └── MLP (EnvironmentEncoder)          -->  [B, D]
│   (可选) 历史 UL/DL CSI                  -->  [B, D]  (本研究中关闭)
│
        Cross-Attention Fusion (可选)     -->  [B, num_tokens, D]
        TransformerFusion                 -->  [B, num_tokens, H]
        Mean Pooling                      -->  [B, H]
        RegressionHead (amp_phase)
        ├── 残差 log-amp: Δ_log_amp      -->  log|H_DL| = log|H_UL| + Δ
        └── 相位分支                     -->  辅助复数 MSE
        exp(log|H_UL| + Δ) + j·phase     -->  预测下行 CSI (角延迟域)
```

### 5.4 为何选择角延迟域、残差 log-幅度与幅度目标

- FDD 上下行共享路径几何，因此角延迟域功率谱高度相关；
- 上下行快衰落独立，导致复数 CSI 的相位不可预测；
- 在逐样本归一化后，最优复数 MSE 预测器近似为零，而幅度谱仍保留可学习结构；
- 直接预测 $|H_{\text{DL}}|$ 会导致网络输出平均功率谱（shortcut learning）；
- 残差 log-幅度头与 log-ratio 损失共同强制网络学习“上行→下行”的相对变换，而非绝对模板。

---

## 六、防止 Shortcut Learning 的机制

### 6.1 残差 Log-幅度头

将输出定义为相对于上行 log-幅度的残差，使得零输入（上行=0）时输出也必然为 0（或接近 0）。任何固定模板都无法在变化的 $|H_{\text{UL}}|$ 上取得低损失。

### 6.2 Log-Ratio 幅度损失

不再最小化 $\big\| |\hat{H}| - |H_{\text{true}}| \big\|_2^2$，而是最小化：

$$
\mathcal{L}_{\text{mag}} = \Big\| \big(\log |\hat{H}| - \log |H_{\text{UL}}|\big) - \big(\log |H_{\text{true}}| - \log |H_{\text{UL}}|\big) \Big\|_2^2
$$

等价于在 log-ratio 空间对齐预测与真值。该形式对动态范围更稳定，且天然与残差 log-幅度头匹配。

### 6.3 UL 几何扰动

在数据生成阶段对上行射线角度、延迟、功率加入高斯扰动，破坏 UL/DL 大尺度几何的完全一致性。

### 6.4 训练时 UL 损坏增强

- **AWGN 加噪**：以 80% 概率从 `[-10, -5, 0, 5, 10, 15, 20]` dB 中随机选择 SNR，对上行 CSI 加噪；
- **随机空间-频率掩蔽**：以 50% 概率随机遮蔽 50% 的上行角延迟域元素。

这些增强使得网络必须从不完整/有噪的上行观测中恢复相对幅度变换。

### 6.5 混合场景训练

每个训练样本随机选自 UMa/UMi/RMa，不同场景的功率谱结构差异显著，固定模板无法同时拟合多个场景。

---

## 七、损失函数

采用组合损失函数：

$$
\mathcal{L} = \lambda_{\text{MSE}} \mathcal{L}_{\text{MSE}} + \lambda_{\text{mag}} \mathcal{L}_{\text{log-ratio-mag}} + \lambda_{\text{AD-L1}} \mathcal{L}_{\text{AD-L1}}
$$

### 7.1 各损失项定义

| 损失项 | 公式 | 默认权重 | 作用 |
|--------|------|---------|------|
| **复数 MSE** | $\|\hat{\mathbf{H}} - \mathbf{H}_{\text{true}}\|_2^2$ | 0.1 | 可选辅助项，保留对相位的弱监督 |
| **Log-Ratio 幅度 MSE** | $\big\| (\log|\hat{\mathbf{H}}|-\log|\mathbf{H}_{\text{UL}}|) - (\log|\mathbf{H}_{\text{true}}|-\log|\mathbf{H}_{\text{UL}}|) \big\|_2^2$ | 1.0 | **主导目标**，防止固定模板 collapse |
| **角延迟域 L1** | $\big\| \log|\hat{\mathbf{H}}| - \log|\mathbf{H}_{\text{true}}| \big\|_1$ | 0.1 | 保持稀疏结构与功率谱锐度 |

注：log 运算使用 `eps=1e-2` 以保证数值稳定性。

### 7.2 为何以 log-ratio 幅度损失为主导

- FDD 数据生成器对上下行使用独立快衰落复增益；
- 复数 MSE 在该设定下接近不可学习（最优预测器趋于零）；
- 原始幅度 MSE 仍可能通过输出平均功率谱获得较低损失；
- log-ratio 损失要求预测与上行幅度成比例变化，固定模板无法最小化该损失；
- 与残差 log-幅度头配合，形成端到端的上行依赖约束。

---

## 八、训练策略

第一阶段采用**单阶段端到端训练**，所有参数同时优化，无需分阶段冻结。

### 8.1 优化器与正则化

| 参数 | 配置 |
|------|------|
| **优化器** | AdamW |
| **初始学习率** | $1 \times 10^{-4}$（为 log-ratio 稳定性降低） |
| **权重衰减** | $1 \times 10^{-4}$ |
| **学习率调度** | Cosine Annealing |
| **梯度裁剪** | $\ell_2$ norm = 1.0 |
| **Dropout** | 0.1（CNN 与 Transformer） |

### 8.2 训练流程

| 配置 | 详情 |
|------|------|
| **训练 epoch** | 50（可扩展） |
| **Batch size** | 128 |
| **早停监控指标** | `val_magnitude_nmse_db` |
| **早停耐心** | 10 个 epoch |
| **混合精度** | 前向使用 bfloat16，损失与优化器状态使用 float32 |
| **最佳模型保存** | `outputs/checkpoints/best.pt` |
| **周期模型保存** | `outputs/checkpoints/epoch{epoch}.pt` |

### 8.3 训练时 UL 增强配置

```yaml
training:
  ul_noise_prob: 0.8
  ul_noise_snr_list: [-10, -5, 0, 5, 10, 15, 20]
  ul_mask_prob: 0.5
  ul_mask_ratio: 0.5
  loss:
    mse_weight: 0.1
    magnitude_weight: 1.0
    angle_delay_l1_weight: 0.1
    use_ratio: true
    ratio_eps: 1.0e-2
```

### 8.4 消融配置

通过 `config.yaml` 中的开关可方便地进行消融实验：

- `use_history: false`：关闭历史 UL/DL CSI 分支（本研究默认）；
- `use_large_scale: false`：关闭大尺度参数分支，验证大尺度先验的贡献；
- `model.regression_head.use_residual_log_amp: false`：退化为普通 amp_phase 头；
- `training.loss.use_ratio: false`：退化为原始幅度 MSE；
- `data.ul_geometry_perturbation.enabled: false`：关闭 UL 几何扰动；
- `data.scenario` 改为单一字符串：验证混合场景训练贡献。

---

## 九、评估指标

| 指标 | 定义 | 用途 |
|------|------|------|
| **NMSE** | $\mathbb{E}\left[ 10\log_{10}\frac{\|\hat{\mathbf{H}} - \mathbf{H}_{\text{true}}\|_F^2}{\|\mathbf{H}_{\text{true}}\|_F^2} \right]$ | 复数 CSI 归一化均方误差（dB） |
| **幅度 NMSE** | 对 $|\hat{\mathbf{H}}|$ 与 $|\mathbf{H}_{\text{true}}|$ 计算 NMSE（dB） | **主指标**，衡量幅度预测精度 |
| **幅度余弦相似度** | 展平幅度谱后的余弦相似度 | 衡量功率谱结构相似性 |
| **复数余弦相似度** | 展平复数 CSI 后的余弦相似度 | 辅助参考 |

### 9.1 基线对比方案

| 基线 | 描述 | 目的 |
|------|------|------|
| **copy_ul** | 直接以当前上行 CSI 作为下行预测 | 性能下界 |
| **tdd_oracle** | 假设上下行完全互易 | 性能上界 |
| **no_large_scale** | 仅使用当前上行 CSI，去掉大尺度参数 | 消融：大尺度参数贡献 |
| **no_residual_head** | 使用普通 amp_phase 头而非残差 log-幅度头 | 消融：残差设计贡献 |
| **no_log_ratio** | 使用原始幅度 MSE 而非 log-ratio 损失 | 消融：log-ratio 损失贡献 |
| **no_ul_perturbation** | 关闭 UL 几何扰动与训练时 UL 损坏 | 消融：上行依赖机制贡献 |
| **single_scene** | 仅在单一场景（如 UMa）上训练 | 消融：混合场景训练贡献 |

---

## 十、核心创新点

1. **FDD 不完全互易性下的角延迟域幅度预测范式**
   明确指出 FDD 上下行快衰落独立导致复数 CSI 不可学习，将预测目标从复数信道降维到角延迟域功率谱，使网络学习具有物理意义的可预测结构。

2. **残差 Log-幅度回归头**
   不直接预测下行幅度，而是预测相对上行 log-幅度的残差，从网络结构上消除“固定模板” shortcut 的可能性。

3. **Log-Ratio 幅度损失**
   在 log-ratio 空间衡量预测误差，与残差头配合，强制网络输出随上行 CSI 变化的相对幅度变换。

4. **UL 几何扰动与训练时 UL 损坏**
   从数据生成和训练增强两个层面破坏 UL/DL 的完美相关性，进一步提升网络对上行 CSI 的依赖性。

5. **混合场景训练**
   训练数据随机选自多种 CDL 场景，提升模型泛化性并抑制对单一传播模板的 memorization。

6. **剔除历史信息的精简输入设计**
   通过实验验证历史 UL/DL CSI 带来的增益有限，最终采用当前上行 CSI + 大尺度参数的轻量输入，降低数据生成、存储与训练成本。

7. **为第二阶段提供可冻结的幅度先验**
   第一阶段输出的幅度谱作为第二阶段相位恢复网络的结构性先验，支撑“零反馈幅度预测 + 少反馈相位恢复”的两阶段低开销 FDD CSI 获取框架。

---

*本文档基于 DL-CSI-v2 项目第一阶段最新实现整理，与第二阶段研究计划（`researchPlan_s2_v3_woULCsi.md`）形成完整的技术路线说明。*
