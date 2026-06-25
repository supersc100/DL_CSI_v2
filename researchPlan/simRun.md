# 仿真曲线运行指南 (simPlan 1.1 / 1.2 / 1.3 / 2.1 / 2.2)

配合 `scripts/run_simulation.py` 使用。该脚本在**评测时**注入 AWGN、切换导频开销/采样策略，
复用已训练的 Stage1 / Stage2 checkpoint，无需重新生成数据或重训（少数消融除外，见文末）。

> ⚠️ **前置：config 必须与 checkpoint 匹配。** `config.yaml` 里的 `model.feature_dim / hidden_dim /
> csi_encoder.base_channels / regression_head.hidden_dim` 等必须与你训练 Stage1/Stage2 时所用的值一致，
> 否则加载 checkpoint 会报 size mismatch。先用一份与 checkpoint 对应的 config 跑通 `evaluate_phase2.py`，
> 再用同一份 config 跑本脚本。

所有命令用项目解释器：`D:/anaconda/envs/csiMinimax/python.exe`。
输出落在 `simulation.output_dir`（默认 `./outputs/sim/`），每图一个 `results_<figure>.csv` + `fig_<figure>.png`。
扫描点/种子可在 `config.yaml` 的 `simulation:` 段改，或用 CLI 覆盖。

---

## 1.1 NMSE vs SNR（图3）
```bash
D:/anaconda/envs/csiMinimax/python.exe scripts/run_simulation.py \
    --figure nmse_snr \
    --checkpoint outputs/checkpoints/phase2_main_best.pt \
    --quant-curve
```
- 曲线：所提(理想)、所提量化(`proposed_quant_{N}bit`)、仅幅度(`magnitude_only`)、线性插值(`linear_interp`)、
  DFT插值(`dft_interp`)、全反馈(`full_feedback`)。
- `--quant-curve` 增加 16-bit 量化导频那条；横轴 SNR 由 `simulation.snr_list` 控制。
- 误差棒：`simulation.seeds`（默认 3 个种子，对应 3 次噪声实现）。

**第一阶段单独的幅度 NMSE vs SNR**（验证 UL 加噪生效）：
```bash
D:/anaconda/envs/csiMinimax/python.exe scripts/run_simulation.py \
    --figure nmse_snr --stage 1 --checkpoint outputs/checkpoints/best.pt
```

## 1.2 NMSE vs 导频开销（图4）
```bash
D:/anaconda/envs/csiMinimax/python.exe scripts/run_simulation.py \
    --figure nmse_overhead \
    --checkpoint outputs/checkpoints/phase2_main_best.pt \
    --quant-curve
```
- 固定 SNR=`simulation.fixed_snr`（默认 10 dB），横轴为导频开销 %。
- 采用 `nonuniform` 采样策略扫描开销（`adaptive` 会按时延扩展自动定间距、忽略 base_spacing，导致所有点塌缩成一个，故不能用于开销扫描）。可用 `--overhead-strategy {uniform,nonuniform}` 切换：
  - `nonuniform`（默认）：均匀网格 + 峰值加密，更贴近训练时的模型输入分布，但实际开销起点偏高（≈22%）。
  - `uniform`：纯均匀网格，贴近名义开销、能画到更低的开销段（如 6.25%），代价是离训练分布更远（更 OOD）。
- CSV 的 `x` 列是脚本实测的**实际开销%**（含峰值加密的固定预算，故实际值高于名义值），非名义目标值。
- **曲线构成（方案1，区分"吃导频"与"参考线"）**：
  - `proposed` / `proposed_quant_{N}bit` / `linear_interp` / `dft_interp`：真正消费稀疏导频，随开销变化的扫描曲线；
  - `magnitude_only`：0% 开销锚点（无 DL 导频、phase=0），单点 @ x=0；
  - `full_feedback`：100% 开销锚点（全带 CSI 以 16-bit 反馈），单点 @ x=100。
- 反馈比特数横轴：`bits = 采样子载波数 × N_tx × N_rx × 2 × num_bits`，可由 CSV 的实际开销换算后另绘。
- ⚠️ 当前 `data.num_subcarriers=32` 较小，间距为整数 → 部分名义开销点会合并（如 40%/50% 都被 clamp 到 spacing=2），且实际开销区间约 22%–62%、画不到超低开销段。需要更细的开销网格请用 M=64/128 的数据重新生成 test 集。

## 1.3 SE vs SNR（图5）
```bash
D:/anaconda/envs/csiMinimax/python.exe scripts/run_simulation.py \
    --figure se_snr --checkpoint outputs/checkpoints/phase2_main_best.pt
```
- 曲线：所提方案预测 CSI 预编码后的 SE(`proposed`) + 完美 CSI 上界(`perfect_csi`)。
- SE 模型：逐子载波单流特征波束成形，`SE=log2(1+snr·||H_true·w_pred||²)`，在归一化信道上、
  以 `--snr-list` 各点为工作 SNR 计算（详见 `src/utils/metrics.spectral_efficiency` docstring）。

## 2.1 图4c 采样策略对比
```bash
D:/anaconda/envs/csiMinimax/python.exe scripts/run_simulation.py \
    --figure sampling_overhead --checkpoint outputs/checkpoints/phase2_main_best.pt
```
- 三条曲线采用**控制变量法**（公平消融）：
  - **uniform**：自适应基础网格 + **无**峰值加密；
  - **nonuniform**：自适应基础网格 + **随机位置**峰值加密；
  - **adaptive**：自适应基础网格 + **UL 能量峰位置**峰值加密（所提方案）。
- 三者的基础网格都由 RMS 时延扩展决定（`use_adaptive=True`），仅“额外加密策略”不同，避免把"不同基础网格"和"不同加密方式"混为一谈。
- 固定 SNR=`fixed_snr`，横轴为名义目标开销%（注意：因基础网格由时延扩展自适应决定，同一 `adaptive` 规则下基础间距不随目标开销变化；不同 x 点之间的差异主要来自峰值加密的预算随 spacing 略有变化）。

---

## 需要重训 / 重新生成的部分（脚手架已就绪，命令如下）

### 2.1 图4a 有无幅度辅助（需训练 no-magnitude 变体）
1. 复制一份 config，设 `phase2.model.use_magnitude: false`，重训 Stage2：
   ```bash
   D:/anaconda/envs/csiMinimax/python.exe scripts/train_phase2.py --config config_nomag.yaml
   ```
2. 分别用 `--figure nmse_snr` 跑两个 checkpoint（有/无幅度），把两条 `proposed` 曲线叠加即图4a。

### 2.1 图4b 有无大尺度参数（需重训 Stage1）
1. 设 `model.use_large_scale: false`，重训 Stage1（及对应 Stage2）。
2. 同上叠加两条 `proposed` 曲线。

### 2.2 不同信道场景 CDL-A/C/D 泛化（需重新生成 3 份数据）
1. 分别设 `data.scenario: "A"` / `"C"` / `"D"`（`_scenario_to_cdl` 对未知名直接透传给 Sionna CDL），
   生成各自的 test 集：
   ```bash
   D:/anaconda/envs/csiMinimax/python.exe scripts/generate_data.py --config config_cdlA.yaml --split test
   ```
2. 逐场景跑 `--figure nmse_snr --split test`（指向对应 H5），把三条 `proposed` 曲线叠加即图7。

---

## 研究 caveat
Stage2 训练时未注入噪声，低 SNR 评测属 OOD 鲁棒性测试，曲线在低 SNR 可能明显下滑。
若需改善低 SNR 表现，可在训练 DataLoader 上设 `snr_db`（随机/课程式）做噪声增强重训——本次未做。
