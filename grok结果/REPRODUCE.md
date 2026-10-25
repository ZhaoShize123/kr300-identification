# 复现说明（REPRODUCE）

## 0. 前提
- Linux x86_64，python3（自检用 3.13.5），`unar`（解压 RAR v4），约 2 GB 磁盘（另需安装 torch 的空间），可上网。
- 环境锁定见 `env/requirements-lock.txt`、`env/ENVIRONMENT.md`。

## 1. 一键运行
```bash
git clone https://github.com/ZhaoShize123/kr300-identification.git && cd kr300-identification
bash grok结果/reproduce.sh            # 自动建 .venv 并安装 numpy 2.5.4 / scipy 1.18.1 / matplotlib 3.11.2 / pin 4.1.0 / torch 2.14.1+cpu
# 已有环境：PY=/path/to/python bash grok结果/reproduce.sh
# 只跑基线：SKIP_THREE_STAGE=1 bash grok结果/reproduce.sh
```
日志写入 `grok结果/selfcheck/`。

## 2. 脚本做了什么（也可手动逐条执行）
```bash
python download_data.py                                   # 下载 + MD5 校验
unar -f -o data/x data/Robot_Identification_Benchmark_With_Raw_Data.rar
mkdir -p data/raw_data && find data/x -name 'recording_*.mat' -exec cp {} data/raw_data/ \;   # 12 个文件
python filter_causal.py
python phase_calibration.py
python identify.py
python psd_constrained.py
cp psd_constrained_params.npz final_psd_params.npz; cp psd_constrained.json final_psd.json   # 原 README 缺失的一步
python residual_analysis.py        # 必须在 redo_identify_j1 之前（生成 residuals.npz）
python redo_identify_j1.py         # 打印 posthoc test mean rmse 1.0288…
(cd grok结果/three_stage && grep ' freeze.json$' freeze.sha256 | sha256sum -c -)
python grok结果/three_stage/run.py      # 约 45–80 分钟（8 线程），首次会生成 Y_*.npz（~360 MB）
python grok结果/three_stage/analyze.py  # 写 summary.txt
```

## 3. 预期数值
- 基线：`posthoc test mean rmse_Nm 1.0288440726212202`；J1 0.952827…
- 三阶段 CV / traj12：见 `README.md` 表格与 `three_stage/summary.txt`。

## 4. 校验和（SHA256）
| 文件 | SHA256 |
|---|---|
| filtered_causal.npz | `c0ecb4b39df40a539a7e480021d8b01f9cba28db6f5d13f06b725584594c10d0` |
| identification.json | `6fe305de873053dd52046f1ed97d6904ec2b92ef7fefc26955702fed862542dd` |
| psd_constrained.json | `f043aac09aba43c742195ec156d65a13adf45c6b5c51fd1187e5fb84f94ea4ab` |
| j1_correction.json | `671d15599f615270644fb6f986d7360b139155a08063de44e280f8fdbb4602a9` |
| three_stage/freeze.json | `fba00a6715f7a79e9f0fb78560955561a6a30a542bcf162474a24b0d85ece8ed` |
| three_stage/results.json | `8173058dbf876b583d619017f2d805db33a49e820b1692db56c14813fb9bd3d2` |
| three_stage/summary.txt | `490a285391f02f20a617f1646f38eb999597345a1d579337597ba5198e6aaa7d` |

`freeze.sha256` 第二行是改路径之前 run.py 的哈希，已不匹配（只改了 `REPO` 一行，见 `docs/CODE_GUIDE.md` §4），因此只校验 freeze.json。

## 5. 自检记录（2026-10-25，Asia/Shanghai）
在 `/workspace/selfcheck_kr300`（全新 `git clone`）中从下载原始数据开始运行 `reproduce.sh`（使用同一锁定环境）：
- 数据两个 rar 的 MD5 校验通过；上表 4 个基线输出与原工作目录**逐位一致**，基线 1.0288 吻合；
- `results.json` 与 `summary.txt` 的 SHA256 与提交版本**完全相同**（GRU 结果也逐位一致）；
- 日志见 `selfcheck/`。
- 容差：不同 CPU、线程数或 torch/numpy 版本下，GRU/MLP 浮点累加顺序可能变化，预计在 1e-3 量级；物理基线不受影响。
