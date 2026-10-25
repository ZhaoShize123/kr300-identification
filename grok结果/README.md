# grok结果

在仓库原始代码之上补充的**物理基线复现**与**冻结式三阶段级联实验**（纯逆动力学，不使用实测扭矩历史），由 Grok Bot 完成。全部可从空目录一键复现。

## 目录
| 路径 | 内容 |
|---|---|
| [`reproduce.sh`](reproduce.sh) | 一键复现：下载数据 → 物理基线 → 三阶段 → 汇总 |
| [`REPRODUCE.md`](REPRODUCE.md) | 逐步命令、预期数值、SHA256、自检记录 |
| [`docs/CODE_GUIDE.md`](docs/CODE_GUIDE.md) | 逐文件详细代码说明（公式、参数、调用关系图） |
| [`env/`](env/) | 环境锁定：`requirements-lock.txt`（pip freeze）、`ENVIRONMENT.md`（OS/Python） |
| [`conversation/`](conversation/) | 对话记录与 [`INDEX.md`](conversation/INDEX.md)（对话段落 → 文件/提交） |
| [`three_stage/`](three_stage/) | `run.py` 主实验、`analyze.py` 汇总、`freeze.json`+`freeze.sha256` 预注册配置、`results.json`/`summary.txt`/`run.log` 结果 |
| [`baseline_repro_logs/`](baseline_repro_logs/) | 第一次跑通物理基线的 6 份日志 |
| [`selfcheck/`](selfcheck/) | 在全新克隆中运行 reproduce.sh 的日志与 summary |

## 快速复现
```bash
git clone https://github.com/ZhaoShize123/kr300-identification.git
cd kr300-identification
bash grok结果/reproduce.sh      # 需 python3、unar；约 1.5 小时（8 核 CPU）
git diff grok结果/three_stage/summary.txt   # 无输出 = 与提交结果逐字一致
```

## 主要结论
物理基线（PSD + J1 事后修正，94 参数）：轨迹 12 macro RMSE **1.0288**（各关节 0.953 / 1.846 / 1.334 / 0.645 / 0.753 / 0.642）。

三阶段五折 CV（轨迹 1–11 按录制分组；3 个种子平均；轨迹 12 只做描述）：

| 组别 | 整体 RMSE | 慢速段 RMSE | traj12 整体 |
|---|---|---|---|
| G1 物理 | 1.0373 | 2.0544 | 1.0288 |
| G2-GRU 物理+慢速 | 0.9557 | 1.8285 | 0.9577 |
| G2-Win | 1.0077 | 1.9777 | 0.9753 |
| G3 物理+阶段2 | 0.9639 | 2.1621 | 0.9315 |
| G4-GRU 全部 | 0.8251 | 1.8062 | 0.8371 |
| G4-Win | 0.8476 | 1.8676 | 0.8510 |

- 预注册判据（整体，G4 每折优于 G2）：GRU 与时间窗都 5/5 通过；但有混杂——G4 的优势包含阶段 2 对运动段的修正。
- 事后只看慢速段：GRU 4/5（未通过），时间窗 5/5、均值 −0.110。要作为正式结论需重新冻结以慢速段为判据的方案。
- 局限：dwell 层与 slow 层相同；慢速阈值 0.05 rad/s 沿用仓库；阶段 2/3 训练目标为样本内。

## 未入库
`three_stage/Y_*.npz`（约 360 MB，run.py 自动生成）、原始数据 `data/`、根目录被重新生成的 json/npz（与仓库原文件逐位一致或属中间产物）。
