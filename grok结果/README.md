# grok结果

这里是在仓库原始代码之上补充的复现记录和三阶段级联实验，由 Grok Bot 在 Linux 机器上完成（Python + numpy，不依赖 sklearn 或 PyTorch）。

## 内容
- `baseline_repro_logs/`：完整跑通仓库原始流程得到的物理基线日志。基线是 PSD 加 J1 事后修正，共 94 个参数。轨迹 12 的 macro RMSE 为 1.0288，各关节为 0.953、1.846、1.334、0.645、0.753、0.642。
- `three_stage/`：冻结式三阶段级联及四组消融，全部为纯逆动力学，不使用实测扭矩历史。
  - `freeze.json` 和 `freeze.sha256`：预注册的配置、数据划分、评价指标和随机种子（sha256 `fba00a67…ece8ed`）。
  - `run.py`：主实验。阶段 1 是 PSD 物理模型加可解释特征；阶段 2 是 numpy 实现的 GRU，只用运动段计算损失，但沿整条轨迹运行，训练后冻结；阶段 3 是只在慢速段训练的残差模型，有 GRU 和滑动历史时间窗两个版本，最终输出 τ = τ2 + m_slow·r3。
  - `analyze.py`：汇总结果，并逐折做配对比较。
  - `results.json`、`summary.txt`、`run.log`：原始结果和运行日志。

## 复现方法
1. 先在仓库根目录按原 README 下载数据，并依次运行 filter_causal → phase_calibration → identify → psd_constrained。
2. 把 psd_constrained 的输出复制为 `final_psd*`。原 README 的运行顺序有误，而且 `final_psd_params.npz` 从未被生成，所以需要这一步。
3. 依次运行 residual_analysis → redo_identify_j1。
4. 运行 `python grok结果/three_stage/run.py`，再运行 `analyze.py`。运行时会生成约 360 MB 的中间文件 `Y_*.npz`，这些文件没有上传。

## 主要结论（交叉验证为轨迹 1–11 的五折；轨迹 12 只做描述，不参与模型选择）

| 组别 | 整体 RMSE | 慢速段 RMSE |
|---|---|---|
| G1 物理 | 1.037 | 2.054 |
| G2 物理+慢速（GRU / 时间窗） | 0.956 / 1.008 | 1.829 / 1.978 |
| G3 物理+阶段 2 | 0.964 | 2.162 |
| G4 全部（GRU / 时间窗） | 0.825 / 0.848 | 1.806 / 1.868 |

- 预注册的判据是整体指标上 G4 逐折都优于 G2。GRU 和时间窗两个版本都 5/5 通过，但这个比较有混杂：G4 的优势里包含了阶段 2 对运动段的修正。
- 事后只比较慢速段：GRU 版本 4/5 折获胜，未通过；时间窗版本 5/5 折获胜，平均差值为 −0.110。若要把它作为正式结论，需要重新冻结一个以慢速段为判据的方案。
- 局限：dwell 层与 slow 层完全相同；慢速阈值 0.05 rad/s 沿用仓库设置；阶段 2 和阶段 3 的训练目标是样本内的。
