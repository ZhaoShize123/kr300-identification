# 长任务日志：冻结式两阶段模型（阶段一 motion + 阶段二 slow GRU）

- 任务：按用户 2026-10-24 设计实现 phy → phy+motion → phy+motion+slow，阶段一冻结后训阶段二，附通用残差网络对照
- 时间：2026-10-24 23:01–23:16 CST
- 工作目录：`~/workspace/datasets/robot-identification-benchmark/work/`
- 环境：venv（Python 3.12，pinocchio 来自 pip 包 `pin` 4.1.0）；本任务新装 torch 2.14.1+cpu

## 执行命令（按顺序）

1. `python3 stage1_motion.py`
   显式 ΔM + J1 h(q2)（运动段样本拟合）+ r_fast 轨迹块 CV + 消融表 + SVG 图
2. 扩展 α CV（1000/3000/10000/30000/100000）+ 门控 fast 重拟合（α=10000）
   修正：fast 加运动标志门控，更新 `stage1_e1.npz`（交付版）、`stage1_motion_params.npz`、`stage1_motion.json`，重出 `stage1_ablation.svg`
3. `TMPDIR=~/workspace/tmp_pip PIP_NO_CACHE_DIR=1 venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu --timeout 120 --retries 10`
   见“故障”第 1 条
4. `venv/bin/python stage2_slow.py`（第一次后台运行被 SIGTERM 杀掉，无输出；本轮重跑完成）
   slow GRU + 通用残差网络对照 + 消融表 + SVG 图

## 阶段一结果（已冻结，拟合只用运动段样本，慢速样本未进拟合）

- 显式系数：J1 ΔM11=11.885，h(q2)=0.10+32.07·sin q2+9.73·cos q2；
  J2/J3 块 [-3.723,-2.265]/[-2.499,1.197]（对称共享 -2.308 时指标到 3 位小数不变）；
  J4 0.636（仅 qdd4 项），J5 0.505，J6 0.427
- r_fast：ridge α=10000，录音块留一 CV 运动段 RMSE 0.5661→0.5305（+6.3%，过预设 2% 门槛，纳入）
- 门控修正：不加门控时 fast 使训练集 J2 慢速 RMSE 3.39→3.73 Nm；交付版 r_motion = r_explicit + 运动标志·r_fast
- 消融（平均 RMSE，Nm；训练 / 测试）：
  - phy：总体 1.236/1.205，运动 0.815/0.746，慢速 2.047/2.159
  - phy+motion 显式：总体 1.004/1.003，运动 0.502/0.452，慢速 2.045/2.157
  - phy+motion 交付：总体 0.987/0.991，运动 0.461/0.417，慢速 2.045/2.157

## 阶段二结果（slow GRU，未通过验证）

- 配置：hidden 32，输入 q/qd/qdd + 慢速标志 + dwell 年龄 + 上次运动方向；状态整轨迹更新、损失只计慢速样本、运动段输出门控置零、每轨迹重置状态；早停用训练集最后一个录音（段 60–65）；S=diag(0,1,1,0,1,1)，只输出到 J2/3/5/6
- 验证：slow GRU 最好验证慢速 RMSE 2.868（第 0 epoch），全程未打过只用阶段一的 2.637
- 测试（平均 RMSE，Nm）：phy+motion+slow 总体 1.005、运动 0.417（与阶段一相同，门控有效）、慢速 2.190（阶段一 2.157）；
  慢速分关节：J2 4.021→4.128，J5 1.771→1.900 变差；J3 3.036→3.009，J6 1.494→1.485 微弱变好；J1/J4 按设计无 slow 输出
- 对照通用残差网络（同结构 GRU，e0、全关节、全样本、无门控）：测试总体 1.192、运动 0.733、慢速 2.176——未找回显式运动修正
- 结论：显式运动修正可迁移且本通用网络未复现；慢速残差用现有因果输入 + 本 GRU 配置学不出可迁移映射，可能由数据中没有的状态（摩擦内态/温度/压力）决定

## 故障记录

1. torch 首次安装失败：默认 pip 临时目录在 512M 的 /tmp tmpfs 上，解包时 ENOSPC（另有 download.pytorch.org 读超时重试）。改 TMPDIR 到 home 磁盘后一次成功，torch 2.14.1+cpu。
2. 阶段二首次后台运行在会话压缩后被 SIGTERM 杀掉，无任何输出文件；本轮重跑完成，耗时约 2 分钟内（与另两个后台作业共享 2 核）。

## 产出文件

- `stage1_motion.py`、`stage1_motion.json`、`stage1_motion_params.npz`、`stage1_e1.npz`
- `stage1_motion_rmse.svg`、`stage1_ablation.svg`（门控版）
- `stage2_slow.py`、`stage2_slow.json`、`stage2_models.pt`
- `stage2_ablation.svg`、`stage2_slow_rmse.svg`
- 本日志：`logs/20261024_two_stage_frozen_model.md`

## 待用户决策（2026-10-24 23:16 已给选项，未选）

- 先试线性历史特征 slow 模型 / 换 GRU 配置再试 / 先接受慢速段下限，走 dwell 实验
