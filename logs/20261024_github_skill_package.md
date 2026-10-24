# 长任务日志：KR300 skill 形成 + GitHub 上传包整理

- 任务：用户 2026-10-24 要求：形成一个 skill，将所有代码、说明、日志以及对话等文件上传至其 GitHub 仓库
- 时间：2026-10-24 23:26–23:4x CST（本文件随任务完成时间补记）
- 工作目录：`~/workspace/datasets/robot-identification-benchmark/work/`
- 上传包位置：`~/workspace/exports/kr300-identification/`（本地已 git init + git add，尚未 commit/push，原因见“待用户决策”）

## 执行命令（按顺序）

1. 现状摸查：`gh auth status`（未登录）、各工作目录 git remote 检查、custom-skills 目录检查、work 目录文件与大小统计
2. 写 skill：`~/workspace/custom-skills/kr300-identification/SKILL.md`（同时复制进上传包根目录）
3. 组装上传包：复制 12 个 pipeline 脚本、全部 JSON 结果、模型参数 .npz、stage2_models.pt、全部 PNG/SVG 图、logs/*.md 到上传包；写 README.md、DATA.md、requirements.txt、.gitignore、conversation/conversation.md
4. 可移植性补丁（仅改 3 处路径常量，计算未动）：filter_causal.py、phase_calibration.py、dwell_diagnosis.py 的 BASE 改为脚本目录下 data/raw_data，dwell 输出路径改为脚本目录；grep 验证无残留绝对路径
5. 核查：文件清单与大小、敏感信息 grep、排除项确认、全部脚本 py_compile 语法检查

## 关键发现与决定

- gh CLI 已安装但未登录；记忆中无用户 GitHub 用户名/仓库；copp-verify 的 remote 是 TOPP-THU/copp（上游组织仓库，不是目标）——因此本次不猜仓库、不推送。
- 对话导出：Muse 对话在磁盘上没有可直接复制的 transcript 文件，且 chat.read_messages 对长消息截断并混入内部 runtime 消息；因此 conversation.md 是“ curated 记录”（全部用户指令、决策、关键数字、文件指向），并在文件内明示了这一点，而不是一份残缺的逐字 dump。
- 排除项（见 README/DATA.md）：原始 .rar 与 extracted/（217M+13M+234M，且受 GitHub 单文件 100MB 限制）、可再生的 data-bearing .npz（filtered_causal、residuals、residuals_corrected、stage1_e1）、venv、__pycache__。纳入的是脚本、JSON 结果、模型参数、图、日志、对话记录。上传包约 8.0MB。
- git identity：本机无全局 user.name/email；commit 将在用户给出 GitHub 用户名后以其身份进行，本次只做到 git init + git add。

## 产出文件

- skill：`~/workspace/custom-skills/kr300-identification/SKILL.md`
- 上传包：`~/workspace/exports/kr300-identification/`（README.md、DATA.md、SKILL.md、requirements.txt、.gitignore、12 脚本、JSON、参数 .npz、stage2_models.pt、图、logs/、conversation/conversation.md）
- 本日志：`work/logs/20261024_github_skill_package.md`（同时复制进上传包 logs/）

## 待用户决策

- 目标仓库：新建还是现有？仓库名？公开还是私有？
- 授权方式：gh auth login（需用户在浏览器/设备端完成）或已注册的 SSH key
- 仓库 license（尚未选）
