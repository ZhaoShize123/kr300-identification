#!/usr/bin/env bash
# 一键复现：数据下载 → 物理基线 → 三阶段级联 → 汇总 → 校验
# 用法（在仓库根目录或任意位置）： bash grok结果/reproduce.sh
# 可选环境变量：PY=已有 python（默认在仓库根建 .venv 并安装锁定依赖）；SKIP_THREE_STAGE=1 只跑基线
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ROOT="$(dirname "$HERE")"; cd "$ROOT"
LOG="$HERE/selfcheck"; mkdir -p "$LOG"
if [ -z "${PY:-}" ]; then
  if [ ! -x .venv/bin/python ]; then
    python3 -m venv .venv
    .venv/bin/pip install -q numpy==2.5.4 scipy==1.18.1 matplotlib==3.11.2 pin==4.1.0
    TMPDIR="${TMPDIR:-$ROOT}" .venv/bin/pip install -q torch==2.14.1+cpu --index-url https://download.pytorch.org/whl/cpu
  fi
  PY="$ROOT/.venv/bin/python"
fi
echo "[1/5] 下载并校验数据（MD5）"
$PY download_data.py
if [ "$(ls data/raw_data/recording_*.mat 2>/dev/null | wc -l)" -ne 12 ]; then
  mkdir -p data/x data/raw_data
  unar -q -f -o data/x data/Robot_Identification_Benchmark_With_Raw_Data.rar
  find data/x -name 'recording_*.mat' -exec cp {} data/raw_data/ \;
fi
test "$(ls data/raw_data/recording_*.mat | wc -l)" -eq 12
echo "[2/5] 物理基线（顺序已按实际依赖修正）"
for s in filter_causal phase_calibration identify psd_constrained; do
  $PY $s.py > "$LOG/baseline_$s.txt" 2>&1; echo "  $s 完成"; done
# 原 README 的坑：final_psd_params.npz 从未被任何脚本生成；交付的物理模型就是 psd_constrained 的输出
cp psd_constrained_params.npz final_psd_params.npz; cp psd_constrained.json final_psd.json
for s in residual_analysis redo_identify_j1; do   # residual_analysis 生成 residuals.npz，redo_identify_j1 需要它
  $PY $s.py > "$LOG/baseline_$s.txt" 2>&1; echo "  $s 完成"; done
grep -m1 "posthoc     test" "$LOG/baseline_redo_identify_j1.txt"
[ "${SKIP_THREE_STAGE:-0}" = 1 ] && exit 0
echo "[3/5] 校验冻结配置"
( cd "$HERE/three_stage" && grep " freeze.json$" freeze.sha256 | sha256sum -c - )  # run.py 的哈希因路径修正已变，见 REPRODUCE.md
echo "[4/5] 三阶段级联 + 消融（8 线程约 80 分钟）"
$PY "$HERE/three_stage/run.py" | tee "$LOG/run.log"
echo "[5/5] 汇总"
$PY "$HERE/three_stage/analyze.py" > /dev/null
cp "$HERE/three_stage/summary.txt" "$LOG/summary.txt"
git -C "$ROOT" diff --stat -- "grok结果/three_stage/summary.txt" || true
echo "完成。与提交版本对比：git diff grok结果/three_stage/summary.txt（无输出即完全一致）"
