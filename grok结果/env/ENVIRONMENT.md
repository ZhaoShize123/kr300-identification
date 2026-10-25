# 运行环境（自检时实测）
- OS: Debian GNU/Linux 13 (trixie), kernel 6.12.94+, x86_64, CPU 核数 8
- Python: Python 3.13.5
- 关键包: numpy 2.5.4, scipy 1.18.1, matplotlib 3.11.2, pin(pinocchio) 4.1.0, torch 2.14.1+cpu
- 解压工具: unar (v1.10.8)

完整锁定见 requirements-lock.txt（pip freeze）。torch 需从 CPU 源安装：pip install torch==2.14.1+cpu --index-url https://download.pytorch.org/whl/cpu

注：run.py 设 torch.set_num_threads(8)。不同 CPU/线程数/torch 版本下浮点累加顺序可能不同，GRU 结果可能有 1e-3 量级差异；物理基线（G1）是确定性的。
