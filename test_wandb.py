#!/usr/bin/env python3
"""测试是否能连接 wandb 并完成初始化（与本项目训练脚本参数一致，含 init_timeout 防超时）。

用法:
    python test_wandb.py

如需使用代理，请先设置环境变量，例如:
    export HTTPS_PROXY=http://127.0.0.1:7890
    export HTTP_PROXY=http://127.0.0.1:7890
    python test_wandb.py
"""

import os
import sys

# 与本项目 scripts/train.py、scripts/train_pytorch.py 保持一致：延长 init 超时，避免 CommError
INIT_TIMEOUT_SEC = 120


def main() -> None:
    # 打印当前代理设置（便于排查）
    for name in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy"):
        val = os.environ.get(name)
        if val:
            print(f"  {name}={val}")
    if not any(os.environ.get(n) for n in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy")):
        print("  未检测到代理环境变量 (HTTPS_PROXY/HTTP_PROXY)")

    try:
        import wandb
    except ImportError:
        print("错误: 未安装 wandb。请运行: pip install wandb")
        sys.exit(1)

    # 与本项目 config 一致：project_name 默认为 "openpi"
    project = "openpi"
    run_name = "openpi_connection_test"
    print(f"\n正在连接 wandb 并初始化 (project={project}, init_timeout={INIT_TIMEOUT_SEC}s) ...")

    try:
        wandb.init(
            name=run_name,
            project=project,
            settings=wandb.Settings(init_timeout=INIT_TIMEOUT_SEC),
        )
        run_id = wandb.run.id
        print(f"✓ wandb 初始化成功: run_id={run_id}")
        wandb.finish()
        print("\nwandb 连接正常，运行本项目训练时可按同样方式使用。")
    except Exception as e:
        print(f"✗ wandb 初始化失败: {e}")
        print("\n若在国内或网络较慢，可先设置代理 (HTTPS_PROXY/HTTP_PROXY)，或增大 init_timeout。")
        sys.exit(1)


if __name__ == "__main__":
    main()
