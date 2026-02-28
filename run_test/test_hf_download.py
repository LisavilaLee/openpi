#!/usr/bin/env python3
"""测试从 Hugging Face 下载极小文件，用于验证代理是否正常工作。

用法:
    python test_hf_download.py

如需使用代理，请先设置环境变量，例如:
    export HTTPS_PROXY=http://127.0.0.1:7890
    export HTTP_PROXY=http://127.0.0.1:7890
    python test_hf_download.py
"""

import os
import sys


def main() -> None:
    # 打印当前代理设置（便于排查）
    for name in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy"):
        val = os.environ.get(name)
        if val:
            print(f"  {name}={val}")
    if not any(os.environ.get(n) for n in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy")):
        print("  未检测到代理环境变量 (HTTPS_PROXY/HTTP_PROXY)")

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("错误: 未安装 huggingface_hub。请运行: pip install huggingface_hub")
        sys.exit(1)

    # 使用一个公开的、极小的文件：gpt2 的 config.json（约 1KB）
    repo_id = "gpt2"
    filename = "config.json"
    print(f"\n正在从 Hugging Face 下载: {repo_id}/{filename} ...")

    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="model",
        )
        print(f"✓ 下载成功: {path}")
        with open(path) as f:
            content = f.read()
        print(f"  文件大小: {len(content)} 字节")
        print("\n代理/网络正常，可以访问 Hugging Face。")
    except Exception as e:
        print(f"✗ 下载失败: {e}")
        print("\n若在国内环境，请配置代理后再试 (HTTPS_PROXY/HTTP_PROXY)。")
        sys.exit(1)


if __name__ == "__main__":
    main()
