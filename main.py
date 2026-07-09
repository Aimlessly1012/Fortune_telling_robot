"""项目入口：既支持命令行运行，也向其他模块暴露图调用接口。"""

from cli import run_cli
from graph import app, chat, resume_with_birth_info

# 作为模块导入时，仅公开稳定的应用调用接口。
__all__ = ["app", "chat", "resume_with_birth_info"]


if __name__ == "__main__":
    run_cli()
