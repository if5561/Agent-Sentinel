from __future__ import annotations

from agent_sentinel.main import run


# 根据 __name__ == "__main__" 判断当前流程该进入哪个处理分支。
if __name__ == "__main__":
    # 在输入或运行状态不满足要求时抛出异常，阻止错误结果继续扩散。
    raise SystemExit(run())
