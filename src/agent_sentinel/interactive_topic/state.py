from __future__ import annotations

from typing import TypedDict

from agent_sentinel.graph.state import DiagnosisState


# 定义 TopicFlowState 组件，集中管理这个模块的状态和行为。
class TopicFlowState(TypedDict, total=False):
    """State carried by the interactive topic LangGraph demo workflow."""

    # 执行当前业务步骤，推动流程继续向下游推进。
    query: str
    # 执行当前业务步骤，推动流程继续向下游推进。
    task_id: str
    # 执行当前业务步骤，推动流程继续向下游推进。
    chat_id: str
    # 执行当前业务步骤，推动流程继续向下游推进。
    root_message_id: str
    # 执行当前业务步骤，推动流程继续向下游推进。
    current_node: str
    # 执行当前业务步骤，推动流程继续向下游推进。
    node_result: str
    # 执行当前业务步骤，推动流程继续向下游推进。
    last_action: str
    # 执行当前业务步骤，推动流程继续向下游推进。
    retry_counts: dict[str, int]
    # 执行当前业务步骤，推动流程继续向下游推进。
    node_results: list[dict[str, str]]
    # 执行当前业务步骤，推动流程继续向下游推进。
    diagnosis_state: DiagnosisState
    # 执行当前业务步骤，推动流程继续向下游推进。
    tool_plan: dict[str, object]
    # 执行当前业务步骤，推动流程继续向下游推进。
    tool_results: list[dict[str, object]]
    # 执行当前业务步骤，推动流程继续向下游推进。
    evidence_review: dict[str, object]
    # 执行当前业务步骤，推动流程继续向下游推进。
    final_text: str


# 定义 append_node_result 相关的处理逻辑，供流程或外部调用复用。
def append_node_result(
    # 执行当前业务步骤，推动流程继续向下游推进。
    state: TopicFlowState,
    # 执行当前业务步骤，推动流程继续向下游推进。
    node_name: str,
    # 执行当前业务步骤，推动流程继续向下游推进。
    node_result: str,
# 执行当前业务步骤，推动流程继续向下游推进。
) -> list[dict[str, str]]:
    # 把当前节点的处理结果追加到话题状态里，方便卡片连续展示整条执行轨迹。
    return [
        # 调用 state.get 完成当前步骤需要的业务处理。
        *state.get("node_results", []),
        # 执行当前业务步骤，推动流程继续向下游推进。
        {"node_name": node_name, "result": node_result},
    ]


# 定义 increment_retry 相关的处理逻辑，供流程或外部调用复用。
def increment_retry(state: TopicFlowState, node_name: str) -> dict[str, int]:
    # 记录某个节点已经被用户要求重试多少次，用来控制最大重试次数。
    retry_counts = dict(state.get("retry_counts", {}))
    # 保存当前计算结果，供后续流程判断或组装响应时使用。
    retry_counts[node_name] = retry_counts.get(node_name, 0) + 1
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return retry_counts
