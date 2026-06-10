from __future__ import annotations

import os
import shlex
from dataclasses import dataclass, field

from dotenv import load_dotenv


# 定义 _to_bool 相关的处理逻辑，供流程或外部调用复用。
def _to_bool(value: str | None, default: bool = False) -> bool:
    # 把环境变量里的 true/false 字符串转换成布尔开关，没配置时使用默认值。
    if value is None:
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return default
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


# 定义 _to_float 相关的处理逻辑，供流程或外部调用复用。
def _to_float(value: str | None, default: float) -> float:
    # 把环境变量里的小数字符串转换成浮点数，例如温度、权重和阈值。
    if value is None or value == "":
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return default
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return float(value)


# 定义 _to_int 相关的处理逻辑，供流程或外部调用复用。
def _to_int(value: str | None, default: int) -> int:
    # 把环境变量里的数字字符串转换成整数，例如端口、超时和条数限制。
    if value is None or value == "":
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return default
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return int(value)


# 定义 _to_list 相关的处理逻辑，供流程或外部调用复用。
def _to_list(value: str | None) -> list[str]:
    # 把逗号分隔的配置字符串转换成列表，例如模型列表或群白名单。
    if value is None or value.strip() == "":
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return []
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return [item.strip() for item in value.split(",") if item.strip()]


# 定义 _to_args 相关的处理逻辑，供流程或外部调用复用。
def _to_args(value: str | None) -> list[str]:
    # 把 MCP 启动参数解析成列表，兼容逗号写法和命令行参数写法。
    if value is None or value.strip() == "":
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return []
    # MCP 启动参数既可能来自逗号分隔的环境变量，也可能是 shell 风格参数串。
    if "," in value:
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return _to_list(value)
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return shlex.split(value)


# 定义 _first_non_empty 相关的处理逻辑，供流程或外部调用复用。
def _first_non_empty(*values: str | None) -> str | None:
    # 多个兼容环境变量按优先级取第一个非空值，便于兼容 Codex/OpenAI/自定义配置名。
    # 从多个候选配置名中取第一个非空值，用来兼容不同部署环境的命名习惯。
    for value in values:
        # 根据 value is not None and value != "" 判断当前流程该进入哪个处理分支。
        if value is not None and value != "":
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return value
    return None


@dataclass(slots=True)
# 定义 LangfuseConfig 组件，集中管理这个模块的状态和行为。
class LangfuseConfig:
    # 将 enabled 的值保存下来，供后续流程判断或组装响应时使用。
    enabled: bool = False
    # 将 host 的值保存下来，供后续流程判断或组装响应时使用。
    host: str | None = None
    # 将 public_key 的值保存下来，供后续流程判断或组装响应时使用。
    public_key: str | None = None
    # 将 secret_key 的值保存下来，供后续流程判断或组装响应时使用。
    secret_key: str | None = None
    # 将 cache_ttl_seconds 的值保存下来，供后续流程判断或组装响应时使用。
    cache_ttl_seconds: int = 300
    # 将 label 的值保存下来，供后续流程判断或组装响应时使用。
    label: str = "production"
    # 将 link_initial_delay_ms 的值保存下来，供后续流程判断或组装响应时使用。
    link_initial_delay_ms: int = 2000
    # 将 link_poll_interval_ms 的值保存下来，供后续流程判断或组装响应时使用。
    link_poll_interval_ms: int = 2000
    # 将 link_max_retries 的值保存下来，供后续流程判断或组装响应时使用。
    link_max_retries: int = 10
    # 将 link_generations 的值保存下来，供后续流程判断或组装响应时使用。
    link_generations: bool = False
    # 将 remote_prompts_enabled 的值保存下来，供后续流程判断或组装响应时使用。
    remote_prompts_enabled: bool = True
    # 将 prompt_dir 的值保存下来，供后续流程判断或组装响应时使用。
    prompt_dir: str = "config/prompts"
    # 将 trace_prompts 的值保存下来，供后续流程判断或组装响应时使用。
    trace_prompts: bool = True
    # 将 trace_rag 的值保存下来，供后续流程判断或组装响应时使用。
    trace_rag: bool = True
    # 将 trace_tools 的值保存下来，供后续流程判断或组装响应时使用。
    trace_tools: bool = True


@dataclass(slots=True)
# 定义 Settings 组件，集中管理这个模块的状态和行为。
class Settings:
    # 执行当前业务步骤，推动流程继续向下游推进。
    app_name: str
    # 执行当前业务步骤，推动流程继续向下游推进。
    app_host: str
    # 执行当前业务步骤，推动流程继续向下游推进。
    app_port: int
    # 执行当前业务步骤，推动流程继续向下游推进。
    app_env: str
    # 执行当前业务步骤，推动流程继续向下游推进。
    log_level: str
    # 执行当前业务步骤，推动流程继续向下游推进。
    openai_api_key: str
    # 执行当前业务步骤，推动流程继续向下游推进。
    openai_base_url: str | None
    # 执行当前业务步骤，推动流程继续向下游推进。
    openai_model: str
    # 执行当前业务步骤，推动流程继续向下游推进。
    openai_temperature: float
    # 执行当前业务步骤，推动流程继续向下游推进。
    openai_http_trust_env: bool
    # 执行当前业务步骤，推动流程继续向下游推进。
    feishu_api_base_url: str
    # 执行当前业务步骤，推动流程继续向下游推进。
    feishu_app_id: str | None
    # 执行当前业务步骤，推动流程继续向下游推进。
    feishu_app_secret: str | None
    # 执行当前业务步骤，推动流程继续向下游推进。
    feishu_event_verification_token: str | None
    # 执行当前业务步骤，推动流程继续向下游推进。
    feishu_event_encrypt_key: str | None
    # 执行当前业务步骤，推动流程继续向下游推进。
    feishu_bot_name: str
    # 执行当前业务步骤，推动流程继续向下游推进。
    feishu_allowed_chat_ids: list[str]
    # 执行当前业务步骤，推动流程继续向下游推进。
    feishu_analyze_mention_only: bool
    # 执行当前业务步骤，推动流程继续向下游推进。
    feishu_long_connection_enabled: bool
    # 执行当前业务步骤，推动流程继续向下游推进。
    feishu_message_polling_enabled: bool
    # 执行当前业务步骤，推动流程继续向下游推进。
    feishu_message_polling_interval_seconds: int
    # 执行当前业务步骤，推动流程继续向下游推进。
    feishu_message_polling_page_size: int
    # 执行当前业务步骤，推动流程继续向下游推进。
    feishu_webhook_url: str | None
    # 执行当前业务步骤，推动流程继续向下游推进。
    feishu_secret: str | None
    # 执行当前业务步骤，推动流程继续向下游推进。
    feishu_alert_enabled: bool
    # 执行当前业务步骤，推动流程继续向下游推进。
    feishu_alert_title_prefix: str
    # 执行当前业务步骤，推动流程继续向下游推进。
    alert_analysis_enabled: bool
    # 执行当前业务步骤，推动流程继续向下游推进。
    alert_analysis_title_prefix: str
    # 执行当前业务步骤，推动流程继续向下游推进。
    alert_api_token: str | None
    # 执行当前业务步骤，推动流程继续向下游推进。
    alert_dedup_window_seconds: int
    # 执行当前业务步骤，推动流程继续向下游推进。
    alert_store_limit: int
    # 将 metrics_enabled 的值保存下来，供后续流程判断或组装响应时使用。
    metrics_enabled: bool = True
    # 将 metrics_port 的值保存下来，供后续流程判断或组装响应时使用。
    metrics_port: int | None = None
    # 将 aiops_workflow_config_path 的值保存下来，供后续流程判断或组装响应时使用。
    aiops_workflow_config_path: str = "config/workflow.yaml"
    # 将 aiops_llm_models 的值保存下来，供后续流程判断或组装响应时使用。
    aiops_llm_models: list[str] = field(default_factory=lambda: ["gpt-4o-mini"])
    # 将 aiops_llm_timeout_seconds 的值保存下来，供后续流程判断或组装响应时使用。
    aiops_llm_timeout_seconds: int = 15
    # 将 aiops_llm_max_retries 的值保存下来，供后续流程判断或组装响应时使用。
    aiops_llm_max_retries: int = 2
    # 将 aiops_mock_llm_enabled 的值保存下来，供后续流程判断或组装响应时使用。
    aiops_mock_llm_enabled: bool = False
    # 将 aiops_human_confirm_timeout_seconds 的值保存下来，供后续流程判断或组装响应时使用。
    aiops_human_confirm_timeout_seconds: int = 300
    # 将 aiops_human_confirm_enabled 的值保存下来，供后续流程判断或组装响应时使用。
    aiops_human_confirm_enabled: bool = True
    # 将 redis_url 的值保存下来，供后续流程判断或组装响应时使用。
    redis_url: str | None = None
    # 将 rag_provider 的值保存下来，供后续流程判断或组装响应时使用。
    rag_provider: str = "mock"
    # 将 rag_final_top_k 的值保存下来，供后续流程判断或组装响应时使用。
    rag_final_top_k: int = 2
    # 将 rag_mmr_lambda 的值保存下来，供后续流程判断或组装响应时使用。
    rag_mmr_lambda: float = 0.55
    # 将 rag_static_enabled 的值保存下来，供后续流程判断或组装响应时使用。
    rag_static_enabled: bool = True
    # 将 rag_static_collection 的值保存下来，供后续流程判断或组装响应时使用。
    rag_static_collection: str = "aiops_static_docs"
    # 将 rag_static_top_k 的值保存下来，供后续流程判断或组装响应时使用。
    rag_static_top_k: int = 2
    # 将 rag_static_weight 的值保存下来，供后续流程判断或组装响应时使用。
    rag_static_weight: float = 0.55
    # 将 rag_message_enabled 的值保存下来，供后续流程判断或组装响应时使用。
    rag_message_enabled: bool = True
    # 将 rag_message_collection 的值保存下来，供后续流程判断或组装响应时使用。
    rag_message_collection: str = "aiops_message_history"
    # 将 rag_message_top_k 的值保存下来，供后续流程判断或组装响应时使用。
    rag_message_top_k: int = 2
    # 将 rag_message_weight 的值保存下来，供后续流程判断或组装响应时使用。
    rag_message_weight: float = 0.45
    # 将 rag_message_default_days 的值保存下来，供后续流程判断或组装响应时使用。
    rag_message_default_days: int = 30
    # 将 rag_case_cache_enabled 的值保存下来，供后续流程判断或组装响应时使用。
    rag_case_cache_enabled: bool = True
    # 将 rag_case_cache_threshold 的值保存下来，供后续流程判断或组装响应时使用。
    rag_case_cache_threshold: float = 0.85
    # 将 rag_case_cache_top_k 的值保存下来，供后续流程判断或组装响应时使用。
    rag_case_cache_top_k: int = 2
    # 将 rag_feedback_enabled 的值保存下来，供后续流程判断或组装响应时使用。
    rag_feedback_enabled: bool = True
    # 将 rag_pruning_enabled 的值保存下来，供后续流程判断或组装响应时使用。
    rag_pruning_enabled: bool = True
    # 将 rag_history_prune_top_k 的值保存下来，供后续流程判断或组装响应时使用。
    rag_history_prune_top_k: int = 1
    # 将 rag_static_recall_top_k 的值保存下来，供后续流程判断或组装响应时使用。
    rag_static_recall_top_k: int = 10
    # 将 rag_reranker_model_name 的值保存下来，供后续流程判断或组装响应时使用。
    rag_reranker_model_name: str = "qwen3-rerank"
    # 将 rag_reranker_device 的值保存下来，供后续流程判断或组装响应时使用。
    rag_reranker_device: str = "cpu"
    # 将 rag_reranker_api_endpoint 的值保存下来，供后续流程判断或组装响应时使用。
    rag_reranker_api_endpoint: str | None = None
    # 将 rag_reranker_api_key 的值保存下来，供后续流程判断或组装响应时使用。
    rag_reranker_api_key: str | None = None
    # 将 rag_reranker_api_format 的值保存下来，供后续流程判断或组装响应时使用。
    rag_reranker_api_format: str = "auto"
    # 将 rag_reranker_timeout_seconds 的值保存下来，供后续流程判断或组装响应时使用。
    rag_reranker_timeout_seconds: int = 15
    # 将 milvus_uri 的值保存下来，供后续流程判断或组装响应时使用。
    milvus_uri: str | None = None
    # 将 milvus_token 的值保存下来，供后续流程判断或组装响应时使用。
    milvus_token: str | None = None
    # 将 milvus_user 的值保存下来，供后续流程判断或组装响应时使用。
    milvus_user: str | None = None
    # 将 milvus_password 的值保存下来，供后续流程判断或组装响应时使用。
    milvus_password: str | None = None
    # 将 milvus_db_name 的值保存下来，供后续流程判断或组装响应时使用。
    milvus_db_name: str | None = None
    # 将 embedding_api_key 的值保存下来，供后续流程判断或组装响应时使用。
    embedding_api_key: str | None = None
    # 将 embedding_base_url 的值保存下来，供后续流程判断或组装响应时使用。
    embedding_base_url: str | None = None
    # 将 embedding_model 的值保存下来，供后续流程判断或组装响应时使用。
    embedding_model: str = "text-embedding-3-small"
    # 将 embedding_dimension 的值保存下来，供后续流程判断或组装响应时使用。
    embedding_dimension: int = 1536
    # 将 embedding_mock_enabled 的值保存下来，供后续流程判断或组装响应时使用。
    embedding_mock_enabled: bool = False
    # 将 interactive_topic_enabled 的值保存下来，供后续流程判断或组装响应时使用。
    interactive_topic_enabled: bool = True
    # 将 interactive_topic_wait_seconds 的值保存下来，供后续流程判断或组装响应时使用。
    interactive_topic_wait_seconds: int = 5

    # Tools provider 配置
    tools_provider: str = "mock"  # mock | aliyun | mcp
    # 将 tools_metrics_enabled 的值保存下来，供后续流程判断或组装响应时使用。
    tools_metrics_enabled: bool = True
    # 将 tools_logs_enabled 的值保存下来，供后续流程判断或组装响应时使用。
    tools_logs_enabled: bool = True
    # 将 tools_topology_enabled 的值保存下来，供后续流程判断或组装响应时使用。
    tools_topology_enabled: bool = True
    # 将 mcp_enabled 的值保存下来，供后续流程判断或组装响应时使用。
    mcp_enabled: bool = False
    # 将 mcp_server_command 的值保存下来，供后续流程判断或组装响应时使用。
    mcp_server_command: str = ""
    # 将 mcp_server_args 的值保存下来，供后续流程判断或组装响应时使用。
    mcp_server_args: list[str] = field(default_factory=list)
    # 将 mcp_timeout_seconds 的值保存下来，供后续流程判断或组装响应时使用。
    mcp_timeout_seconds: int = 20
    # 将 mcp_sls_tool 的值保存下来，供后续流程判断或组装响应时使用。
    mcp_sls_tool: str = "aliyun_sls_query_logs"
    # 将 mcp_prometheus_tool 的值保存下来，供后续流程判断或组装响应时使用。
    mcp_prometheus_tool: str = "prometheus_query_metrics"
    # 将 mcp_prometheus_base_url 的值保存下来，供后续流程判断或组装响应时使用。
    mcp_prometheus_base_url: str | None = None

    # 阿里云 SLS 配置
    aliyun_sls_access_key_id: str | None = None
    # 将 aliyun_sls_access_key_secret 的值保存下来，供后续流程判断或组装响应时使用。
    aliyun_sls_access_key_secret: str | None = None
    # 将 aliyun_sls_endpoint 的值保存下来，供后续流程判断或组装响应时使用。
    aliyun_sls_endpoint: str = "cn-hangzhou.log.aliyuncs.com"
    # 将 aliyun_sls_project 的值保存下来，供后续流程判断或组装响应时使用。
    aliyun_sls_project: str = ""
    # 将 aliyun_sls_logstore 的值保存下来，供后续流程判断或组装响应时使用。
    aliyun_sls_logstore: str = ""
    # 将 aliyun_sls_query_timeout_seconds 的值保存下来，供后续流程判断或组装响应时使用。
    aliyun_sls_query_timeout_seconds: int = 10
    # 将 aliyun_sls_max_lines 的值保存下来，供后续流程判断或组装响应时使用。
    aliyun_sls_max_lines: int = 100

    # 阿里云 ARMS 配置
    aliyun_arms_access_key_id: str | None = None
    # 将 aliyun_arms_access_key_secret 的值保存下来，供后续流程判断或组装响应时使用。
    aliyun_arms_access_key_secret: str | None = None
    # 将 aliyun_arms_region_id 的值保存下来，供后续流程判断或组装响应时使用。
    aliyun_arms_region_id: str = "cn-hangzhou"
    # 将 aliyun_arms_app_id 的值保存下来，供后续流程判断或组装响应时使用。
    aliyun_arms_app_id: str = ""
    # 将 aliyun_arms_query_timeout_seconds 的值保存下来，供后续流程判断或组装响应时使用。
    aliyun_arms_query_timeout_seconds: int = 10
    # 将 langfuse 的值保存下来，供后续流程判断或组装响应时使用。
    langfuse: LangfuseConfig = field(default_factory=LangfuseConfig)


# 定义 get_settings 相关的处理逻辑，供流程或外部调用复用。
def get_settings() -> Settings:
    # 汇总 YAML、.env 和系统环境变量，生成整个应用运行时使用的统一配置对象。
    load_dotenv()
    # settings.yaml 提供默认结构化配置，环境变量用于部署时覆盖敏感项和环境差异。
    yaml_settings = _load_yaml_settings()
    # 将 app_cfg 的值保存下来，供后续流程判断或组装响应时使用。
    app_cfg = yaml_settings.get("app", {})
    # 将 llm_cfg 的值保存下来，供后续流程判断或组装响应时使用。
    llm_cfg = yaml_settings.get("llm", {})
    # 将 workflow_cfg 的值保存下来，供后续流程判断或组装响应时使用。
    workflow_cfg = yaml_settings.get("workflow", {})
    # 将 redis_cfg 的值保存下来，供后续流程判断或组装响应时使用。
    redis_cfg = yaml_settings.get("redis", {})
    # 将 rag_cfg 的值保存下来，供后续流程判断或组装响应时使用。
    rag_cfg = yaml_settings.get("rag", {})
    # 将 static_cfg 的值保存下来，供后续流程判断或组装响应时使用。
    static_cfg = dict(rag_cfg.get("static_docs", {})) if isinstance(rag_cfg.get("static_docs", {}), dict) else {}
    # 将 message_cfg 的值保存下来，供后续流程判断或组装响应时使用。
    message_cfg = (
        # 调用 dict 完成当前步骤需要的业务处理。
        dict(rag_cfg.get("message_history", {}))
        # 根据 isinstance(rag_cfg.get("message_history", {}), dict) 判断当前流程该进入哪个处理分支。
        if isinstance(rag_cfg.get("message_history", {}), dict)
        # 执行当前业务步骤，推动流程继续向下游推进。
        else {}
    )
    # 子配置统一转成 dict，避免 YAML 写错类型时在后续 .get 调用处抛异常。
    embedding_cfg = yaml_settings.get("embedding", {})
    # 将 tools_cfg 的值保存下来，供后续流程判断或组装响应时使用。
    tools_cfg = yaml_settings.get("tools", {}) if isinstance(yaml_settings.get("tools", {}), dict) else {}
    # 将 mcp_cfg 的值保存下来，供后续流程判断或组装响应时使用。
    mcp_cfg = tools_cfg.get("mcp", {}) if isinstance(tools_cfg.get("mcp", {}), dict) else {}
    # 将 sls_cfg 的值保存下来，供后续流程判断或组装响应时使用。
    sls_cfg = tools_cfg.get("aliyun_sls", {}) if isinstance(tools_cfg.get("aliyun_sls", {}), dict) else {}
    # 将 arms_cfg 的值保存下来，供后续流程判断或组装响应时使用。
    arms_cfg = tools_cfg.get("aliyun_arms", {}) if isinstance(tools_cfg.get("aliyun_arms", {}), dict) else {}
    # 将 langfuse_cfg 的值保存下来，供后续流程判断或组装响应时使用。
    langfuse_cfg = yaml_settings.get("langfuse", {}) if isinstance(yaml_settings.get("langfuse", {}), dict) else {}
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return Settings(
        # 将 app_name 的值保存下来，供后续流程判断或组装响应时使用。
        app_name=os.getenv("APP_NAME", str(app_cfg.get("name", "Agent Sentinel"))),
        # 将 app_host 的值保存下来，供后续流程判断或组装响应时使用。
        app_host=os.getenv("APP_HOST", str(app_cfg.get("host", "0.0.0.0"))),
        # 将 app_port 的值保存下来，供后续流程判断或组装响应时使用。
        app_port=_to_int(os.getenv("APP_PORT"), int(app_cfg.get("port", 8000))),
        # 将 app_env 的值保存下来，供后续流程判断或组装响应时使用。
        app_env=os.getenv("APP_ENV", str(app_cfg.get("env", "dev"))),
        # 将 log_level 的值保存下来，供后续流程判断或组装响应时使用。
        log_level=os.getenv("LOG_LEVEL", str(app_cfg.get("log_level", "INFO"))),
        # 将 openai_api_key 的值保存下来，供后续流程判断或组装响应时使用。
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        # 将 openai_base_url 的值保存下来，供后续流程判断或组装响应时使用。
        openai_base_url=_first_non_empty(
            # 兼容不同运行环境中常见的 base_url 命名。
            os.getenv("CODEX_BASE_URL"),
            # 调用 os.getenv 完成当前步骤需要的业务处理。
            os.getenv("OPENAI_BASE_URL"),
            # 调用 os.getenv 完成当前步骤需要的业务处理。
            os.getenv("base_url"),
        ),
        # 将 openai_model 的值保存下来，供后续流程判断或组装响应时使用。
        openai_model=_first_non_empty(
            # 调用 os.getenv 完成当前步骤需要的业务处理。
            os.getenv("CODEX_MODEL"),
            # 调用 os.getenv 完成当前步骤需要的业务处理。
            os.getenv("OPENAI_MODEL"),
            # 调用 os.getenv 完成当前步骤需要的业务处理。
            os.getenv("model"),
        )
        # 执行当前业务步骤，推动流程继续向下游推进。
        or "gpt-4o-mini",
        # 将 openai_temperature 的值保存下来，供后续流程判断或组装响应时使用。
        openai_temperature=_to_float(os.getenv("OPENAI_TEMPERATURE"), 0.0),
        # 将 openai_http_trust_env 的值保存下来，供后续流程判断或组装响应时使用。
        openai_http_trust_env=_to_bool(os.getenv("OPENAI_HTTP_TRUST_ENV"), False),
        # 将 feishu_api_base_url 的值保存下来，供后续流程判断或组装响应时使用。
        feishu_api_base_url=os.getenv("FEISHU_API_BASE_URL", "https://open.feishu.cn"),
        # 将 feishu_app_id 的值保存下来，供后续流程判断或组装响应时使用。
        feishu_app_id=os.getenv("FEISHU_APP_ID"),
        # 将 feishu_app_secret 的值保存下来，供后续流程判断或组装响应时使用。
        feishu_app_secret=os.getenv("FEISHU_APP_SECRET"),
        # 将 feishu_event_verification_token 的值保存下来，供后续流程判断或组装响应时使用。
        feishu_event_verification_token=os.getenv("FEISHU_EVENT_VERIFICATION_TOKEN"),
        # 将 feishu_event_encrypt_key 的值保存下来，供后续流程判断或组装响应时使用。
        feishu_event_encrypt_key=os.getenv("FEISHU_EVENT_ENCRYPT_KEY"),
        # 将 feishu_bot_name 的值保存下来，供后续流程判断或组装响应时使用。
        feishu_bot_name=os.getenv("FEISHU_BOT_NAME", "Analysis Bot"),
        # 将 feishu_allowed_chat_ids 的值保存下来，供后续流程判断或组装响应时使用。
        feishu_allowed_chat_ids=_to_list(os.getenv("FEISHU_ALLOWED_CHAT_IDS")),
        # 将 feishu_analyze_mention_only 的值保存下来，供后续流程判断或组装响应时使用。
        feishu_analyze_mention_only=_to_bool(os.getenv("FEISHU_ANALYZE_MENTION_ONLY"), True),
        # 将 feishu_long_connection_enabled 的值保存下来，供后续流程判断或组装响应时使用。
        feishu_long_connection_enabled=_to_bool(os.getenv("FEISHU_LONG_CONNECTION_ENABLED"), True),
        # 将 feishu_message_polling_enabled 的值保存下来，供后续流程判断或组装响应时使用。
        feishu_message_polling_enabled=_to_bool(os.getenv("FEISHU_MESSAGE_POLLING_ENABLED"), False),
        # 将 feishu_message_polling_interval_seconds 的值保存下来，供后续流程判断或组装响应时使用。
        feishu_message_polling_interval_seconds=_to_int(
            # 调用 os.getenv 完成当前步骤需要的业务处理。
            os.getenv("FEISHU_MESSAGE_POLLING_INTERVAL_SECONDS"),
            # 执行当前业务步骤，推动流程继续向下游推进。
            5,
        ),
        # 将 feishu_message_polling_page_size 的值保存下来，供后续流程判断或组装响应时使用。
        feishu_message_polling_page_size=_to_int(
            # 调用 os.getenv 完成当前步骤需要的业务处理。
            os.getenv("FEISHU_MESSAGE_POLLING_PAGE_SIZE"),
            # 执行当前业务步骤，推动流程继续向下游推进。
            20,
        ),
        # 将 feishu_webhook_url 的值保存下来，供后续流程判断或组装响应时使用。
        feishu_webhook_url=os.getenv("FEISHU_WEBHOOK_URL"),
        # 将 feishu_secret 的值保存下来，供后续流程判断或组装响应时使用。
        feishu_secret=os.getenv("FEISHU_SECRET"),
        # 将 feishu_alert_enabled 的值保存下来，供后续流程判断或组装响应时使用。
        feishu_alert_enabled=_to_bool(os.getenv("FEISHU_ALERT_ENABLED"), True),
        # 将 feishu_alert_title_prefix 的值保存下来，供后续流程判断或组装响应时使用。
        feishu_alert_title_prefix=os.getenv("FEISHU_ALERT_TITLE_PREFIX", "Agent Sentinel"),
        # 将 alert_analysis_enabled 的值保存下来，供后续流程判断或组装响应时使用。
        alert_analysis_enabled=_to_bool(os.getenv("ALERT_ANALYSIS_ENABLED"), True),
        # 将 alert_analysis_title_prefix 的值保存下来，供后续流程判断或组装响应时使用。
        alert_analysis_title_prefix=os.getenv("ALERT_ANALYSIS_TITLE_PREFIX", "Alert Analysis"),
        # 将 alert_api_token 的值保存下来，供后续流程判断或组装响应时使用。
        alert_api_token=os.getenv("ALERT_API_TOKEN"),
        # 将 alert_dedup_window_seconds 的值保存下来，供后续流程判断或组装响应时使用。
        alert_dedup_window_seconds=_to_int(os.getenv("ALERT_DEDUP_WINDOW_SECONDS"), 60),
        # 将 alert_store_limit 的值保存下来，供后续流程判断或组装响应时使用。
        alert_store_limit=_to_int(os.getenv("ALERT_STORE_LIMIT"), 100),
        # 将 metrics_enabled 的值保存下来，供后续流程判断或组装响应时使用。
        metrics_enabled=_to_bool(os.getenv("METRICS_ENABLED"), True),
        # 将 metrics_port 的值保存下来，供后续流程判断或组装响应时使用。
        metrics_port=_to_int(os.getenv("METRICS_PORT"), 0) or None,
        # 将 aiops_workflow_config_path 的值保存下来，供后续流程判断或组装响应时使用。
        aiops_workflow_config_path=os.getenv(
            # 执行当前业务步骤，推动流程继续向下游推进。
            "AIOPS_WORKFLOW_CONFIG_PATH",
            # 调用 str 完成当前步骤需要的业务处理。
            str(workflow_cfg.get("config_path", "config/workflow.yaml")),
        ),
        # 将 aiops_llm_models 的值保存下来，供后续流程判断或组装响应时使用。
        aiops_llm_models=_to_list(os.getenv("AIOPS_LLM_MODELS"))
        # 调用 str 完成当前步骤需要的业务处理。
        or [str(item) for item in llm_cfg.get("models", [])]
        # 未显式配置多模型时，退回到单模型配置，保证诊断流程总有可用模型名。
        or [_first_non_empty(os.getenv("CODEX_MODEL"), os.getenv("OPENAI_MODEL")) or "gpt-4o-mini"],
        # 将 aiops_llm_timeout_seconds 的值保存下来，供后续流程判断或组装响应时使用。
        aiops_llm_timeout_seconds=_to_int(
            # 调用 os.getenv 完成当前步骤需要的业务处理。
            os.getenv("AIOPS_LLM_TIMEOUT_SECONDS"),
            # 调用 int 完成当前步骤需要的业务处理。
            int(llm_cfg.get("timeout_seconds", 15)),
        ),
        # 将 aiops_llm_max_retries 的值保存下来，供后续流程判断或组装响应时使用。
        aiops_llm_max_retries=_to_int(
            # 调用 os.getenv 完成当前步骤需要的业务处理。
            os.getenv("AIOPS_LLM_MAX_RETRIES"),
            # 调用 int 完成当前步骤需要的业务处理。
            int(llm_cfg.get("max_retries", 2)),
        ),
        # 将 aiops_mock_llm_enabled 的值保存下来，供后续流程判断或组装响应时使用。
        aiops_mock_llm_enabled=_to_bool(
            # 调用 os.getenv 完成当前步骤需要的业务处理。
            os.getenv("AIOPS_MOCK_LLM_ENABLED"),
            # 调用 bool 完成当前步骤需要的业务处理。
            bool(llm_cfg.get("mock_enabled", False)),
        ),
        # 将 aiops_human_confirm_timeout_seconds 的值保存下来，供后续流程判断或组装响应时使用。
        aiops_human_confirm_timeout_seconds=_to_int(
            # 调用 os.getenv 完成当前步骤需要的业务处理。
            os.getenv("AIOPS_HUMAN_CONFIRM_TIMEOUT_SECONDS"),
            # 调用 int 完成当前步骤需要的业务处理。
            int(workflow_cfg.get("human_confirm_timeout_seconds", 300)),
        ),
        # 将 aiops_human_confirm_enabled 的值保存下来，供后续流程判断或组装响应时使用。
        aiops_human_confirm_enabled=_to_bool(
            # 调用 os.getenv 完成当前步骤需要的业务处理。
            os.getenv("AIOPS_HUMAN_CONFIRM_ENABLED"),
            # 调用 bool 完成当前步骤需要的业务处理。
            bool(workflow_cfg.get("human_confirm_enabled", True)),
        ),
        # 将 redis_url 的值保存下来，供后续流程判断或组装响应时使用。
        redis_url=os.getenv("REDIS_URL", str(redis_cfg.get("url", "")) or None),
        # 将 rag_provider 的值保存下来，供后续流程判断或组装响应时使用。
        rag_provider=os.getenv("RAG_PROVIDER", str(rag_cfg.get("provider", "mock"))),
        # 将 rag_final_top_k 的值保存下来，供后续流程判断或组装响应时使用。
        rag_final_top_k=_to_int(os.getenv("RAG_FINAL_TOP_K"), int(rag_cfg.get("final_top_k", 2))),
        # 将 rag_mmr_lambda 的值保存下来，供后续流程判断或组装响应时使用。
        rag_mmr_lambda=_to_float(os.getenv("RAG_MMR_LAMBDA"), float(rag_cfg.get("mmr_lambda", 0.55))),
        # 将 rag_static_enabled 的值保存下来，供后续流程判断或组装响应时使用。
        rag_static_enabled=_to_bool(
            # 调用 os.getenv 完成当前步骤需要的业务处理。
            os.getenv("RAG_STATIC_ENABLED"),
            # 调用 bool 完成当前步骤需要的业务处理。
            bool(static_cfg.get("enabled", True)),
        ),
        # 将 rag_static_collection 的值保存下来，供后续流程判断或组装响应时使用。
        rag_static_collection=os.getenv(
            # 执行当前业务步骤，推动流程继续向下游推进。
            "RAG_STATIC_COLLECTION",
            # 调用 str 完成当前步骤需要的业务处理。
            str(static_cfg.get("collection", "aiops_static_docs")),
        ),
        # 将 rag_static_top_k 的值保存下来，供后续流程判断或组装响应时使用。
        rag_static_top_k=_to_int(os.getenv("RAG_STATIC_TOP_K"), int(static_cfg.get("top_k", 2))),
        # 将 rag_static_weight 的值保存下来，供后续流程判断或组装响应时使用。
        rag_static_weight=_to_float(os.getenv("RAG_STATIC_WEIGHT"), float(static_cfg.get("weight", 0.55))),
        # 将 rag_message_enabled 的值保存下来，供后续流程判断或组装响应时使用。
        rag_message_enabled=_to_bool(
            # 调用 os.getenv 完成当前步骤需要的业务处理。
            os.getenv("RAG_MESSAGE_ENABLED"),
            # 调用 bool 完成当前步骤需要的业务处理。
            bool(message_cfg.get("enabled", True)),
        ),
        # 将 rag_message_collection 的值保存下来，供后续流程判断或组装响应时使用。
        rag_message_collection=os.getenv(
            # 执行当前业务步骤，推动流程继续向下游推进。
            "RAG_MESSAGE_COLLECTION",
            # 调用 str 完成当前步骤需要的业务处理。
            str(message_cfg.get("collection", "aiops_message_history")),
        ),
        # 将 rag_message_top_k 的值保存下来，供后续流程判断或组装响应时使用。
        rag_message_top_k=_to_int(os.getenv("RAG_MESSAGE_TOP_K"), int(message_cfg.get("top_k", 2))),
        # 将 rag_message_weight 的值保存下来，供后续流程判断或组装响应时使用。
        rag_message_weight=_to_float(os.getenv("RAG_MESSAGE_WEIGHT"), float(message_cfg.get("weight", 0.45))),
        # 将 rag_message_default_days 的值保存下来，供后续流程判断或组装响应时使用。
        rag_message_default_days=_to_int(
            # 调用 os.getenv 完成当前步骤需要的业务处理。
            os.getenv("RAG_MESSAGE_DEFAULT_DAYS"),
            # 调用 int 完成当前步骤需要的业务处理。
            int(message_cfg.get("default_days", 30)),
        ),
        # 将 rag_case_cache_enabled 的值保存下来，供后续流程判断或组装响应时使用。
        rag_case_cache_enabled=_to_bool(
            # 调用 os.getenv 完成当前步骤需要的业务处理。
            os.getenv("RAG_CASE_CACHE_ENABLED"),
            # 调用 bool 完成当前步骤需要的业务处理。
            bool(message_cfg.get("case_cache_enabled", True)),
        ),
        # 将 rag_case_cache_threshold 的值保存下来，供后续流程判断或组装响应时使用。
        rag_case_cache_threshold=_to_float(
            # 调用 os.getenv 完成当前步骤需要的业务处理。
            os.getenv("RAG_CASE_CACHE_THRESHOLD"),
            # 调用 float 完成当前步骤需要的业务处理。
            float(message_cfg.get("case_cache_threshold", 0.85)),
        ),
        # 将 rag_case_cache_top_k 的值保存下来，供后续流程判断或组装响应时使用。
        rag_case_cache_top_k=_to_int(
            # 调用 os.getenv 完成当前步骤需要的业务处理。
            os.getenv("RAG_CASE_CACHE_TOP_K"),
            # 调用 int 完成当前步骤需要的业务处理。
            int(message_cfg.get("case_cache_top_k", 2)),
        ),
        # 将 rag_feedback_enabled 的值保存下来，供后续流程判断或组装响应时使用。
        rag_feedback_enabled=_to_bool(
            # 调用 os.getenv 完成当前步骤需要的业务处理。
            os.getenv("RAG_FEEDBACK_ENABLED"),
            # 调用 bool 完成当前步骤需要的业务处理。
            bool(message_cfg.get("feedback_enabled", True)),
        ),
        # 将 rag_pruning_enabled 的值保存下来，供后续流程判断或组装响应时使用。
        rag_pruning_enabled=_to_bool(os.getenv("RAG_PRUNING_ENABLED"), True),
        # 将 rag_history_prune_top_k 的值保存下来，供后续流程判断或组装响应时使用。
        rag_history_prune_top_k=_to_int(os.getenv("RAG_HISTORY_PRUNE_TOP_K"), 1),
        # 将 rag_static_recall_top_k 的值保存下来，供后续流程判断或组装响应时使用。
        rag_static_recall_top_k=_to_int(os.getenv("RAG_STATIC_RECALL_TOP_K"), 10),
        # 将 rag_reranker_model_name 的值保存下来，供后续流程判断或组装响应时使用。
        rag_reranker_model_name=os.getenv("RAG_RERANKER_MODEL_NAME", "qwen3-rerank"),
        # 将 rag_reranker_device 的值保存下来，供后续流程判断或组装响应时使用。
        rag_reranker_device=os.getenv("RAG_RERANKER_DEVICE", "cpu"),
        # 将 rag_reranker_api_endpoint 的值保存下来，供后续流程判断或组装响应时使用。
        rag_reranker_api_endpoint=os.getenv("RAG_RERANKER_API_ENDPOINT"),
        # 将 rag_reranker_api_key 的值保存下来，供后续流程判断或组装响应时使用。
        rag_reranker_api_key=os.getenv("RAG_RERANKER_API_KEY"),
        # 将 rag_reranker_api_format 的值保存下来，供后续流程判断或组装响应时使用。
        rag_reranker_api_format=os.getenv("RAG_RERANKER_API_FORMAT", "auto"),
        # 将 rag_reranker_timeout_seconds 的值保存下来，供后续流程判断或组装响应时使用。
        rag_reranker_timeout_seconds=_to_int(os.getenv("RAG_RERANKER_TIMEOUT_SECONDS"), 15),
        # 将 milvus_uri 的值保存下来，供后续流程判断或组装响应时使用。
        milvus_uri=os.getenv("MILVUS_URI"),
        # 将 milvus_token 的值保存下来，供后续流程判断或组装响应时使用。
        milvus_token=os.getenv("MILVUS_TOKEN"),
        # 将 milvus_user 的值保存下来，供后续流程判断或组装响应时使用。
        milvus_user=os.getenv("MILVUS_USER"),
        # 将 milvus_password 的值保存下来，供后续流程判断或组装响应时使用。
        milvus_password=os.getenv("MILVUS_PASSWORD"),
        # 将 milvus_db_name 的值保存下来，供后续流程判断或组装响应时使用。
        milvus_db_name=os.getenv("MILVUS_DB_NAME"),
        # 将 embedding_api_key 的值保存下来，供后续流程判断或组装响应时使用。
        embedding_api_key=_first_non_empty(os.getenv("EMBEDDING_API_KEY"), os.getenv("OPENAI_API_KEY")),
        # 将 embedding_base_url 的值保存下来，供后续流程判断或组装响应时使用。
        embedding_base_url=_first_non_empty(os.getenv("EMBEDDING_BASE_URL"), os.getenv("OPENAI_BASE_URL")),
        # 将 embedding_model 的值保存下来，供后续流程判断或组装响应时使用。
        embedding_model=os.getenv(
            # 执行当前业务步骤，推动流程继续向下游推进。
            "EMBEDDING_MODEL",
            # 调用 str 完成当前步骤需要的业务处理。
            str(embedding_cfg.get("model", "text-embedding-3-small")),
        ),
        # 将 embedding_dimension 的值保存下来，供后续流程判断或组装响应时使用。
        embedding_dimension=_to_int(
            # 调用 os.getenv 完成当前步骤需要的业务处理。
            os.getenv("EMBEDDING_DIMENSION"),
            # 调用 int 完成当前步骤需要的业务处理。
            int(embedding_cfg.get("dimension", 1536)),
        ),
        # 将 embedding_mock_enabled 的值保存下来，供后续流程判断或组装响应时使用。
        embedding_mock_enabled=_to_bool(
            # 调用 os.getenv 完成当前步骤需要的业务处理。
            os.getenv("EMBEDDING_MOCK_ENABLED"),
            # 调用 bool 完成当前步骤需要的业务处理。
            bool(embedding_cfg.get("mock_enabled", False)),
        ),
        # 将 interactive_topic_enabled 的值保存下来，供后续流程判断或组装响应时使用。
        interactive_topic_enabled=_to_bool(os.getenv("INTERACTIVE_TOPIC_ENABLED"), True),
        # 将 interactive_topic_wait_seconds 的值保存下来，供后续流程判断或组装响应时使用。
        interactive_topic_wait_seconds=_to_int(os.getenv("INTERACTIVE_TOPIC_WAIT_SECONDS"), 5),
        # Tools provider 配置
        tools_provider=os.getenv("TOOLS_PROVIDER", str(tools_cfg.get("provider", "mock"))),
        # 将 tools_metrics_enabled 的值保存下来，供后续流程判断或组装响应时使用。
        tools_metrics_enabled=_to_bool(
            # 调用 os.getenv 完成当前步骤需要的业务处理。
            os.getenv("TOOLS_METRICS_ENABLED"),
            # 调用 bool 完成当前步骤需要的业务处理。
            bool(tools_cfg.get("metrics_enabled", True)),
        ),
        # 将 tools_logs_enabled 的值保存下来，供后续流程判断或组装响应时使用。
        tools_logs_enabled=_to_bool(
            # 调用 os.getenv 完成当前步骤需要的业务处理。
            os.getenv("TOOLS_LOGS_ENABLED"),
            # 调用 bool 完成当前步骤需要的业务处理。
            bool(tools_cfg.get("logs_enabled", True)),
        ),
        # 将 tools_topology_enabled 的值保存下来，供后续流程判断或组装响应时使用。
        tools_topology_enabled=_to_bool(
            # 调用 os.getenv 完成当前步骤需要的业务处理。
            os.getenv("TOOLS_TOPOLOGY_ENABLED"),
            # 调用 bool 完成当前步骤需要的业务处理。
            bool(tools_cfg.get("topology_enabled", True)),
        ),
        # 阿里云 SLS 配置
        mcp_enabled=_to_bool(os.getenv("MCP_ENABLED"), bool(mcp_cfg.get("enabled", False))),
        # 将 mcp_server_command 的值保存下来，供后续流程判断或组装响应时使用。
        mcp_server_command=os.getenv("MCP_SERVER_COMMAND", str(mcp_cfg.get("server_command", ""))),
        # MCP 参数需要支持两种来源：YAML 原生列表和环境变量字符串。
        mcp_server_args=_to_args(os.getenv("MCP_SERVER_ARGS"))
        # 调用 str 完成当前步骤需要的业务处理。
        or [str(item) for item in mcp_cfg.get("server_args", [])],
        # 将 mcp_timeout_seconds 的值保存下来，供后续流程判断或组装响应时使用。
        mcp_timeout_seconds=_to_int(os.getenv("MCP_TIMEOUT_SECONDS"), int(mcp_cfg.get("timeout_seconds", 20))),
        # 将 mcp_sls_tool 的值保存下来，供后续流程判断或组装响应时使用。
        mcp_sls_tool=os.getenv("MCP_SLS_TOOL", str(mcp_cfg.get("sls_tool", "aliyun_sls_query_logs"))),
        # 将 mcp_prometheus_tool 的值保存下来，供后续流程判断或组装响应时使用。
        mcp_prometheus_tool=os.getenv(
            # 执行当前业务步骤，推动流程继续向下游推进。
            "MCP_PROMETHEUS_TOOL",
            # 调用 str 完成当前步骤需要的业务处理。
            str(mcp_cfg.get("prometheus_tool", "prometheus_query_metrics")),
        ),
        # 将 mcp_prometheus_base_url 的值保存下来，供后续流程判断或组装响应时使用。
        mcp_prometheus_base_url=_first_non_empty(
            # 调用 os.getenv 完成当前步骤需要的业务处理。
            os.getenv("MCP_PROMETHEUS_BASE_URL"),
            # 调用 str 完成当前步骤需要的业务处理。
            str(mcp_cfg.get("prometheus_base_url", "")) or None,
        ),
        # 将 aliyun_sls_access_key_id 的值保存下来，供后续流程判断或组装响应时使用。
        aliyun_sls_access_key_id=os.getenv("ALIYUN_SLS_ACCESS_KEY_ID", sls_cfg.get("access_key_id")),
        # 将 aliyun_sls_access_key_secret 的值保存下来，供后续流程判断或组装响应时使用。
        aliyun_sls_access_key_secret=os.getenv("ALIYUN_SLS_ACCESS_KEY_SECRET", sls_cfg.get("access_key_secret")),
        # 将 aliyun_sls_endpoint 的值保存下来，供后续流程判断或组装响应时使用。
        aliyun_sls_endpoint=os.getenv(
            # 执行当前业务步骤，推动流程继续向下游推进。
            "ALIYUN_SLS_ENDPOINT",
            # 调用 str 完成当前步骤需要的业务处理。
            str(sls_cfg.get("endpoint", "cn-hangzhou.log.aliyuncs.com")),
        ),
        # 将 aliyun_sls_project 的值保存下来，供后续流程判断或组装响应时使用。
        aliyun_sls_project=os.getenv("ALIYUN_SLS_PROJECT", str(sls_cfg.get("project", ""))),
        # 将 aliyun_sls_logstore 的值保存下来，供后续流程判断或组装响应时使用。
        aliyun_sls_logstore=os.getenv("ALIYUN_SLS_LOGSTORE", str(sls_cfg.get("logstore", ""))),
        # 将 aliyun_sls_query_timeout_seconds 的值保存下来，供后续流程判断或组装响应时使用。
        aliyun_sls_query_timeout_seconds=_to_int(
            # 调用 os.getenv 完成当前步骤需要的业务处理。
            os.getenv("ALIYUN_SLS_QUERY_TIMEOUT_SECONDS"),
            # 调用 int 完成当前步骤需要的业务处理。
            int(sls_cfg.get("query_timeout_seconds", 10)),
        ),
        # 将 aliyun_sls_max_lines 的值保存下来，供后续流程判断或组装响应时使用。
        aliyun_sls_max_lines=_to_int(
            # 调用 os.getenv 完成当前步骤需要的业务处理。
            os.getenv("ALIYUN_SLS_MAX_LINES"),
            # 调用 int 完成当前步骤需要的业务处理。
            int(sls_cfg.get("max_lines", 100)),
        ),
        # 阿里云 ARMS 配置
        aliyun_arms_access_key_id=os.getenv("ALIYUN_ARMS_ACCESS_KEY_ID", arms_cfg.get("access_key_id")),
        # 将 aliyun_arms_access_key_secret 的值保存下来，供后续流程判断或组装响应时使用。
        aliyun_arms_access_key_secret=os.getenv("ALIYUN_ARMS_ACCESS_KEY_SECRET", arms_cfg.get("access_key_secret")),
        # 将 aliyun_arms_region_id 的值保存下来，供后续流程判断或组装响应时使用。
        aliyun_arms_region_id=os.getenv(
            # 执行当前业务步骤，推动流程继续向下游推进。
            "ALIYUN_ARMS_REGION_ID",
            # 调用 str 完成当前步骤需要的业务处理。
            str(arms_cfg.get("region_id", "cn-hangzhou")),
        ),
        # 将 aliyun_arms_app_id 的值保存下来，供后续流程判断或组装响应时使用。
        aliyun_arms_app_id=os.getenv("ALIYUN_ARMS_APP_ID", str(arms_cfg.get("app_id", ""))),
        # 将 aliyun_arms_query_timeout_seconds 的值保存下来，供后续流程判断或组装响应时使用。
        aliyun_arms_query_timeout_seconds=_to_int(
            # 调用 os.getenv 完成当前步骤需要的业务处理。
            os.getenv("ALIYUN_ARMS_QUERY_TIMEOUT_SECONDS"),
            # 调用 int 完成当前步骤需要的业务处理。
            int(arms_cfg.get("query_timeout_seconds", 10)),
        ),
        # 将 langfuse 的值保存下来，供后续流程判断或组装响应时使用。
        langfuse=LangfuseConfig(
            # 将 enabled 的值保存下来，供后续流程判断或组装响应时使用。
            enabled=_to_bool(os.getenv("LANGFUSE_ENABLED"), bool(langfuse_cfg.get("enabled", False))),
            # 将 host 的值保存下来，供后续流程判断或组装响应时使用。
            host=_first_non_empty(
                # 调用 os.getenv 完成当前步骤需要的业务处理。
                os.getenv("LANGFUSE_HOST"),
                # 调用 os.getenv 完成当前步骤需要的业务处理。
                os.getenv("LANGFUSE_BASE_URL"),
                # 调用 str 完成当前步骤需要的业务处理。
                str(langfuse_cfg.get("host", "")) or None,
            ),
            # 将 public_key 的值保存下来，供后续流程判断或组装响应时使用。
            public_key=_first_non_empty(os.getenv("LANGFUSE_PUBLIC_KEY"), langfuse_cfg.get("public_key")),
            # 将 secret_key 的值保存下来，供后续流程判断或组装响应时使用。
            secret_key=_first_non_empty(os.getenv("LANGFUSE_SECRET_KEY"), langfuse_cfg.get("secret_key")),
            # 将 cache_ttl_seconds 的值保存下来，供后续流程判断或组装响应时使用。
            cache_ttl_seconds=_to_int(
                # 调用 os.getenv 完成当前步骤需要的业务处理。
                os.getenv("LANGFUSE_CACHE_TTL_SECONDS"),
                # 调用 int 完成当前步骤需要的业务处理。
                int(langfuse_cfg.get("cache_ttl_seconds", 300)),
            ),
            # 将 label 的值保存下来，供后续流程判断或组装响应时使用。
            label=os.getenv("LANGFUSE_LABEL", str(langfuse_cfg.get("label", "production"))),
            # 将 link_initial_delay_ms 的值保存下来，供后续流程判断或组装响应时使用。
            link_initial_delay_ms=_to_int(
                # 调用 os.getenv 完成当前步骤需要的业务处理。
                os.getenv("LANGFUSE_LINK_INITIAL_DELAY_MS"),
                # 调用 int 完成当前步骤需要的业务处理。
                int(langfuse_cfg.get("link_initial_delay_ms", 2000)),
            ),
            # 将 link_poll_interval_ms 的值保存下来，供后续流程判断或组装响应时使用。
            link_poll_interval_ms=_to_int(
                # 调用 os.getenv 完成当前步骤需要的业务处理。
                os.getenv("LANGFUSE_LINK_POLL_INTERVAL_MS"),
                # 调用 int 完成当前步骤需要的业务处理。
                int(langfuse_cfg.get("link_poll_interval_ms", 2000)),
            ),
            # 将 link_max_retries 的值保存下来，供后续流程判断或组装响应时使用。
            link_max_retries=_to_int(
                # 调用 os.getenv 完成当前步骤需要的业务处理。
                os.getenv("LANGFUSE_LINK_MAX_RETRIES"),
                # 调用 int 完成当前步骤需要的业务处理。
                int(langfuse_cfg.get("link_max_retries", 10)),
            ),
            # 将 link_generations 的值保存下来，供后续流程判断或组装响应时使用。
            link_generations=_to_bool(
                # 调用 os.getenv 完成当前步骤需要的业务处理。
                os.getenv("LANGFUSE_LINK_GENERATIONS"),
                # 调用 bool 完成当前步骤需要的业务处理。
                bool(langfuse_cfg.get("link_generations", False)),
            ),
            # 将 remote_prompts_enabled 的值保存下来，供后续流程判断或组装响应时使用。
            remote_prompts_enabled=_to_bool(
                # 调用 os.getenv 完成当前步骤需要的业务处理。
                os.getenv("LANGFUSE_REMOTE_PROMPTS_ENABLED"),
                # 调用 bool 完成当前步骤需要的业务处理。
                bool(langfuse_cfg.get("remote_prompts_enabled", True)),
            ),
            # 将 prompt_dir 的值保存下来，供后续流程判断或组装响应时使用。
            prompt_dir=os.getenv("LANGFUSE_PROMPT_DIR", str(langfuse_cfg.get("prompt_dir", "config/prompts"))),
            # 将 trace_prompts 的值保存下来，供后续流程判断或组装响应时使用。
            trace_prompts=_to_bool(
                # 调用 os.getenv 完成当前步骤需要的业务处理。
                os.getenv("LANGFUSE_TRACE_PROMPTS"),
                # 调用 bool 完成当前步骤需要的业务处理。
                bool(langfuse_cfg.get("trace_prompts", True)),
            ),
            # 将 trace_rag 的值保存下来，供后续流程判断或组装响应时使用。
            trace_rag=_to_bool(os.getenv("LANGFUSE_TRACE_RAG"), bool(langfuse_cfg.get("trace_rag", True))),
            # 将 trace_tools 的值保存下来，供后续流程判断或组装响应时使用。
            trace_tools=_to_bool(os.getenv("LANGFUSE_TRACE_TOOLS"), bool(langfuse_cfg.get("trace_tools", True))),
        ),
    )


# 定义 _load_yaml_settings 相关的处理逻辑，供流程或外部调用复用。
def _load_yaml_settings() -> dict[str, object]:
    # 读取结构化配置文件；读取失败时返回空配置，让环境变量和默认值继续兜底。
    try:
        from agent_sentinel.utils.config_loader import load_yaml

        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return load_yaml(os.getenv("AIOPS_SETTINGS_PATH", "config/settings.yaml"))
    # 捕获当前步骤中的异常，转换为日志、指标或降级结果。
    except Exception:
        # 配置文件缺失或格式错误时使用环境变量/默认值继续启动，便于容器和测试环境运行。
        return {}
