from __future__ import annotations

import os
import shlex
from dataclasses import dataclass, field

from dotenv import load_dotenv


def _to_bool(value: str | None, default: bool = False) -> bool:
    # 方法说明：把环境变量里的 true/false 字符串转换成布尔开关，没配置时使用默认值。
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _to_float(value: str | None, default: float) -> float:
    # 方法说明：把环境变量里的小数字符串转换成浮点数，例如温度、权重和阈值。
    if value is None or value == "":
        return default
    return float(value)


def _to_int(value: str | None, default: int) -> int:
    # 方法说明：把环境变量里的数字字符串转换成整数，例如端口、超时和条数限制。
    if value is None or value == "":
        return default
    return int(value)


def _to_list(value: str | None) -> list[str]:
    # 方法说明：把逗号分隔的配置字符串转换成列表，例如模型列表或群白名单。
    if value is None or value.strip() == "":
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _to_args(value: str | None) -> list[str]:
    # 方法说明：把 MCP 启动参数解析成列表，兼容逗号写法和命令行参数写法。
    if value is None or value.strip() == "":
        return []
    # MCP 启动参数既可能来自逗号分隔的环境变量，也可能是 shell 风格参数串。
    if "," in value:
        return _to_list(value)
    return shlex.split(value)


def _first_non_empty(*values: str | None) -> str | None:
    # 多个兼容环境变量按优先级取第一个非空值，便于兼容 Codex/OpenAI/自定义配置名。
    # 方法说明：从多个候选配置名中取第一个非空值，用来兼容不同部署环境的命名习惯。
    for value in values:
        if value is not None and value != "":
            return value
    return None


@dataclass(slots=True)
class LangfuseConfig:
    enabled: bool = False
    host: str | None = None
    public_key: str | None = None
    secret_key: str | None = None
    cache_ttl_seconds: int = 300
    label: str = "production"
    link_initial_delay_ms: int = 2000
    link_poll_interval_ms: int = 2000
    link_max_retries: int = 10
    link_generations: bool = False
    remote_prompts_enabled: bool = True
    prompt_dir: str = "config/prompts"
    trace_prompts: bool = True
    trace_rag: bool = True
    trace_tools: bool = True


@dataclass(slots=True)
class Settings:
    app_name: str
    app_host: str
    app_port: int
    app_env: str
    log_level: str
    openai_api_key: str
    openai_base_url: str | None
    openai_model: str
    openai_temperature: float
    openai_http_trust_env: bool
    feishu_api_base_url: str
    feishu_app_id: str | None
    feishu_app_secret: str | None
    feishu_event_verification_token: str | None
    feishu_event_encrypt_key: str | None
    feishu_bot_name: str
    feishu_allowed_chat_ids: list[str]
    feishu_analyze_mention_only: bool
    feishu_long_connection_enabled: bool
    feishu_message_polling_enabled: bool
    feishu_message_polling_interval_seconds: int
    feishu_message_polling_page_size: int
    feishu_webhook_url: str | None
    feishu_secret: str | None
    feishu_alert_enabled: bool
    feishu_alert_title_prefix: str
    alert_analysis_enabled: bool
    alert_analysis_title_prefix: str
    alert_api_token: str | None
    alert_dedup_window_seconds: int
    alert_store_limit: int
    metrics_enabled: bool = True
    metrics_port: int | None = None
    aiops_workflow_config_path: str = "config/workflow.yaml"
    aiops_llm_models: list[str] = field(default_factory=lambda: ["gpt-4o-mini"])
    aiops_llm_timeout_seconds: int = 15
    aiops_llm_max_retries: int = 2
    aiops_mock_llm_enabled: bool = False
    aiops_human_confirm_timeout_seconds: int = 300
    aiops_human_confirm_enabled: bool = True
    redis_url: str | None = None
    rag_provider: str = "mock"
    rag_final_top_k: int = 2
    rag_mmr_lambda: float = 0.55
    rag_static_enabled: bool = True
    rag_static_collection: str = "aiops_static_docs"
    rag_static_top_k: int = 2
    rag_static_weight: float = 0.55
    rag_message_enabled: bool = True
    rag_message_collection: str = "aiops_message_history"
    rag_message_top_k: int = 2
    rag_message_weight: float = 0.45
    rag_message_default_days: int = 30
    rag_case_cache_enabled: bool = True
    rag_case_cache_threshold: float = 0.85
    rag_case_cache_top_k: int = 2
    rag_feedback_enabled: bool = True
    rag_pruning_enabled: bool = True
    rag_history_prune_top_k: int = 1
    rag_static_recall_top_k: int = 10
    rag_reranker_model_name: str = "qwen3-rerank"
    rag_reranker_device: str = "cpu"
    rag_reranker_api_endpoint: str | None = None
    rag_reranker_api_key: str | None = None
    rag_reranker_api_format: str = "auto"
    rag_reranker_timeout_seconds: int = 15
    milvus_uri: str | None = None
    milvus_token: str | None = None
    milvus_user: str | None = None
    milvus_password: str | None = None
    milvus_db_name: str | None = None
    embedding_api_key: str | None = None
    embedding_base_url: str | None = None
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536
    embedding_mock_enabled: bool = False
    interactive_topic_enabled: bool = True
    interactive_topic_wait_seconds: int = 5

    # Tools provider 配置
    tools_provider: str = "mock"  # mock | aliyun | mcp
    tools_metrics_enabled: bool = True
    tools_logs_enabled: bool = True
    tools_topology_enabled: bool = True
    mcp_enabled: bool = False
    mcp_server_command: str = ""
    mcp_server_args: list[str] = field(default_factory=list)
    mcp_timeout_seconds: int = 20
    mcp_sls_tool: str = "aliyun_sls_query_logs"
    mcp_prometheus_tool: str = "prometheus_query_metrics"
    mcp_prometheus_base_url: str | None = None

    # 阿里云 SLS 配置
    aliyun_sls_access_key_id: str | None = None
    aliyun_sls_access_key_secret: str | None = None
    aliyun_sls_endpoint: str = "cn-hangzhou.log.aliyuncs.com"
    aliyun_sls_project: str = ""
    aliyun_sls_logstore: str = ""
    aliyun_sls_query_timeout_seconds: int = 10
    aliyun_sls_max_lines: int = 100

    # 阿里云 ARMS 配置
    aliyun_arms_access_key_id: str | None = None
    aliyun_arms_access_key_secret: str | None = None
    aliyun_arms_region_id: str = "cn-hangzhou"
    aliyun_arms_app_id: str = ""
    aliyun_arms_query_timeout_seconds: int = 10
    langfuse: LangfuseConfig = field(default_factory=LangfuseConfig)


def get_settings() -> Settings:
    # 方法说明：汇总 YAML、.env 和系统环境变量，生成整个应用运行时使用的统一配置对象。
    load_dotenv()
    # settings.yaml 提供默认结构化配置，环境变量用于部署时覆盖敏感项和环境差异。
    yaml_settings = _load_yaml_settings()
    app_cfg = yaml_settings.get("app", {})
    llm_cfg = yaml_settings.get("llm", {})
    workflow_cfg = yaml_settings.get("workflow", {})
    redis_cfg = yaml_settings.get("redis", {})
    rag_cfg = yaml_settings.get("rag", {})
    static_cfg = dict(rag_cfg.get("static_docs", {})) if isinstance(rag_cfg.get("static_docs", {}), dict) else {}
    message_cfg = (
        dict(rag_cfg.get("message_history", {}))
        if isinstance(rag_cfg.get("message_history", {}), dict)
        else {}
    )
    # 子配置统一转成 dict，避免 YAML 写错类型时在后续 .get 调用处抛异常。
    embedding_cfg = yaml_settings.get("embedding", {})
    tools_cfg = yaml_settings.get("tools", {}) if isinstance(yaml_settings.get("tools", {}), dict) else {}
    mcp_cfg = tools_cfg.get("mcp", {}) if isinstance(tools_cfg.get("mcp", {}), dict) else {}
    sls_cfg = tools_cfg.get("aliyun_sls", {}) if isinstance(tools_cfg.get("aliyun_sls", {}), dict) else {}
    arms_cfg = tools_cfg.get("aliyun_arms", {}) if isinstance(tools_cfg.get("aliyun_arms", {}), dict) else {}
    langfuse_cfg = yaml_settings.get("langfuse", {}) if isinstance(yaml_settings.get("langfuse", {}), dict) else {}
    return Settings(
        app_name=os.getenv("APP_NAME", str(app_cfg.get("name", "Agent Sentinel"))),
        app_host=os.getenv("APP_HOST", str(app_cfg.get("host", "0.0.0.0"))),
        app_port=_to_int(os.getenv("APP_PORT"), int(app_cfg.get("port", 8000))),
        app_env=os.getenv("APP_ENV", str(app_cfg.get("env", "dev"))),
        log_level=os.getenv("LOG_LEVEL", str(app_cfg.get("log_level", "INFO"))),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_base_url=_first_non_empty(
            # 兼容不同运行环境中常见的 base_url 命名。
            os.getenv("CODEX_BASE_URL"),
            os.getenv("OPENAI_BASE_URL"),
            os.getenv("base_url"),
        ),
        openai_model=_first_non_empty(
            os.getenv("CODEX_MODEL"),
            os.getenv("OPENAI_MODEL"),
            os.getenv("model"),
        )
        or "gpt-4o-mini",
        openai_temperature=_to_float(os.getenv("OPENAI_TEMPERATURE"), 0.0),
        openai_http_trust_env=_to_bool(os.getenv("OPENAI_HTTP_TRUST_ENV"), False),
        feishu_api_base_url=os.getenv("FEISHU_API_BASE_URL", "https://open.feishu.cn"),
        feishu_app_id=os.getenv("FEISHU_APP_ID"),
        feishu_app_secret=os.getenv("FEISHU_APP_SECRET"),
        feishu_event_verification_token=os.getenv("FEISHU_EVENT_VERIFICATION_TOKEN"),
        feishu_event_encrypt_key=os.getenv("FEISHU_EVENT_ENCRYPT_KEY"),
        feishu_bot_name=os.getenv("FEISHU_BOT_NAME", "Analysis Bot"),
        feishu_allowed_chat_ids=_to_list(os.getenv("FEISHU_ALLOWED_CHAT_IDS")),
        feishu_analyze_mention_only=_to_bool(os.getenv("FEISHU_ANALYZE_MENTION_ONLY"), True),
        feishu_long_connection_enabled=_to_bool(os.getenv("FEISHU_LONG_CONNECTION_ENABLED"), True),
        feishu_message_polling_enabled=_to_bool(os.getenv("FEISHU_MESSAGE_POLLING_ENABLED"), False),
        feishu_message_polling_interval_seconds=_to_int(
            os.getenv("FEISHU_MESSAGE_POLLING_INTERVAL_SECONDS"),
            5,
        ),
        feishu_message_polling_page_size=_to_int(
            os.getenv("FEISHU_MESSAGE_POLLING_PAGE_SIZE"),
            20,
        ),
        feishu_webhook_url=os.getenv("FEISHU_WEBHOOK_URL"),
        feishu_secret=os.getenv("FEISHU_SECRET"),
        feishu_alert_enabled=_to_bool(os.getenv("FEISHU_ALERT_ENABLED"), True),
        feishu_alert_title_prefix=os.getenv("FEISHU_ALERT_TITLE_PREFIX", "Agent Sentinel"),
        alert_analysis_enabled=_to_bool(os.getenv("ALERT_ANALYSIS_ENABLED"), True),
        alert_analysis_title_prefix=os.getenv("ALERT_ANALYSIS_TITLE_PREFIX", "Alert Analysis"),
        alert_api_token=os.getenv("ALERT_API_TOKEN"),
        alert_dedup_window_seconds=_to_int(os.getenv("ALERT_DEDUP_WINDOW_SECONDS"), 60),
        alert_store_limit=_to_int(os.getenv("ALERT_STORE_LIMIT"), 100),
        metrics_enabled=_to_bool(os.getenv("METRICS_ENABLED"), True),
        metrics_port=_to_int(os.getenv("METRICS_PORT"), 0) or None,
        aiops_workflow_config_path=os.getenv(
            "AIOPS_WORKFLOW_CONFIG_PATH",
            str(workflow_cfg.get("config_path", "config/workflow.yaml")),
        ),
        aiops_llm_models=_to_list(os.getenv("AIOPS_LLM_MODELS"))
        or [str(item) for item in llm_cfg.get("models", [])]
        # 未显式配置多模型时，退回到单模型配置，保证诊断流程总有可用模型名。
        or [_first_non_empty(os.getenv("CODEX_MODEL"), os.getenv("OPENAI_MODEL")) or "gpt-4o-mini"],
        aiops_llm_timeout_seconds=_to_int(
            os.getenv("AIOPS_LLM_TIMEOUT_SECONDS"),
            int(llm_cfg.get("timeout_seconds", 15)),
        ),
        aiops_llm_max_retries=_to_int(
            os.getenv("AIOPS_LLM_MAX_RETRIES"),
            int(llm_cfg.get("max_retries", 2)),
        ),
        aiops_mock_llm_enabled=_to_bool(
            os.getenv("AIOPS_MOCK_LLM_ENABLED"),
            bool(llm_cfg.get("mock_enabled", False)),
        ),
        aiops_human_confirm_timeout_seconds=_to_int(
            os.getenv("AIOPS_HUMAN_CONFIRM_TIMEOUT_SECONDS"),
            int(workflow_cfg.get("human_confirm_timeout_seconds", 300)),
        ),
        aiops_human_confirm_enabled=_to_bool(
            os.getenv("AIOPS_HUMAN_CONFIRM_ENABLED"),
            bool(workflow_cfg.get("human_confirm_enabled", True)),
        ),
        redis_url=os.getenv("REDIS_URL", str(redis_cfg.get("url", "")) or None),
        rag_provider=os.getenv("RAG_PROVIDER", str(rag_cfg.get("provider", "mock"))),
        rag_final_top_k=_to_int(os.getenv("RAG_FINAL_TOP_K"), int(rag_cfg.get("final_top_k", 2))),
        rag_mmr_lambda=_to_float(os.getenv("RAG_MMR_LAMBDA"), float(rag_cfg.get("mmr_lambda", 0.55))),
        rag_static_enabled=_to_bool(
            os.getenv("RAG_STATIC_ENABLED"),
            bool(static_cfg.get("enabled", True)),
        ),
        rag_static_collection=os.getenv(
            "RAG_STATIC_COLLECTION",
            str(static_cfg.get("collection", "aiops_static_docs")),
        ),
        rag_static_top_k=_to_int(os.getenv("RAG_STATIC_TOP_K"), int(static_cfg.get("top_k", 2))),
        rag_static_weight=_to_float(os.getenv("RAG_STATIC_WEIGHT"), float(static_cfg.get("weight", 0.55))),
        rag_message_enabled=_to_bool(
            os.getenv("RAG_MESSAGE_ENABLED"),
            bool(message_cfg.get("enabled", True)),
        ),
        rag_message_collection=os.getenv(
            "RAG_MESSAGE_COLLECTION",
            str(message_cfg.get("collection", "aiops_message_history")),
        ),
        rag_message_top_k=_to_int(os.getenv("RAG_MESSAGE_TOP_K"), int(message_cfg.get("top_k", 2))),
        rag_message_weight=_to_float(os.getenv("RAG_MESSAGE_WEIGHT"), float(message_cfg.get("weight", 0.45))),
        rag_message_default_days=_to_int(
            os.getenv("RAG_MESSAGE_DEFAULT_DAYS"),
            int(message_cfg.get("default_days", 30)),
        ),
        rag_case_cache_enabled=_to_bool(
            os.getenv("RAG_CASE_CACHE_ENABLED"),
            bool(message_cfg.get("case_cache_enabled", True)),
        ),
        rag_case_cache_threshold=_to_float(
            os.getenv("RAG_CASE_CACHE_THRESHOLD"),
            float(message_cfg.get("case_cache_threshold", 0.85)),
        ),
        rag_case_cache_top_k=_to_int(
            os.getenv("RAG_CASE_CACHE_TOP_K"),
            int(message_cfg.get("case_cache_top_k", 2)),
        ),
        rag_feedback_enabled=_to_bool(
            os.getenv("RAG_FEEDBACK_ENABLED"),
            bool(message_cfg.get("feedback_enabled", True)),
        ),
        rag_pruning_enabled=_to_bool(os.getenv("RAG_PRUNING_ENABLED"), True),
        rag_history_prune_top_k=_to_int(os.getenv("RAG_HISTORY_PRUNE_TOP_K"), 1),
        rag_static_recall_top_k=_to_int(os.getenv("RAG_STATIC_RECALL_TOP_K"), 10),
        rag_reranker_model_name=os.getenv("RAG_RERANKER_MODEL_NAME", "qwen3-rerank"),
        rag_reranker_device=os.getenv("RAG_RERANKER_DEVICE", "cpu"),
        rag_reranker_api_endpoint=os.getenv("RAG_RERANKER_API_ENDPOINT"),
        rag_reranker_api_key=os.getenv("RAG_RERANKER_API_KEY"),
        rag_reranker_api_format=os.getenv("RAG_RERANKER_API_FORMAT", "auto"),
        rag_reranker_timeout_seconds=_to_int(os.getenv("RAG_RERANKER_TIMEOUT_SECONDS"), 15),
        milvus_uri=os.getenv("MILVUS_URI"),
        milvus_token=os.getenv("MILVUS_TOKEN"),
        milvus_user=os.getenv("MILVUS_USER"),
        milvus_password=os.getenv("MILVUS_PASSWORD"),
        milvus_db_name=os.getenv("MILVUS_DB_NAME"),
        embedding_api_key=_first_non_empty(os.getenv("EMBEDDING_API_KEY"), os.getenv("OPENAI_API_KEY")),
        embedding_base_url=_first_non_empty(os.getenv("EMBEDDING_BASE_URL"), os.getenv("OPENAI_BASE_URL")),
        embedding_model=os.getenv(
            "EMBEDDING_MODEL",
            str(embedding_cfg.get("model", "text-embedding-3-small")),
        ),
        embedding_dimension=_to_int(
            os.getenv("EMBEDDING_DIMENSION"),
            int(embedding_cfg.get("dimension", 1536)),
        ),
        embedding_mock_enabled=_to_bool(
            os.getenv("EMBEDDING_MOCK_ENABLED"),
            bool(embedding_cfg.get("mock_enabled", False)),
        ),
        interactive_topic_enabled=_to_bool(os.getenv("INTERACTIVE_TOPIC_ENABLED"), True),
        interactive_topic_wait_seconds=_to_int(os.getenv("INTERACTIVE_TOPIC_WAIT_SECONDS"), 5),
        # Tools provider 配置
        tools_provider=os.getenv("TOOLS_PROVIDER", str(tools_cfg.get("provider", "mock"))),
        tools_metrics_enabled=_to_bool(
            os.getenv("TOOLS_METRICS_ENABLED"),
            bool(tools_cfg.get("metrics_enabled", True)),
        ),
        tools_logs_enabled=_to_bool(
            os.getenv("TOOLS_LOGS_ENABLED"),
            bool(tools_cfg.get("logs_enabled", True)),
        ),
        tools_topology_enabled=_to_bool(
            os.getenv("TOOLS_TOPOLOGY_ENABLED"),
            bool(tools_cfg.get("topology_enabled", True)),
        ),
        # 阿里云 SLS 配置
        mcp_enabled=_to_bool(os.getenv("MCP_ENABLED"), bool(mcp_cfg.get("enabled", False))),
        mcp_server_command=os.getenv("MCP_SERVER_COMMAND", str(mcp_cfg.get("server_command", ""))),
        # MCP 参数需要支持两种来源：YAML 原生列表和环境变量字符串。
        mcp_server_args=_to_args(os.getenv("MCP_SERVER_ARGS"))
        or [str(item) for item in mcp_cfg.get("server_args", [])],
        mcp_timeout_seconds=_to_int(os.getenv("MCP_TIMEOUT_SECONDS"), int(mcp_cfg.get("timeout_seconds", 20))),
        mcp_sls_tool=os.getenv("MCP_SLS_TOOL", str(mcp_cfg.get("sls_tool", "aliyun_sls_query_logs"))),
        mcp_prometheus_tool=os.getenv(
            "MCP_PROMETHEUS_TOOL",
            str(mcp_cfg.get("prometheus_tool", "prometheus_query_metrics")),
        ),
        mcp_prometheus_base_url=_first_non_empty(
            os.getenv("MCP_PROMETHEUS_BASE_URL"),
            str(mcp_cfg.get("prometheus_base_url", "")) or None,
        ),
        aliyun_sls_access_key_id=os.getenv("ALIYUN_SLS_ACCESS_KEY_ID", sls_cfg.get("access_key_id")),
        aliyun_sls_access_key_secret=os.getenv("ALIYUN_SLS_ACCESS_KEY_SECRET", sls_cfg.get("access_key_secret")),
        aliyun_sls_endpoint=os.getenv(
            "ALIYUN_SLS_ENDPOINT",
            str(sls_cfg.get("endpoint", "cn-hangzhou.log.aliyuncs.com")),
        ),
        aliyun_sls_project=os.getenv("ALIYUN_SLS_PROJECT", str(sls_cfg.get("project", ""))),
        aliyun_sls_logstore=os.getenv("ALIYUN_SLS_LOGSTORE", str(sls_cfg.get("logstore", ""))),
        aliyun_sls_query_timeout_seconds=_to_int(
            os.getenv("ALIYUN_SLS_QUERY_TIMEOUT_SECONDS"),
            int(sls_cfg.get("query_timeout_seconds", 10)),
        ),
        aliyun_sls_max_lines=_to_int(
            os.getenv("ALIYUN_SLS_MAX_LINES"),
            int(sls_cfg.get("max_lines", 100)),
        ),
        # 阿里云 ARMS 配置
        aliyun_arms_access_key_id=os.getenv("ALIYUN_ARMS_ACCESS_KEY_ID", arms_cfg.get("access_key_id")),
        aliyun_arms_access_key_secret=os.getenv("ALIYUN_ARMS_ACCESS_KEY_SECRET", arms_cfg.get("access_key_secret")),
        aliyun_arms_region_id=os.getenv(
            "ALIYUN_ARMS_REGION_ID",
            str(arms_cfg.get("region_id", "cn-hangzhou")),
        ),
        aliyun_arms_app_id=os.getenv("ALIYUN_ARMS_APP_ID", str(arms_cfg.get("app_id", ""))),
        aliyun_arms_query_timeout_seconds=_to_int(
            os.getenv("ALIYUN_ARMS_QUERY_TIMEOUT_SECONDS"),
            int(arms_cfg.get("query_timeout_seconds", 10)),
        ),
        langfuse=LangfuseConfig(
            enabled=_to_bool(os.getenv("LANGFUSE_ENABLED"), bool(langfuse_cfg.get("enabled", False))),
            host=_first_non_empty(
                os.getenv("LANGFUSE_HOST"),
                os.getenv("LANGFUSE_BASE_URL"),
                str(langfuse_cfg.get("host", "")) or None,
            ),
            public_key=_first_non_empty(os.getenv("LANGFUSE_PUBLIC_KEY"), langfuse_cfg.get("public_key")),
            secret_key=_first_non_empty(os.getenv("LANGFUSE_SECRET_KEY"), langfuse_cfg.get("secret_key")),
            cache_ttl_seconds=_to_int(
                os.getenv("LANGFUSE_CACHE_TTL_SECONDS"),
                int(langfuse_cfg.get("cache_ttl_seconds", 300)),
            ),
            label=os.getenv("LANGFUSE_LABEL", str(langfuse_cfg.get("label", "production"))),
            link_initial_delay_ms=_to_int(
                os.getenv("LANGFUSE_LINK_INITIAL_DELAY_MS"),
                int(langfuse_cfg.get("link_initial_delay_ms", 2000)),
            ),
            link_poll_interval_ms=_to_int(
                os.getenv("LANGFUSE_LINK_POLL_INTERVAL_MS"),
                int(langfuse_cfg.get("link_poll_interval_ms", 2000)),
            ),
            link_max_retries=_to_int(
                os.getenv("LANGFUSE_LINK_MAX_RETRIES"),
                int(langfuse_cfg.get("link_max_retries", 10)),
            ),
            link_generations=_to_bool(
                os.getenv("LANGFUSE_LINK_GENERATIONS"),
                bool(langfuse_cfg.get("link_generations", False)),
            ),
            remote_prompts_enabled=_to_bool(
                os.getenv("LANGFUSE_REMOTE_PROMPTS_ENABLED"),
                bool(langfuse_cfg.get("remote_prompts_enabled", True)),
            ),
            prompt_dir=os.getenv("LANGFUSE_PROMPT_DIR", str(langfuse_cfg.get("prompt_dir", "config/prompts"))),
            trace_prompts=_to_bool(
                os.getenv("LANGFUSE_TRACE_PROMPTS"),
                bool(langfuse_cfg.get("trace_prompts", True)),
            ),
            trace_rag=_to_bool(os.getenv("LANGFUSE_TRACE_RAG"), bool(langfuse_cfg.get("trace_rag", True))),
            trace_tools=_to_bool(os.getenv("LANGFUSE_TRACE_TOOLS"), bool(langfuse_cfg.get("trace_tools", True))),
        ),
    )


def _load_yaml_settings() -> dict[str, object]:
    # 方法说明：读取结构化配置文件；读取失败时返回空配置，让环境变量和默认值继续兜底。
    try:
        from agent_sentinel.utils.config_loader import load_yaml

        return load_yaml(os.getenv("AIOPS_SETTINGS_PATH", "config/settings.yaml"))
    except Exception:
        # 配置文件缺失或格式错误时使用环境变量/默认值继续启动，便于容器和测试环境运行。
        return {}
