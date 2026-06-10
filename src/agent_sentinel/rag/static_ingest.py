from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from langchain_core.documents import Document
from pydantic import BaseModel, Field

from agent_sentinel.rag.embedding import EmbeddingClient
from agent_sentinel.rag.milvus_client import MilvusVectorClient

logger = logging.getLogger(__name__)

try:  # LangChain 0.x compatibility.
    from langchain.text_splitter import MarkdownHeaderTextSplitter
except ModuleNotFoundError:  # LangChain 1.x splitters live in a companion package.
    from langchain_text_splitters import MarkdownHeaderTextSplitter


MANUAL_DOC_ID = "fault_manual_v1"
MANUAL_LAST_UPDATED = "2026-05-17"
ERROR_CODE_PATTERN = re.compile(
    r"\b(429|500|503|504|Too many connections|Metaspace OOM|OutOfMemoryError|Connection pool exhausted)\b",
    re.IGNORECASE,
)
STOPWORDS = {
    "故障",
    "现象",
    "根因",
    "解决方案",
    "排查",
    "处理",
    "方案",
    "定义",
    "详细",
    "标准",
    "业务",
    "系统",
    "导致",
    "出现",
    "查看",
    "检查",
    "进行",
    "如果",
    "所有",
}


class StaticDocument(BaseModel):
    id: str
    text: str
    doc_type: str = "runbook"
    title: str
    service: str = "global"
    component: str = ""
    tags: list[str] = Field(default_factory=list)
    version: str = "v1"
    updated_at: int = 0
    source_uri: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass(slots=True)
class StaticDocIngestor:
    milvus: MilvusVectorClient
    embedding: EmbeddingClient
    collection_name: str
    dimension: int
    batch_size: int = 16

    async def ensure_collection(self) -> None:
        # 确保 Milvus 中存在静态文档集合，第一次导入知识库前会自动创建。
        await self.milvus.ensure_static_doc_collection(self.collection_name, self.dimension)

    async def ingest_documents(self, docs: list[StaticDocument]) -> int:
        # 把静态文档分批向量化并写入 Milvus，返回成功入库的文档块数量。
        if not docs:
            return 0
        await self.ensure_collection()

        total = 0
        for start in range(0, len(docs), self.batch_size):
            batch = docs[start : start + self.batch_size]
            # 同一批文档并发生成 embedding，再统一 upsert，减少单条写入的往返成本。
            records = await asyncio.gather(*(self._to_record(doc) for doc in batch))
            await self.milvus.upsert(self.collection_name, records)
            total += len(records)
            logger.info("Static docs ingested batch=%s total=%s", len(records), total)
        return total

    async def _to_record(self, doc: StaticDocument) -> dict[str, Any]:
        # 把一条静态文档转换成 Milvus 记录，包含原文、向量和检索元数据。
        embedding = await self.embedding.embed(doc.text)
        now = int(time.time())
        # tags 既作为独立字段存储，也写入 metadata，便于检索结果展示和调试。
        metadata = {
            **doc.metadata,
            "tags": doc.tags,
        }
        return {
            "id": doc.id,
            "text": doc.text,
            "embedding": embedding,
            "doc_type": doc.doc_type,
            "title": doc.title,
            "service": doc.service,
            "component": doc.component,
            "tags": json.dumps(doc.tags, ensure_ascii=False),
            "version": doc.version,
            "updated_at": doc.updated_at or now,
            "created_at": doc.updated_at or now,
            "source_uri": doc.source_uri,
            "source": doc.source_uri,
            "metadata": json.dumps(metadata, ensure_ascii=False),
        }


def load_static_documents(path: str | Path) -> list[StaticDocument]:
    # 从文件或目录加载静态知识文档，支持 Markdown 手册和 JSONL 结构化文档。
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"Static docs path does not exist: {root}")
    if root.is_file():
        return _load_file(root)

    docs: list[StaticDocument] = []
    for file_path in sorted(root.rglob("*")):
        if file_path.suffix.lower() not in {".md", ".markdown", ".jsonl"}:
            continue
        # 支持目录批量导入，Markdown 走标题切片，JSONL 走逐行结构化导入。
        docs.extend(_load_file(file_path))
    return docs


def _load_file(path: Path) -> list[StaticDocument]:
    # 根据文件后缀选择加载方式，Markdown 走标题切片，JSONL 逐行读取。
    suffix = path.suffix.lower()
    if suffix in {".md", ".markdown"}:
        return _load_markdown_chunks(path)
    if suffix == ".jsonl":
        return _load_jsonl(path)
    return []


def split_and_extract_metadata(file_path: str) -> list[Document]:
    """Split a Markdown fault manual by heading hierarchy and enrich RAG metadata.

    The returned Documents can be embedded and upserted into vector stores such as
    Milvus; keep page_content as the chunk text and persist metadata alongside it.
    """
    # 按 Markdown 标题层级切分故障手册，并为每个片段补充故障类型、级别和关键词。

    path = Path(file_path)
    raw = path.read_text(encoding="utf-8")
    frontmatter, body = _split_frontmatter(raw)
    # 按标题层级切片能保留 runbook 的章节语义，比固定长度切片更利于故障手册检索。
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        strip_headers=False,
    )
    chunks = splitter.split_text(body)

    documents: list[Document] = []
    for index, chunk in enumerate(chunks, start=1):
        # frontmatter 提供全局元数据，chunk.metadata 提供当前标题路径，两者合并成检索过滤字段。
        metadata = {
            **_to_dict(frontmatter.get("metadata")),
            **{key: value for key, value in frontmatter.items() if key != "metadata"},
            **chunk.metadata,
        }
        h1 = str(metadata.get("h1") or "")
        h2 = str(metadata.get("h2") or "")
        h3 = str(metadata.get("h3") or "")
        page_content = chunk.page_content.strip()
        metadata.update(
            {
                "doc_id": MANUAL_DOC_ID,
                "chunk_index": index,
                "section": _build_section(h1, h2, h3),
                # 以下字段用于后续按故障类型、严重级别、错误码和关键词做过滤或展示。
                "alert_category": _detect_alert_category(h1),
                "severity_level": _detect_severity(page_content),
                "error_code": _extract_error_codes(page_content),
                "keywords": _extract_keywords(page_content),
                "last_updated": MANUAL_LAST_UPDATED,
                "source": path.as_posix(),
                "file_path": str(path),
            }
        )
        documents.append(Document(page_content=page_content, metadata=metadata))
    return documents


def _load_markdown_chunks(path: Path) -> list[StaticDocument]:
    # 把一个 Markdown 文件切成多个可检索文档块，并生成稳定的块 ID。
    docs = split_and_extract_metadata(str(path))
    if not docs:
        return []
    raw = path.read_text(encoding="utf-8")
    frontmatter, body = _split_frontmatter(raw)
    base_id = str(frontmatter.get("id") or (MANUAL_DOC_ID if path.name == "线上全场景故障排查终极手册（企业级超详细完整版）.md" else path.stem))
    title = str(frontmatter.get("title") or _extract_markdown_title(body) or path.stem)
    updated_at = _to_int(frontmatter.get("updated_at"), 0)
    tags = _to_tags(frontmatter.get("tags"))
    static_docs: list[StaticDocument] = []
    for index, doc in enumerate(docs, start=1):
        metadata = dict(doc.metadata)
        # 第一个 chunk 复用文档 id，其余 chunk 追加序号，保证 Milvus 主键稳定且不冲突。
        chunk_id = base_id if index == 1 else f"{base_id}-{index:04d}"
        chunk_title = str(metadata.get("h3") or metadata.get("h2") or metadata.get("h1") or title)
        static_docs.append(
            StaticDocument(
                id=chunk_id,
                text=doc.page_content,
                doc_type=str(frontmatter.get("doc_type") or "runbook"),
                title=chunk_title,
                service=str(frontmatter.get("service") or "global"),
                component=str(frontmatter.get("component") or metadata.get("alert_category") or ""),
                tags=tags,
                version=str(frontmatter.get("version") or "v1"),
                updated_at=updated_at,
                source_uri=str(frontmatter.get("source_uri") or path.as_posix()),
                metadata=metadata,
            )
        )
    return static_docs


def _load_markdown(path: Path) -> StaticDocument:
    # 把一个 Markdown 文件作为单个文档加载，保留 frontmatter 中的基础元数据。
    raw = path.read_text(encoding="utf-8")
    frontmatter, body = _split_frontmatter(raw)
    title = str(frontmatter.get("title") or _extract_markdown_title(body) or path.stem)
    doc_id = str(frontmatter.get("id") or path.stem)
    updated_at = _to_int(frontmatter.get("updated_at"), 0)
    tags = _to_tags(frontmatter.get("tags"))
    metadata = _to_dict(frontmatter.get("metadata"))
    metadata.setdefault("file_path", str(path))
    return StaticDocument(
        id=doc_id,
        text=body.strip(),
        doc_type=str(frontmatter.get("doc_type") or "runbook"),
        title=title,
        service=str(frontmatter.get("service") or "global"),
        component=str(frontmatter.get("component") or ""),
        tags=tags,
        version=str(frontmatter.get("version") or "v1"),
        updated_at=updated_at,
        source_uri=str(frontmatter.get("source_uri") or path.as_posix()),
        metadata=metadata,
    )


def _load_jsonl(path: Path) -> list[StaticDocument]:
    # 逐行读取 JSONL 文档，每一行都会转成一条可入库的静态知识记录。
    docs: list[StaticDocument] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        # JSONL 允许每行是一个独立文档，缺省 id/title 时用文件名和行号兜底。
        payload.setdefault("id", f"{path.stem}-{line_number}")
        payload.setdefault("title", payload["id"])
        payload.setdefault("metadata", {})
        payload["metadata"]["file_path"] = str(path)
        payload["metadata"]["line_number"] = line_number
        docs.append(StaticDocument.model_validate(payload))
    return docs


def _split_frontmatter(raw: str) -> tuple[dict[str, Any], str]:
    # 拆分 Markdown 顶部的 frontmatter 元数据和正文内容。
    if not raw.startswith("---"):
        return {}, raw
    parts = raw.split("---", 2)
    if len(parts) < 3:
        return {}, raw
    metadata = yaml.safe_load(parts[1]) or {}
    if not isinstance(metadata, dict):
        metadata = {}
    return metadata, parts[2]


def _extract_markdown_title(text: str) -> str | None:
    # 从 Markdown 正文中提取一级标题，作为文档默认标题。
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return None


def _build_section(h1: str, h2: str, h3: str) -> str:
    # 把标题层级拼成章节路径，后续检索结果可显示文档片段来自哪里。
    return "_".join(_normalize_section_part(part) for part in (h1, h2, h3) if part.strip())


def _normalize_section_part(value: str) -> str:
    # 清理章节名中的 Markdown 符号和空格，生成稳定的章节路径片段。
    value = value.replace(r"\.", ".").strip()
    value = re.sub(r"^#+\s*", "", value)
    value = re.sub(r"\s+", "_", value)
    return value.strip("_")


def _detect_alert_category(h1: str) -> str:
    # 根据一级标题判断文档片段属于 MQ、Redis、MySQL 等哪类故障。
    mapping = (
        ("消息队列", "MQ"),
        ("应用服务", "应用服务"),
        ("JVM", "JVM"),
        ("MySQL", "MySQL"),
        ("Redis", "Redis"),
        ("网络、网关、注册中心", "网络/网关"),
        ("容器K8s", "K8s"),
        ("日志、监控、告警", "日志监控"),
        ("业务逻辑", "业务逻辑"),
    )
    for needle, category in mapping:
        if needle in h1:
            return category
    return "其他"


def _detect_severity(page_content: str) -> str:
    # 根据片段中的严重程度关键词推断 P0/P1/P2 级别。
    if any(token in page_content for token in ("核心业务致命故障", "高危", "雪崩")):
        return "P0"
    if any(token in page_content for token in ("严重", "暴涨", "飙升")):
        return "P1"
    return "P2"


def _extract_error_codes(page_content: str) -> list[str]:
    # 从文档片段中提取常见错误码或异常短语，供检索过滤和展示使用。
    seen: set[str] = set()
    values: list[str] = []
    for match in ERROR_CODE_PATTERN.finditer(page_content):
        value = match.group(1)
        key = value.lower()
        if key not in seen:
            seen.add(key)
            values.append(value)
    return values


def _extract_keywords(page_content: str) -> list[str]:
    # 从粗体词、关键小节和高频词中抽取代表文档语义的关键词。
    candidates: list[str] = []
    # 粗体词通常是手册中的关键概念，优先作为候选关键词。
    candidates.extend(_clean_keyword(item) for item in re.findall(r"\*\*([^*]{2,40})\*\*", page_content))

    for line in page_content.splitlines():
        if any(label in line for label in ("故障定义", "根因", "解决方案", "处理建议", "排查要点", "排查步骤")):
            # 关键小节标题附近的信息通常比普通正文更能代表故障语义。
            candidates.extend(_keyword_tokens(line))

    candidates.extend(
        token
        for token, _ in Counter(_keyword_tokens(page_content)).most_common(30)
    )

    keywords: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        keyword = _clean_keyword(candidate)
        if not keyword or keyword in STOPWORDS or keyword in seen:
            continue
        seen.add(keyword)
        keywords.append(keyword)
        if len(keywords) >= 10:
            break
    return keywords[:10]


def _keyword_tokens(text: str) -> list[str]:
    # 从中英文混合文本里提取可作为关键词的短词片段。
    tokens = re.findall(r"[\u4e00-\u9fffA-Za-z0-9][\u4e00-\u9fffA-Za-z0-9+/._-]{1,24}", text)
    return [token for token in tokens if len(token) >= 2]


def _clean_keyword(value: str) -> str:
    # 去掉关键词周围的 Markdown 符号和标点，限制关键词长度。
    value = re.sub(r"[*`#>\-：:，,。；;、（）()\[\]【】]", "", str(value)).strip()
    return value[:40]


def _to_tags(value: Any) -> list[str]:
    # 把 frontmatter 中的标签字段统一转换成字符串列表。
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _to_dict(value: Any) -> dict[str, Any]:
    # 确保 metadata 字段一定是字典，写错类型时退回空字典。
    return value if isinstance(value, dict) else {}


def _to_int(value: Any, default: int) -> int:
    # 把 frontmatter 里的时间戳等字段转成整数，失败时使用默认值。
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


if __name__ == "__main__":
    input_file = "线上全场景故障排查终极手册（企业级超详细完整版）.md"
    file_path = Path(input_file)
    if not file_path.exists():
        file_path = Path("data/static_docs") / input_file
    documents = split_and_extract_metadata(str(file_path))
    for index, document in enumerate(documents, start=1):
        print(f"[{index}] content preview:")
        print(document.page_content[:200].replace("\n", "\\n"))
        print("metadata:")
        print(json.dumps(document.metadata, ensure_ascii=False, indent=2))
