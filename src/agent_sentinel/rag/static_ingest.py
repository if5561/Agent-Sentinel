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

# 将 logger 的值保存下来，供后续流程判断或组装响应时使用。
logger = logging.getLogger(__name__)

# 进入可能失败的处理块，便于后续统一捕获和恢复。
try:  # LangChain 0.x compatibility.
    from langchain.text_splitter import MarkdownHeaderTextSplitter
# 捕获当前步骤中的异常，转换为日志、指标或降级结果。
except ModuleNotFoundError:  # LangChain 1.x splitters live in a companion package.
    from langchain_text_splitters import MarkdownHeaderTextSplitter


# 将 MANUAL_DOC_ID 的值保存下来，供后续流程判断或组装响应时使用。
MANUAL_DOC_ID = "fault_manual_v1"
# 将 MANUAL_LAST_UPDATED 的值保存下来，供后续流程判断或组装响应时使用。
MANUAL_LAST_UPDATED = "2026-05-17"
# 将 ERROR_CODE_PATTERN 的值保存下来，供后续流程判断或组装响应时使用。
ERROR_CODE_PATTERN = re.compile(
    # 调用 b 完成当前步骤需要的业务处理。
    r"\b(429|500|503|504|Too many connections|Metaspace OOM|OutOfMemoryError|Connection pool exhausted)\b",
    # 执行当前业务步骤，推动流程继续向下游推进。
    re.IGNORECASE,
)
# 将 STOPWORDS 的值保存下来，供后续流程判断或组装响应时使用。
STOPWORDS = {
    # 执行当前业务步骤，推动流程继续向下游推进。
    "故障",
    # 执行当前业务步骤，推动流程继续向下游推进。
    "现象",
    # 执行当前业务步骤，推动流程继续向下游推进。
    "根因",
    # 执行当前业务步骤，推动流程继续向下游推进。
    "解决方案",
    # 执行当前业务步骤，推动流程继续向下游推进。
    "排查",
    # 执行当前业务步骤，推动流程继续向下游推进。
    "处理",
    # 执行当前业务步骤，推动流程继续向下游推进。
    "方案",
    # 执行当前业务步骤，推动流程继续向下游推进。
    "定义",
    # 执行当前业务步骤，推动流程继续向下游推进。
    "详细",
    # 执行当前业务步骤，推动流程继续向下游推进。
    "标准",
    # 执行当前业务步骤，推动流程继续向下游推进。
    "业务",
    # 执行当前业务步骤，推动流程继续向下游推进。
    "系统",
    # 执行当前业务步骤，推动流程继续向下游推进。
    "导致",
    # 执行当前业务步骤，推动流程继续向下游推进。
    "出现",
    # 执行当前业务步骤，推动流程继续向下游推进。
    "查看",
    # 执行当前业务步骤，推动流程继续向下游推进。
    "检查",
    # 执行当前业务步骤，推动流程继续向下游推进。
    "进行",
    # 执行当前业务步骤，推动流程继续向下游推进。
    "如果",
    # 执行当前业务步骤，推动流程继续向下游推进。
    "所有",
}


# 定义 StaticDocument 组件，集中管理这个模块的状态和行为。
class StaticDocument(BaseModel):
    # 执行当前业务步骤，推动流程继续向下游推进。
    id: str
    # 执行当前业务步骤，推动流程继续向下游推进。
    text: str
    # 将 doc_type 的值保存下来，供后续流程判断或组装响应时使用。
    doc_type: str = "runbook"
    # 执行当前业务步骤，推动流程继续向下游推进。
    title: str
    # 将 service 的值保存下来，供后续流程判断或组装响应时使用。
    service: str = "global"
    # 将 component 的值保存下来，供后续流程判断或组装响应时使用。
    component: str = ""
    # 将 tags 的值保存下来，供后续流程判断或组装响应时使用。
    tags: list[str] = Field(default_factory=list)
    # 将 version 的值保存下来，供后续流程判断或组装响应时使用。
    version: str = "v1"
    # 将 updated_at 的值保存下来，供后续流程判断或组装响应时使用。
    updated_at: int = 0
    # 将 source_uri 的值保存下来，供后续流程判断或组装响应时使用。
    source_uri: str = ""
    # 将 metadata 的值保存下来，供后续流程判断或组装响应时使用。
    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass(slots=True)
# 定义 StaticDocIngestor 组件，集中管理这个模块的状态和行为。
class StaticDocIngestor:
    # 执行当前业务步骤，推动流程继续向下游推进。
    milvus: MilvusVectorClient
    # 执行当前业务步骤，推动流程继续向下游推进。
    embedding: EmbeddingClient
    # 执行当前业务步骤，推动流程继续向下游推进。
    collection_name: str
    # 执行当前业务步骤，推动流程继续向下游推进。
    dimension: int
    # 将 batch_size 的值保存下来，供后续流程判断或组装响应时使用。
    batch_size: int = 16

    # 定义 ensure_collection 相关的处理逻辑，供流程或外部调用复用。
    async def ensure_collection(self) -> None:
        # 确保 Milvus 中存在静态文档集合，第一次导入知识库前会自动创建。
        await self.milvus.ensure_static_doc_collection(self.collection_name, self.dimension)

    # 定义 ingest_documents 相关的处理逻辑，供流程或外部调用复用。
    async def ingest_documents(self, docs: list[StaticDocument]) -> int:
        # 把静态文档分批向量化并写入 Milvus，返回成功入库的文档块数量。
        if not docs:
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return 0
        # 等待异步操作完成，再继续推进当前业务流程。
        await self.ensure_collection()

        # 将 total 的值保存下来，供后续流程判断或组装响应时使用。
        total = 0
        # 遍历当前集合中的每一项，逐步完成聚合或转换处理。
        for start in range(0, len(docs), self.batch_size):
            # 将 batch 的值保存下来，供后续流程判断或组装响应时使用。
            batch = docs[start : start + self.batch_size]
            # 同一批文档并发生成 embedding，再统一 upsert，减少单条写入的往返成本。
            records = await asyncio.gather(*(self._to_record(doc) for doc in batch))
            # 等待异步操作完成，再继续推进当前业务流程。
            await self.milvus.upsert(self.collection_name, records)
            # 保存当前计算结果，供后续流程判断或组装响应时使用。
            total += len(records)
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.info("Static docs ingested batch=%s total=%s", len(records), total)
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return total

    # 定义 _to_record 相关的处理逻辑，供流程或外部调用复用。
    async def _to_record(self, doc: StaticDocument) -> dict[str, Any]:
        # 把一条静态文档转换成 Milvus 记录，包含原文、向量和检索元数据。
        embedding = await self.embedding.embed(doc.text)
        # 将 now 的值保存下来，供后续流程判断或组装响应时使用。
        now = int(time.time())
        # tags 既作为独立字段存储，也写入 metadata，便于检索结果展示和调试。
        metadata = {
            # 执行当前业务步骤，推动流程继续向下游推进。
            **doc.metadata,
            # 执行当前业务步骤，推动流程继续向下游推进。
            "tags": doc.tags,
        }
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return {
            # 执行当前业务步骤，推动流程继续向下游推进。
            "id": doc.id,
            # 执行当前业务步骤，推动流程继续向下游推进。
            "text": doc.text,
            # 执行当前业务步骤，推动流程继续向下游推进。
            "embedding": embedding,
            # 执行当前业务步骤，推动流程继续向下游推进。
            "doc_type": doc.doc_type,
            # 执行当前业务步骤，推动流程继续向下游推进。
            "title": doc.title,
            # 执行当前业务步骤，推动流程继续向下游推进。
            "service": doc.service,
            # 执行当前业务步骤，推动流程继续向下游推进。
            "component": doc.component,
            # 保存当前计算结果，供后续流程判断或组装响应时使用。
            "tags": json.dumps(doc.tags, ensure_ascii=False),
            # 执行当前业务步骤，推动流程继续向下游推进。
            "version": doc.version,
            # 执行当前业务步骤，推动流程继续向下游推进。
            "updated_at": doc.updated_at or now,
            # 执行当前业务步骤，推动流程继续向下游推进。
            "created_at": doc.updated_at or now,
            # 执行当前业务步骤，推动流程继续向下游推进。
            "source_uri": doc.source_uri,
            # 执行当前业务步骤，推动流程继续向下游推进。
            "source": doc.source_uri,
            # 保存当前计算结果，供后续流程判断或组装响应时使用。
            "metadata": json.dumps(metadata, ensure_ascii=False),
        }


# 定义 load_static_documents 相关的处理逻辑，供流程或外部调用复用。
def load_static_documents(path: str | Path) -> list[StaticDocument]:
    # 从文件或目录加载静态知识文档，支持 Markdown 手册和 JSONL 结构化文档。
    root = Path(path)
    # 根据 not root.exists() 判断当前流程该进入哪个处理分支。
    if not root.exists():
        # 在输入或运行状态不满足要求时抛出异常，阻止错误结果继续扩散。
        raise FileNotFoundError(f"Static docs path does not exist: {root}")
    # 根据 root.is_file() 判断当前流程该进入哪个处理分支。
    if root.is_file():
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return _load_file(root)

    # 将 docs 的值保存下来，供后续流程判断或组装响应时使用。
    docs: list[StaticDocument] = []
    # 遍历当前集合中的每一项，逐步完成聚合或转换处理。
    for file_path in sorted(root.rglob("*")):
        # 根据 file_path.suffix.lower() not in {".md", ".markdown",... 判断当前流程该进入哪个处理分支。
        if file_path.suffix.lower() not in {".md", ".markdown", ".jsonl"}:
            # 跳过当前项剩余逻辑，继续处理下一项数据。
            continue
        # 支持目录批量导入，Markdown 走标题切片，JSONL 走逐行结构化导入。
        docs.extend(_load_file(file_path))
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return docs


# 定义 _load_file 相关的处理逻辑，供流程或外部调用复用。
def _load_file(path: Path) -> list[StaticDocument]:
    # 根据文件后缀选择加载方式，Markdown 走标题切片，JSONL 逐行读取。
    suffix = path.suffix.lower()
    # 根据 suffix in {".md", ".markdown"} 判断当前流程该进入哪个处理分支。
    if suffix in {".md", ".markdown"}:
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return _load_markdown_chunks(path)
    # 根据 suffix == ".jsonl" 判断当前流程该进入哪个处理分支。
    if suffix == ".jsonl":
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return _load_jsonl(path)
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return []


# 定义 split_and_extract_metadata 相关的处理逻辑，供流程或外部调用复用。
def split_and_extract_metadata(file_path: str) -> list[Document]:
    """Split a Markdown fault manual by heading hierarchy and enrich RAG metadata.

    The returned Documents can be embedded and upserted into vector stores such as
    Milvus; keep page_content as the chunk text and persist metadata alongside it.
    """
    # 按 Markdown 标题层级切分故障手册，并为每个片段补充故障类型、级别和关键词。

    path = Path(file_path)
    # 将 raw 的值保存下来，供后续流程判断或组装响应时使用。
    raw = path.read_text(encoding="utf-8")
    # 保存当前计算结果，供后续流程判断或组装响应时使用。
    frontmatter, body = _split_frontmatter(raw)
    # 按标题层级切片能保留 runbook 的章节语义，比固定长度切片更利于故障手册检索。
    splitter = MarkdownHeaderTextSplitter(
        # 将 headers_to_split_on 的值保存下来，供后续流程判断或组装响应时使用。
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        # 将 strip_headers 的值保存下来，供后续流程判断或组装响应时使用。
        strip_headers=False,
    )
    # 将 chunks 的值保存下来，供后续流程判断或组装响应时使用。
    chunks = splitter.split_text(body)

    # 将 documents 的值保存下来，供后续流程判断或组装响应时使用。
    documents: list[Document] = []
    # 遍历当前集合中的每一项，逐步完成聚合或转换处理。
    for index, chunk in enumerate(chunks, start=1):
        # frontmatter 提供全局元数据，chunk.metadata 提供当前标题路径，两者合并成检索过滤字段。
        metadata = {
            # 调用 _to_dict 完成当前步骤需要的业务处理。
            **_to_dict(frontmatter.get("metadata")),
            # 调用 frontmatter.items 完成当前步骤需要的业务处理。
            **{key: value for key, value in frontmatter.items() if key != "metadata"},
            # 执行当前业务步骤，推动流程继续向下游推进。
            **chunk.metadata,
        }
        # 将 h1 的值保存下来，供后续流程判断或组装响应时使用。
        h1 = str(metadata.get("h1") or "")
        # 将 h2 的值保存下来，供后续流程判断或组装响应时使用。
        h2 = str(metadata.get("h2") or "")
        # 将 h3 的值保存下来，供后续流程判断或组装响应时使用。
        h3 = str(metadata.get("h3") or "")
        # 将 page_content 的值保存下来，供后续流程判断或组装响应时使用。
        page_content = chunk.page_content.strip()
        # 把新的状态合并回上下文，保证后续步骤读取到最新数据。
        metadata.update(
            {
                # 执行当前业务步骤，推动流程继续向下游推进。
                "doc_id": MANUAL_DOC_ID,
                # 执行当前业务步骤，推动流程继续向下游推进。
                "chunk_index": index,
                # 调用 _build_section 完成当前步骤需要的业务处理。
                "section": _build_section(h1, h2, h3),
                # 以下字段用于后续按故障类型、严重级别、错误码和关键词做过滤或展示。
                "alert_category": _detect_alert_category(h1),
                # 调用 _detect_severity 完成当前步骤需要的业务处理。
                "severity_level": _detect_severity(page_content),
                # 调用 _extract_error_codes 完成当前步骤需要的业务处理。
                "error_code": _extract_error_codes(page_content),
                # 调用 _extract_keywords 完成当前步骤需要的业务处理。
                "keywords": _extract_keywords(page_content),
                # 执行当前业务步骤，推动流程继续向下游推进。
                "last_updated": MANUAL_LAST_UPDATED,
                # 调用 path.as_posix 完成当前步骤需要的业务处理。
                "source": path.as_posix(),
                # 调用 str 完成当前步骤需要的业务处理。
                "file_path": str(path),
            }
        )
        # 把当前结果追加到集合中，逐步构建最终输出。
        documents.append(Document(page_content=page_content, metadata=metadata))
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return documents


# 定义 _load_markdown_chunks 相关的处理逻辑，供流程或外部调用复用。
def _load_markdown_chunks(path: Path) -> list[StaticDocument]:
    # 把一个 Markdown 文件切成多个可检索文档块，并生成稳定的块 ID。
    docs = split_and_extract_metadata(str(path))
    # 根据 not docs 判断当前流程该进入哪个处理分支。
    if not docs:
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return []
    # 将 raw 的值保存下来，供后续流程判断或组装响应时使用。
    raw = path.read_text(encoding="utf-8")
    # 保存当前计算结果，供后续流程判断或组装响应时使用。
    frontmatter, body = _split_frontmatter(raw)
    # 将 base_id 的值保存下来，供后续流程判断或组装响应时使用。
    base_id = str(frontmatter.get("id") or (MANUAL_DOC_ID if path.name == "线上全场景故障排查终极手册（企业级超详细完整版）.md" else path.stem))
    # 将 title 的值保存下来，供后续流程判断或组装响应时使用。
    title = str(frontmatter.get("title") or _extract_markdown_title(body) or path.stem)
    # 将 updated_at 的值保存下来，供后续流程判断或组装响应时使用。
    updated_at = _to_int(frontmatter.get("updated_at"), 0)
    # 将 tags 的值保存下来，供后续流程判断或组装响应时使用。
    tags = _to_tags(frontmatter.get("tags"))
    # 将 static_docs 的值保存下来，供后续流程判断或组装响应时使用。
    static_docs: list[StaticDocument] = []
    # 遍历当前集合中的每一项，逐步完成聚合或转换处理。
    for index, doc in enumerate(docs, start=1):
        # 将 metadata 的值保存下来，供后续流程判断或组装响应时使用。
        metadata = dict(doc.metadata)
        # 第一个 chunk 复用文档 id，其余 chunk 追加序号，保证 Milvus 主键稳定且不冲突。
        chunk_id = base_id if index == 1 else f"{base_id}-{index:04d}"
        # 将 chunk_title 的值保存下来，供后续流程判断或组装响应时使用。
        chunk_title = str(metadata.get("h3") or metadata.get("h2") or metadata.get("h1") or title)
        # 把当前结果追加到集合中，逐步构建最终输出。
        static_docs.append(
            # 调用 StaticDocument 完成当前步骤需要的业务处理。
            StaticDocument(
                # 将 id 的值保存下来，供后续流程判断或组装响应时使用。
                id=chunk_id,
                # 将 text 的值保存下来，供后续流程判断或组装响应时使用。
                text=doc.page_content,
                # 将 doc_type 的值保存下来，供后续流程判断或组装响应时使用。
                doc_type=str(frontmatter.get("doc_type") or "runbook"),
                # 将 title 的值保存下来，供后续流程判断或组装响应时使用。
                title=chunk_title,
                # 将 service 的值保存下来，供后续流程判断或组装响应时使用。
                service=str(frontmatter.get("service") or "global"),
                # 将 component 的值保存下来，供后续流程判断或组装响应时使用。
                component=str(frontmatter.get("component") or metadata.get("alert_category") or ""),
                # 将 tags 的值保存下来，供后续流程判断或组装响应时使用。
                tags=tags,
                # 将 version 的值保存下来，供后续流程判断或组装响应时使用。
                version=str(frontmatter.get("version") or "v1"),
                # 将 updated_at 的值保存下来，供后续流程判断或组装响应时使用。
                updated_at=updated_at,
                # 将 source_uri 的值保存下来，供后续流程判断或组装响应时使用。
                source_uri=str(frontmatter.get("source_uri") or path.as_posix()),
                # 将 metadata 的值保存下来，供后续流程判断或组装响应时使用。
                metadata=metadata,
            )
        )
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return static_docs


# 定义 _load_markdown 相关的处理逻辑，供流程或外部调用复用。
def _load_markdown(path: Path) -> StaticDocument:
    # 把一个 Markdown 文件作为单个文档加载，保留 frontmatter 中的基础元数据。
    raw = path.read_text(encoding="utf-8")
    # 保存当前计算结果，供后续流程判断或组装响应时使用。
    frontmatter, body = _split_frontmatter(raw)
    # 将 title 的值保存下来，供后续流程判断或组装响应时使用。
    title = str(frontmatter.get("title") or _extract_markdown_title(body) or path.stem)
    # 将 doc_id 的值保存下来，供后续流程判断或组装响应时使用。
    doc_id = str(frontmatter.get("id") or path.stem)
    # 将 updated_at 的值保存下来，供后续流程判断或组装响应时使用。
    updated_at = _to_int(frontmatter.get("updated_at"), 0)
    # 将 tags 的值保存下来，供后续流程判断或组装响应时使用。
    tags = _to_tags(frontmatter.get("tags"))
    # 将 metadata 的值保存下来，供后续流程判断或组装响应时使用。
    metadata = _to_dict(frontmatter.get("metadata"))
    # 调用 metadata.setdefault 完成当前步骤需要的业务处理。
    metadata.setdefault("file_path", str(path))
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return StaticDocument(
        # 将 id 的值保存下来，供后续流程判断或组装响应时使用。
        id=doc_id,
        # 将 text 的值保存下来，供后续流程判断或组装响应时使用。
        text=body.strip(),
        # 将 doc_type 的值保存下来，供后续流程判断或组装响应时使用。
        doc_type=str(frontmatter.get("doc_type") or "runbook"),
        # 将 title 的值保存下来，供后续流程判断或组装响应时使用。
        title=title,
        # 将 service 的值保存下来，供后续流程判断或组装响应时使用。
        service=str(frontmatter.get("service") or "global"),
        # 将 component 的值保存下来，供后续流程判断或组装响应时使用。
        component=str(frontmatter.get("component") or ""),
        # 将 tags 的值保存下来，供后续流程判断或组装响应时使用。
        tags=tags,
        # 将 version 的值保存下来，供后续流程判断或组装响应时使用。
        version=str(frontmatter.get("version") or "v1"),
        # 将 updated_at 的值保存下来，供后续流程判断或组装响应时使用。
        updated_at=updated_at,
        # 将 source_uri 的值保存下来，供后续流程判断或组装响应时使用。
        source_uri=str(frontmatter.get("source_uri") or path.as_posix()),
        # 将 metadata 的值保存下来，供后续流程判断或组装响应时使用。
        metadata=metadata,
    )


# 定义 _load_jsonl 相关的处理逻辑，供流程或外部调用复用。
def _load_jsonl(path: Path) -> list[StaticDocument]:
    # 逐行读取 JSONL 文档，每一行都会转成一条可入库的静态知识记录。
    docs: list[StaticDocument] = []
    # 遍历当前集合中的每一项，逐步完成聚合或转换处理。
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        # 根据 not line.strip() 判断当前流程该进入哪个处理分支。
        if not line.strip():
            # 跳过当前项剩余逻辑，继续处理下一项数据。
            continue
        # 将 payload 的值保存下来，供后续流程判断或组装响应时使用。
        payload = json.loads(line)
        # JSONL 允许每行是一个独立文档，缺省 id/title 时用文件名和行号兜底。
        payload.setdefault("id", f"{path.stem}-{line_number}")
        # 调用 payload.setdefault 完成当前步骤需要的业务处理。
        payload.setdefault("title", payload["id"])
        # 调用 payload.setdefault 完成当前步骤需要的业务处理。
        payload.setdefault("metadata", {})
        # 保存当前计算结果，供后续流程判断或组装响应时使用。
        payload["metadata"]["file_path"] = str(path)
        # 保存当前计算结果，供后续流程判断或组装响应时使用。
        payload["metadata"]["line_number"] = line_number
        # 把当前结果追加到集合中，逐步构建最终输出。
        docs.append(StaticDocument.model_validate(payload))
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return docs


# 定义 _split_frontmatter 相关的处理逻辑，供流程或外部调用复用。
def _split_frontmatter(raw: str) -> tuple[dict[str, Any], str]:
    # 拆分 Markdown 顶部的 frontmatter 元数据和正文内容。
    if not raw.startswith("---"):
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return {}, raw
    # 将 parts 的值保存下来，供后续流程判断或组装响应时使用。
    parts = raw.split("---", 2)
    # 根据 len(parts) < 3 判断当前流程该进入哪个处理分支。
    if len(parts) < 3:
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return {}, raw
    # 将 metadata 的值保存下来，供后续流程判断或组装响应时使用。
    metadata = yaml.safe_load(parts[1]) or {}
    # 根据 not isinstance(metadata, dict) 判断当前流程该进入哪个处理分支。
    if not isinstance(metadata, dict):
        # 将 metadata 的值保存下来，供后续流程判断或组装响应时使用。
        metadata = {}
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return metadata, parts[2]


# 定义 _extract_markdown_title 相关的处理逻辑，供流程或外部调用复用。
def _extract_markdown_title(text: str) -> str | None:
    # 从 Markdown 正文中提取一级标题，作为文档默认标题。
    for line in text.splitlines():
        # 将 stripped 的值保存下来，供后续流程判断或组装响应时使用。
        stripped = line.strip()
        # 根据 stripped.startswith("# ") 判断当前流程该进入哪个处理分支。
        if stripped.startswith("# "):
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return stripped[2:].strip()
    return None


# 定义 _build_section 相关的处理逻辑，供流程或外部调用复用。
def _build_section(h1: str, h2: str, h3: str) -> str:
    # 把标题层级拼成章节路径，后续检索结果可显示文档片段来自哪里。
    return "_".join(_normalize_section_part(part) for part in (h1, h2, h3) if part.strip())


# 定义 _normalize_section_part 相关的处理逻辑，供流程或外部调用复用。
def _normalize_section_part(value: str) -> str:
    # 清理章节名中的 Markdown 符号和空格，生成稳定的章节路径片段。
    value = value.replace(r"\.", ".").strip()
    # 将 value 的值保存下来，供后续流程判断或组装响应时使用。
    value = re.sub(r"^#+\s*", "", value)
    # 将 value 的值保存下来，供后续流程判断或组装响应时使用。
    value = re.sub(r"\s+", "_", value)
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return value.strip("_")


# 定义 _detect_alert_category 相关的处理逻辑，供流程或外部调用复用。
def _detect_alert_category(h1: str) -> str:
    # 根据一级标题判断文档片段属于 MQ、Redis、MySQL 等哪类故障。
    mapping = (
        # 执行当前业务步骤，推动流程继续向下游推进。
        ("消息队列", "MQ"),
        # 执行当前业务步骤，推动流程继续向下游推进。
        ("应用服务", "应用服务"),
        # 执行当前业务步骤，推动流程继续向下游推进。
        ("JVM", "JVM"),
        # 执行当前业务步骤，推动流程继续向下游推进。
        ("MySQL", "MySQL"),
        # 执行当前业务步骤，推动流程继续向下游推进。
        ("Redis", "Redis"),
        # 执行当前业务步骤，推动流程继续向下游推进。
        ("网络、网关、注册中心", "网络/网关"),
        # 执行当前业务步骤，推动流程继续向下游推进。
        ("容器K8s", "K8s"),
        # 执行当前业务步骤，推动流程继续向下游推进。
        ("日志、监控、告警", "日志监控"),
        # 执行当前业务步骤，推动流程继续向下游推进。
        ("业务逻辑", "业务逻辑"),
    )
    # 遍历当前集合中的每一项，逐步完成聚合或转换处理。
    for needle, category in mapping:
        # 根据 needle in h1 判断当前流程该进入哪个处理分支。
        if needle in h1:
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return category
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return "其他"


# 定义 _detect_severity 相关的处理逻辑，供流程或外部调用复用。
def _detect_severity(page_content: str) -> str:
    # 根据片段中的严重程度关键词推断 P0/P1/P2 级别。
    if any(token in page_content for token in ("核心业务致命故障", "高危", "雪崩")):
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return "P0"
    # 根据 any(token in page_content for token in ("严重", "暴涨", ... 判断当前流程该进入哪个处理分支。
    if any(token in page_content for token in ("严重", "暴涨", "飙升")):
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return "P1"
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return "P2"


# 定义 _extract_error_codes 相关的处理逻辑，供流程或外部调用复用。
def _extract_error_codes(page_content: str) -> list[str]:
    # 从文档片段中提取常见错误码或异常短语，供检索过滤和展示使用。
    seen: set[str] = set()
    # 将 values 的值保存下来，供后续流程判断或组装响应时使用。
    values: list[str] = []
    # 遍历当前集合中的每一项，逐步完成聚合或转换处理。
    for match in ERROR_CODE_PATTERN.finditer(page_content):
        # 将 value 的值保存下来，供后续流程判断或组装响应时使用。
        value = match.group(1)
        # 将 key 的值保存下来，供后续流程判断或组装响应时使用。
        key = value.lower()
        # 根据 key not in seen 判断当前流程该进入哪个处理分支。
        if key not in seen:
            # 调用 seen.add 完成当前步骤需要的业务处理。
            seen.add(key)
            # 把当前结果追加到集合中，逐步构建最终输出。
            values.append(value)
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return values


# 定义 _extract_keywords 相关的处理逻辑，供流程或外部调用复用。
def _extract_keywords(page_content: str) -> list[str]:
    # 从粗体词、关键小节和高频词中抽取代表文档语义的关键词。
    candidates: list[str] = []
    # 粗体词通常是手册中的关键概念，优先作为候选关键词。
    candidates.extend(_clean_keyword(item) for item in re.findall(r"\*\*([^*]{2,40})\*\*", page_content))

    # 遍历当前集合中的每一项，逐步完成聚合或转换处理。
    for line in page_content.splitlines():
        # 根据 any(label in line for label in ("故障定义", "根因", "解决方案"... 判断当前流程该进入哪个处理分支。
        if any(label in line for label in ("故障定义", "根因", "解决方案", "处理建议", "排查要点", "排查步骤")):
            # 关键小节标题附近的信息通常比普通正文更能代表故障语义。
            candidates.extend(_keyword_tokens(line))

    # 把一组结果合并到集合中，扩展后续可使用的数据范围。
    candidates.extend(
        # 执行当前业务步骤，推动流程继续向下游推进。
        token
        # 遍历当前集合中的每一项，逐步完成聚合或转换处理。
        for token, _ in Counter(_keyword_tokens(page_content)).most_common(30)
    )

    # 将 keywords 的值保存下来，供后续流程判断或组装响应时使用。
    keywords: list[str] = []
    # 将 seen 的值保存下来，供后续流程判断或组装响应时使用。
    seen: set[str] = set()
    # 遍历当前集合中的每一项，逐步完成聚合或转换处理。
    for candidate in candidates:
        # 将 keyword 的值保存下来，供后续流程判断或组装响应时使用。
        keyword = _clean_keyword(candidate)
        # 根据 not keyword or keyword in STOPWORDS or keyword in se... 判断当前流程该进入哪个处理分支。
        if not keyword or keyword in STOPWORDS or keyword in seen:
            # 跳过当前项剩余逻辑，继续处理下一项数据。
            continue
        # 调用 seen.add 完成当前步骤需要的业务处理。
        seen.add(keyword)
        # 把当前结果追加到集合中，逐步构建最终输出。
        keywords.append(keyword)
        # 根据 len(keywords) >= 10 判断当前流程该进入哪个处理分支。
        if len(keywords) >= 10:
            # 满足停止条件后退出循环，避免继续执行无效处理。
            break
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return keywords[:10]


# 定义 _keyword_tokens 相关的处理逻辑，供流程或外部调用复用。
def _keyword_tokens(text: str) -> list[str]:
    # 从中英文混合文本里提取可作为关键词的短词片段。
    tokens = re.findall(r"[\u4e00-\u9fffA-Za-z0-9][\u4e00-\u9fffA-Za-z0-9+/._-]{1,24}", text)
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return [token for token in tokens if len(token) >= 2]


# 定义 _clean_keyword 相关的处理逻辑，供流程或外部调用复用。
def _clean_keyword(value: str) -> str:
    # 去掉关键词周围的 Markdown 符号和标点，限制关键词长度。
    value = re.sub(r"[*`#>\-：:，,。；;、（）()\[\]【】]", "", str(value)).strip()
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return value[:40]


# 定义 _to_tags 相关的处理逻辑，供流程或外部调用复用。
def _to_tags(value: Any) -> list[str]:
    # 把 frontmatter 中的标签字段统一转换成字符串列表。
    if value is None:
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return []
    # 根据 isinstance(value, str) 判断当前流程该进入哪个处理分支。
    if isinstance(value, str):
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return [item.strip() for item in value.split(",") if item.strip()]
    # 根据 isinstance(value, list) 判断当前流程该进入哪个处理分支。
    if isinstance(value, list):
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return [str(item).strip() for item in value if str(item).strip()]
    # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
    return []


# 定义 _to_dict 相关的处理逻辑，供流程或外部调用复用。
def _to_dict(value: Any) -> dict[str, Any]:
    # 确保 metadata 字段一定是字典，写错类型时退回空字典。
    return value if isinstance(value, dict) else {}


# 定义 _to_int 相关的处理逻辑，供流程或外部调用复用。
def _to_int(value: Any, default: int) -> int:
    # 把 frontmatter 里的时间戳等字段转成整数，失败时使用默认值。
    try:
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return int(value)
    # 捕获当前步骤中的异常，转换为日志、指标或降级结果。
    except (TypeError, ValueError):
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return default


# 根据 __name__ == "__main__" 判断当前流程该进入哪个处理分支。
if __name__ == "__main__":
    # 将 input_file 的值保存下来，供后续流程判断或组装响应时使用。
    input_file = "线上全场景故障排查终极手册（企业级超详细完整版）.md"
    # 将 file_path 的值保存下来，供后续流程判断或组装响应时使用。
    file_path = Path(input_file)
    # 根据 not file_path.exists() 判断当前流程该进入哪个处理分支。
    if not file_path.exists():
        # 将 file_path 的值保存下来，供后续流程判断或组装响应时使用。
        file_path = Path("data/static_docs") / input_file
    # 将 documents 的值保存下来，供后续流程判断或组装响应时使用。
    documents = split_and_extract_metadata(str(file_path))
    # 遍历当前集合中的每一项，逐步完成聚合或转换处理。
    for index, document in enumerate(documents, start=1):
        # 调用 print 完成当前步骤需要的业务处理。
        print(f"[{index}] content preview:")
        # 调用 print 完成当前步骤需要的业务处理。
        print(document.page_content[:200].replace("\n", "\\n"))
        # 调用 print 完成当前步骤需要的业务处理。
        print("metadata:")
        # 保存当前计算结果，供后续流程判断或组装响应时使用。
        print(json.dumps(document.metadata, ensure_ascii=False, indent=2))
