from __future__ import annotations

import hashlib
import logging
import random
from dataclasses import dataclass

from openai import AsyncOpenAI

# 将 logger 的值保存下来，供后续流程判断或组装响应时使用。
logger = logging.getLogger(__name__)


@dataclass(slots=True)
# 定义 EmbeddingClient 组件，集中管理这个模块的状态和行为。
class EmbeddingClient:
    # 将 api_key 的值保存下来，供后续流程判断或组装响应时使用。
    api_key: str = ""
    # 将 base_url 的值保存下来，供后续流程判断或组装响应时使用。
    base_url: str | None = None
    # 将 model 的值保存下来，供后续流程判断或组装响应时使用。
    model: str = "text-embedding-v4"
    # 将 dimension 的值保存下来，供后续流程判断或组装响应时使用。
    dimension: int = 1536
    # 将 mock_enabled 的值保存下来，供后续流程判断或组装响应时使用。
    mock_enabled: bool = False

    # 定义 embed 相关的处理逻辑，供流程或外部调用复用。
    async def embed(self, text: str) -> list[float]:
        # 把文本转换成向量；真实环境调用 Embedding 服务，本地或无密钥时走稳定的模拟向量。
        if self.mock_enabled or not self.api_key:
            # 记录关键运行信息，方便排查流程进展和异常现场。
            logger.info("Embedding mock enabled text_chars=%s", len(text))
            # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
            return self._mock_embedding(text)

        # 将 client 的值保存下来，供后续流程判断或组装响应时使用。
        client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        # 将 response 的值保存下来，供后续流程判断或组装响应时使用。
        response = await client.embeddings.create(model=self.model, input=text)
        # 将 embedding 的值保存下来，供后续流程判断或组装响应时使用。
        embedding = response.data[0].embedding
        # 记录关键运行信息，方便排查流程进展和异常现场。
        logger.info("Embedding completed model=%s dimension=%s", self.model, len(embedding))
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return list(embedding)

    # 定义 _mock_embedding 相关的处理逻辑，供流程或外部调用复用。
    def _mock_embedding(self, text: str) -> list[float]:
        # 按文本内容生成可重复的随机向量，让测试环境每次检索结果保持稳定。
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)
        # 将 rng 的值保存下来，供后续流程判断或组装响应时使用。
        rng = random.Random(seed)
        # 返回当前步骤整理后的结果，交给调用方或流程下游继续使用。
        return [rng.uniform(-1.0, 1.0) for _ in range(self.dimension)]
