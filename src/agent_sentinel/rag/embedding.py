from __future__ import annotations

import hashlib
import logging
import random
from dataclasses import dataclass

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class EmbeddingClient:
    api_key: str = ""
    base_url: str | None = None
    model: str = "text-embedding-v4"
    dimension: int = 1536
    mock_enabled: bool = False

    async def embed(self, text: str) -> list[float]:
        # 方法说明：封装当前处理步骤，保持调用方关注输入和输出。
        if self.mock_enabled or not self.api_key:
            logger.info("Embedding mock enabled text_chars=%s", len(text))
            return self._mock_embedding(text)

        client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        response = await client.embeddings.create(model=self.model, input=text)
        embedding = response.data[0].embedding
        logger.info("Embedding completed model=%s dimension=%s", self.model, len(embedding))
        return list(embedding)

    def _mock_embedding(self, text: str) -> list[float]:
        # 方法说明：封装当前处理步骤，保持调用方关注输入和输出。
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)
        rng = random.Random(seed)
        return [rng.uniform(-1.0, 1.0) for _ in range(self.dimension)]
