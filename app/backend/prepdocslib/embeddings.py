import asyncio
import logging
import random
from abc import ABC
from collections.abc import Awaitable, Callable
from urllib.parse import urljoin

import aiohttp
import tiktoken
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, RateLimitError
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)
from typing_extensions import TypedDict

logger = logging.getLogger("scripts")


class EmbeddingBatch:
    """Represents a batch of text that is going to be embedded."""

    def __init__(self, texts: list[str], token_length: int):
        self.texts = texts
        self.token_length = token_length


class ExtraArgs(TypedDict, total=False):
    dimensions: int


class OpenAIEmbeddings(ABC):
    """Client wrapper that handles batching, retries, and token accounting."""

    SUPPORTED_BATCH_MODEL = {
        "text-embedding-ada-002": {"token_limit": 8100, "max_batch_size": 16},
        "text-embedding-3-small": {"token_limit": 8100, "max_batch_size": 16},
        "text-embedding-3-large": {"token_limit": 8100, "max_batch_size": 16},
    }
    SUPPORTED_DIMENSIONS_MODEL = {
        "text-embedding-ada-002": False,
        "text-embedding-3-small": True,
        "text-embedding-3-large": True,
    }

    def __init__(
        self,
        open_ai_client: AsyncOpenAI,
        open_ai_model_name: str,
        open_ai_dimensions: int,
        *,
        disable_batch: bool = False,
        azure_deployment_name: str | None = None,
        azure_endpoint: str | None = None,
    ):
        self.open_ai_client = open_ai_client
        self.open_ai_model_name = open_ai_model_name
        self.open_ai_dimensions = open_ai_dimensions
        self.disable_batch = disable_batch
        self.azure_deployment_name = azure_deployment_name
        self.azure_endpoint = azure_endpoint.rstrip("/") if azure_endpoint else None

    @property
    def _api_model(self) -> str:
        return self.azure_deployment_name or self.open_ai_model_name

    def before_retry_sleep(self, retry_state):
        logger.info("Rate limited on the OpenAI embeddings API, sleeping before retrying...")

    @staticmethod
    def _retry_delay(exc: Exception, attempt: int) -> float:
        response = getattr(exc, "response", None)
        retry_after = response.headers.get("retry-after") if response is not None else None
        if retry_after:
            try:
                return max(0.0, float(retry_after))
            except (TypeError, ValueError):
                pass
        return min(60.0, (2**attempt) + random.uniform(0.0, 1.0))

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        if isinstance(exc, (RateLimitError, APIConnectionError, APITimeoutError)):
            return True
        status_code = getattr(exc, "status_code", None)
        return status_code is not None and 500 <= status_code < 600

    async def _create_embedding_batch_with_retry(
        self, batch: EmbeddingBatch, dimensions_args: ExtraArgs, max_attempts: int = 15
    ) -> list[list[float]]:
        async for attempt in AsyncRetrying(
            retry=retry_if_exception_type((RateLimitError, APIConnectionError, APITimeoutError)),
            wait=wait_random_exponential(min=15, max=60),
            stop=stop_after_attempt(max_attempts),
            before_sleep=self.before_retry_sleep,
        ):
            with attempt:
                emb_response = await self.open_ai_client.embeddings.create(
                    model=self._api_model, input=batch.texts, **dimensions_args
                )
                logger.info(
                    "Computed embeddings in batch. Batch size: %d, Token count: %d",
                    len(batch.texts),
                    batch.token_length,
                )
                return [data.embedding for data in emb_response.data]
        raise RuntimeError("Embedding retry loop exited unexpectedly")

    def calculate_token_length(self, text: str):
        encoding = tiktoken.encoding_for_model(self.open_ai_model_name)
        return len(encoding.encode(text))

    def split_text_into_batches(self, texts: list[str]) -> list[EmbeddingBatch]:
        batch_info = OpenAIEmbeddings.SUPPORTED_BATCH_MODEL.get(self.open_ai_model_name)
        if not batch_info:
            raise NotImplementedError(
                f"Model {self.open_ai_model_name} is not supported with batch embedding operations"
            )

        batch_token_limit = batch_info["token_limit"]
        batch_max_size = batch_info["max_batch_size"]
        encoding = tiktoken.encoding_for_model(self.open_ai_model_name)
        batches: list[EmbeddingBatch] = []
        batch: list[str] = []
        batch_token_length = 0
        for text in texts:
            text_token_length = self.calculate_token_length(text)

            # If a single text exceeds the token limit, split it into smaller chunks
            if text_token_length > batch_token_limit:
                logger.warning(
                    "Text with %d tokens exceeds batch token limit of %d, splitting into smaller chunks",
                    text_token_length,
                    batch_token_limit,
                )
                tokens = encoding.encode(text)
                for start in range(0, len(tokens), batch_token_limit):
                    chunk = encoding.decode(tokens[start : start + batch_token_limit])
                    chunk_token_length = len(tokens[start : start + batch_token_limit])

                    if batch_token_length + chunk_token_length >= batch_token_limit and len(batch) > 0:
                        batches.append(EmbeddingBatch(batch, batch_token_length))
                        batch = []
                        batch_token_length = 0

                    batch.append(chunk)
                    batch_token_length = batch_token_length + chunk_token_length
                    if len(batch) == batch_max_size:
                        batches.append(EmbeddingBatch(batch, batch_token_length))
                        batch = []
                        batch_token_length = 0
                continue

            if batch_token_length + text_token_length >= batch_token_limit and len(batch) > 0:
                batches.append(EmbeddingBatch(batch, batch_token_length))
                batch = []
                batch_token_length = 0

            batch.append(text)
            batch_token_length = batch_token_length + text_token_length
            if len(batch) == batch_max_size:
                batches.append(EmbeddingBatch(batch, batch_token_length))
                batch = []
                batch_token_length = 0

        if len(batch) > 0:
            batches.append(EmbeddingBatch(batch, batch_token_length))

        return batches

    async def create_embedding_batch(self, texts: list[str], dimensions_args: ExtraArgs) -> list[list[float]]:
        batches = self.split_text_into_batches(texts)
        embeddings = []
        for batch in batches:
            embeddings.extend(await self._create_embedding_batch_with_retry(batch, dimensions_args))

        return embeddings

    def split_prepared_into_batches(self, prepared: list[tuple[str, int, object]]) -> list[EmbeddingBatch]:
        batches: list[EmbeddingBatch] = []
        batch: list[str] = []
        batch_token_length = 0
        for prepared_item in prepared:
            text, token_length = prepared_item[:2]
            if token_length > 8100:
                raise ValueError("Prepared embedding input exceeds the 8100-token batch limit")
            if batch and (batch_token_length + token_length > 8100 or len(batch) == 16):
                batches.append(EmbeddingBatch(batch, batch_token_length))
                batch = []
                batch_token_length = 0
            batch.append(text)
            batch_token_length += token_length
        if batch:
            batches.append(EmbeddingBatch(batch, batch_token_length))
        return batches

    async def create_embeddings_concurrent(
        self, prepared: list[tuple[str, int, object]], concurrency: int = 8
    ) -> list[list[float]]:
        if concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        dimensions_args: ExtraArgs = (
            {"dimensions": self.open_ai_dimensions}
            if OpenAIEmbeddings.SUPPORTED_DIMENSIONS_MODEL.get(self.open_ai_model_name)
            else {}
        )
        batches = self.split_prepared_into_batches(prepared)
        semaphore = asyncio.Semaphore(concurrency)

        async def create_batch(batch: EmbeddingBatch) -> list[list[float]]:
            async with semaphore:
                return await self._create_embedding_batch_with_retry(batch, dimensions_args)

        results = await asyncio.gather(*(create_batch(batch) for batch in batches))
        return [vector for batch_result in results for vector in batch_result]

    async def create_embedding_single(self, text: str, dimensions_args: ExtraArgs) -> list[float]:
        async for attempt in AsyncRetrying(
            retry=retry_if_exception_type(RateLimitError),
            wait=wait_random_exponential(min=15, max=60),
            stop=stop_after_attempt(15),
            before_sleep=self.before_retry_sleep,
        ):
            with attempt:
                emb_response = await self.open_ai_client.embeddings.create(
                    model=self._api_model, input=text, **dimensions_args
                )
                logger.info(
                    "Computed embedding for text section. Character count: %d",
                    len(text),
                )

        return emb_response.data[0].embedding

    async def create_embeddings(self, texts: list[str]) -> list[list[float]]:
        dimensions_args: ExtraArgs = (
            {"dimensions": self.open_ai_dimensions}
            if OpenAIEmbeddings.SUPPORTED_DIMENSIONS_MODEL.get(self.open_ai_model_name)
            else {}
        )

        if not self.disable_batch and self.open_ai_model_name in OpenAIEmbeddings.SUPPORTED_BATCH_MODEL:
            return await self.create_embedding_batch(texts, dimensions_args)

        return [await self.create_embedding_single(text, dimensions_args) for text in texts]


class ImageEmbeddings:
    """
    Class for using image embeddings from Azure AI Vision
    To learn more, please visit https://learn.microsoft.com/azure/ai-services/computer-vision/how-to/image-retrieval#call-the-vectorize-image-api
    """

    def __init__(self, endpoint: str, token_provider: Callable[[], Awaitable[str]]):
        self.token_provider = token_provider
        self.endpoint = endpoint

    async def create_embedding_for_image(self, image_bytes: bytes) -> list[float]:
        endpoint = urljoin(self.endpoint, "computervision/retrieval:vectorizeImage")
        params = {"api-version": "2024-02-01", "model-version": "2023-04-15"}
        headers = {"Authorization": "Bearer " + await self.token_provider()}

        async with aiohttp.ClientSession(headers=headers) as session:
            async for attempt in AsyncRetrying(
                retry=retry_if_exception_type(Exception),
                wait=wait_random_exponential(min=15, max=60),
                stop=stop_after_attempt(15),
                before_sleep=self.before_retry_sleep,
            ):
                with attempt:
                    async with session.post(url=endpoint, params=params, data=image_bytes) as resp:
                        resp_json = await resp.json()
                        return resp_json["vector"]
        raise ValueError("Failed to get image embedding after multiple retries.")

    async def create_embedding_for_text(self, q: str):
        endpoint = urljoin(self.endpoint, "computervision/retrieval:vectorizeText")
        headers = {"Content-Type": "application/json"}
        params = {"api-version": "2024-02-01", "model-version": "2023-04-15"}
        data = {"text": q}
        headers["Authorization"] = "Bearer " + await self.token_provider()

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url=endpoint,
                params=params,
                headers=headers,
                json=data,
                raise_for_status=True,
            ) as response:
                json = await response.json()
                return json["vector"]
        raise ValueError("Failed to get text embedding after multiple retries.")

    def before_retry_sleep(self, retry_state):
        logger.info("Rate limited on the Vision embeddings API, sleeping before retrying...")
