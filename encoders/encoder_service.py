import requests
import torch
import time

from loguru import logger
from tqdm import tqdm
from typing import Union
import aiohttp
import asyncio


url = "https://fun-embedder-a83c86056e29.herokuapp.com"


def init():
    pass


# ...


async def _encode_batch(text_list: list[str], batch_size: int = 8) -> torch.Tensor:
    ret = []
    logger.info(f"Encoding {len(text_list)} chunks")
    start = time.monotonic()

    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in tqdm(
            range(0, len(text_list), batch_size),
            total=len(text_list) // batch_size,
            leave=False,
        ):
            text_batch = text_list[i : i + batch_size]
            task = asyncio.ensure_future(encode_batch(session, text_batch))
            tasks.append(task)

        responses = await asyncio.gather(*tasks)

        for res in responses:
            ret.append(torch.tensor(res["embeddings"]))

    logger.info(f"Encoding took {time.monotonic() - start:.2f} seconds")
    return torch.cat(ret, dim=0)


async def encode_batch(
    session: aiohttp.ClientSession, text_batch: list[str]
) -> aiohttp.ClientResponse:
    async with session.post(url + "/embed_batch", json=text_batch) as response:
        if response.status != 200:
            raise Exception(
                f"Failed to encode text: {text_batch}. Got text: {await response.text()}"
            )
        return await response.json()


def encode(text: Union[str, list[str]]) -> torch.Tensor:
    if isinstance(text, str):
        text = [text]
    return _encode_batch(text, batch_size=8)
