import requests
import torch

from typing import Union


def init():
    pass


def encode(text: Union[str, list[str]]) -> torch.Tensor:
    url = "https://fun-embedder-a83c86056e29.herokuapp.com"
    if isinstance(text, str):
        res = requests.post(url + "/embed", params={"text": text})
    else:
        res = requests.post(url + "/embed_batch", json=text)

    if res.status_code != 200:
        raise Exception(f"Failed to encode text: {text}. Got text: {res.text}")
    ret = torch.tensor(res.json()["embeddings"])
    if ret.ndim == 1:
        ret = ret.unsqueeze(0)
    return ret
