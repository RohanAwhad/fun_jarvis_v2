#!/opt/homebrew/bin/python3

import config
import helper
import web_search

from readable import Readable

# 3rd -party libs
import bz2
import os
import pandas as pd
import pickle
import pyperclip
import time
import torch

from loguru import logger
from nltk.tokenize import sent_tokenize
from pypdf import PdfReader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from typing import Union
from sklearn.metrics.pairwise import cosine_similarity
import requests

pd.set_option("display.max_columns", None)


def get_context(question_embd, book_name):
    embd_fn = f"{config.EMBD_DIR}/{book_name}.{config.ENCODER.__name__}.bin"
    if os.path.exists(embd_fn):
        all_chunks, embeddings = helper.load_data(embd_fn)
    else:
        if book_name.endswith(".txt"):
            with open(f"{config.BOOK_DIR}/{book_name}", "r") as f:
                all_text = f.read()
        else:
            logger.info(f"embedding {book_name}")
            reader = PdfReader(f"{config.BOOK_DIR}/{book_name}.pdf")
            all_text = " ".join(
                [
                    x.strip()
                    for page in reader.pages
                    for x in page.extract_text().split("\n")
                ]
            )

        all_sents = sent_tokenize(all_text)
        all_chunks = helper.get_chunks(all_sents, max_len=config.MAX_LEN)
        embeddings = config.ENCODER.encode(all_chunks)
        helper.save_data((all_chunks, embeddings), embd_fn)

    # similarity_score = (question_embd @ embeddings.T).squeeze()
    similarity_score = torch.tensor(
        cosine_similarity(question_embd, embeddings).squeeze()
    )
    print(f"book: {book_name}, max score: {similarity_score.max()}")
    chunk_ids = similarity_score.argsort().tolist()[-config.PASSAGE_K :]
    return similarity_score[chunk_ids], [all_chunks[x] for x in chunk_ids]


if __name__ == "__main__":
    """
    ducky_res = web_search.search('Custom bootloader PMP configuration for limiting OS memory access')
    internet_res = []
    for url in ducky_res:
      try:
        page = Readable(url)
        print(page.text)
        internet_res.append(page)
        if len(internet_res) >= config.SEARCH_K: break
      except Exception as e:
        logger.error(e)
    exit(0)
    """

    config.ENCODER.init()

    # books = [
    #     "data_science_for_business",
    # ]

    # books = [
    #     "data_mgmt_for_multimedia_retrieval",
    #     "intro_to_info_retrieval",
    #     "search_engines_info_retrieval_in_practice",
    #     "mwdb_all_lectures_transcription.txt",
    # ]

    books = [
        "riscv_isa_privileged",
        "xv6_book",
        "the_little_os_book",
        "modern_os",
        "os_concepts",
    ]  # TODO (rohan): this should be taken from yaml
    # books = ['swe_tb', 'sommerville_se']
    with open(config.QUESTION_FILE, "r") as f:
        question = (
            f.read()
        )  # TODO (rohan): this should be run in loop, to avoid reinitialization of encoder model
    question_embd = config.ENCODER.encode(question)
    scores, chunks, book_names = [], [], []
    for book_name in books:
        sc, chks = get_context(question_embd, book_name)
        scores.append(sc)
        chunks.extend(chks)
        book_names.extend([book_name] * len(chks))

    scores = torch.cat(scores)
    chunk_ids = scores.argsort().tolist()[-config.SEARCH_K :][::-1]

    print("Before Reranking")
    print(
        pd.DataFrame(
            {
                "chunk": [chunks[x] for x in chunk_ids],
                "score": scores[chunk_ids],
                "book": [book_names[x] for x in chunk_ids],
            }
        )
    )

    # rerank
    res = requests.post(
        "https://fun-reranker-84b02fb456da.herokuapp.com/rerank",
        # "http://localhost:8000/rerank",
        json={"query": question, "passages": chunks},
    )
    if res.status_code == 200:
        rerank_scores = torch.tensor(res.json()["scores"])
    else:
        logger.error(f"reranking failed with status code {res.status_code}")
        logger.error(res.text)
        rerank_scores = scores

    chunk_ids = rerank_scores.argsort().tolist()[-config.SEARCH_K :][::-1]

    print("After Reranking")
    print(
        pd.DataFrame(
            {
                "chunk": [chunks[x] for x in chunk_ids],
                "reranked_score": rerank_scores[chunk_ids],
                "book": [book_names[x] for x in chunk_ids],
                "cosine_score": scores[chunk_ids],
            }
        )
    )

    # create prompt
    # TODO (rohan): should I initialize the template in config?
    with open(config.PROMPT_TEMPLATE, "r") as f:
        prompt_template = f.read()
    passages = "\n-\n".join((f"[{i+1}] {chunks[x]}" for i, x in enumerate(chunk_ids)))
    prompt = prompt_template.format(passages=passages, question=question)
    with open(config.PROMPT_FILE, "w") as f:
        f.write(prompt)
    pyperclip.copy(prompt)
