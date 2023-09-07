#!/opt/homebrew/bin/python3

import config
import helper
import web_search

from readable import Readable

# 3rd -party libs
import bz2
import os
import pickle
import time
import torch

from loguru import logger
from nltk.tokenize import sent_tokenize
from pypdf import PdfReader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from typing import Union


def get_context(question_embd, book_name):
  embd_fn = f'{config.EMBD_DIR}/{book_name}.bin'
  if os.path.exists(embd_fn): all_chunks, embeddings = helper.load_data(embd_fn)
  else:
    logger.info(f'embedding {book_name}')
    reader = PdfReader(f'{config.BOOK_DIR}/{book_name}.pdf')
    all_text = ' '.join([x.strip() for page in reader.pages for x in page.extract_text().split('\n')])
    all_sents = sent_tokenize(all_text)
    all_chunks = helper.get_chunks(all_sents, max_len=config.MAX_LEN)
    embeddings = config.ENCODER.encode(all_chunks)
    helper.save_data((all_chunks, embeddings), embd_fn)

  similarity_score = (question_embd @ embeddings.T).squeeze()
  chunk_ids = similarity_score.argsort().tolist()[-config.PASSAGE_K:]
  return similarity_score[chunk_ids], [all_chunks[x] for x in chunk_ids]


if __name__ == '__main__':
  '''
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
  '''

  config.ENCODER.init()
  books = ['riscv_isa_privileged', 'xv6_book']  # TODO (rohan): this should be taken from yaml
  with open(config.QUESTION_FILE, 'r') as f: question = f.read()  # TODO (rohan): this should be run in loop, to avoid reinitialization of encoder model
  question_embd = config.ENCODER.encode(question)
  scores, chunks = [], []
  for book_name in books:
    sc, chks = get_context(question_embd, book_name)
    scores.append(sc)
    chunks.extend(chks)

  scores = torch.cat(scores)
  chunk_ids = scores.argsort().tolist()[-config.PASSAGE_K:]

  # create prompt
  # TODO (rohan): should I initialize the template in config?
  with open(config.PROMPT_TEMPLATE, 'r') as f: prompt_template = f.read()
  passages = '\n-\n'.join((chunks[x] for x in chunk_ids))
  prompt = prompt_template.format(passages=passages, question=question)
  with open(config.PROMPT_FILE, 'w') as f: f.write(prompt)
