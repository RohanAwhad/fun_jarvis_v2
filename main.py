#!/opt/homebrew/bin/python3

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

import encoder

k = 15

def save_data(data_tuple, fd):
  bin_data = pickle.dumps(data_tuple)
  cmprsd_bin_data = bz2.compress(bin_data)
  logger.debug(f'- orginal size: {len(bin_data) / 1e6} MB')
  logger.debug(f'- compressed size: {len(cmprsd_bin_data) / 1e6} MB')
  with open(f'embeddings/{fd}.bin', 'wb') as f: f.write(cmprsd_bin_data)

def load_data(fd):
  with open(f'embeddings/{fd}.bin', 'rb') as f: cmprsd_bin_data = f.read()
  bin_data = bz2.decompress(cmprsd_bin_data)
  return pickle.loads(bin_data)

#  book_name = 'riscv_isa_privileged'
def get_context(question_embd, book_name):
  if os.path.exists(f'embeddings/{book_name}.bin'): all_chunks, embeddings = load_data(book_name)
  else:
    logger.info(f'embedding {book_name}')
    reader = PdfReader(f'books/{book_name}.pdf')
    all_text = ' '.join([x.strip() for page in reader.pages for x in page.extract_text().split('\n')])
    all_sents = sent_tokenize(all_text)
    all_chunks = encoder.get_chunks(all_sents, max_len=1200)
    embeddings = encoder.encode(all_chunks)
    save_data((all_chunks, embeddings), book_name)

  similarity_score = (question_embd @ embeddings.T).squeeze()
  chunk_ids = similarity_score.argsort().tolist()[-k:]
  return similarity_score[chunk_ids], [all_chunks[x] for x in chunk_ids]



#books = ['assign_1_bootloader', 'riscv_isa_privileged', 'xv6_book']
books = ['riscv_isa_privileged', 'xv6_book']
with open('question.txt', 'r') as f: question = f.read()
question_embd = encoder.encode(question)
scores, chunks = [], []
for book_name in books:
  sc, chks = get_context(question_embd, book_name)
  scores.append(sc)
  chunks.extend(chks)

scores = torch.cat(scores)
chunk_ids = scores.argsort().tolist()[-k:]
print(f'Selected chunk ids: {chunk_ids}')
print(f'Similarity Scores: {scores}')

# create prompt
passages = '\n-\n'.join((chunks[x] for x in chunk_ids))
prompt_template = '''Following are some passages:
{passages}
---
Answer the following question based on the passages: {question}'''
prompt = prompt_template.format(passages=passages, question=question)
with open('prompt.txt', 'w') as f: f.write(prompt)
