#!/opt/homebrew/bin/python3

import config
import dsp
import helper
import prompts as dsp_prompts

# 3rd -party libs
import asyncio
import aiohttp
import os
import pandas as pd
import pyperclip
import re
import torch

from loguru import logger
from nltk.tokenize import sent_tokenize
from pypdf import PdfReader
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

def remove_reference_ids(s):
    # Pattern to match reference IDs like [1] [2] [3] etc. at the end of the string
    pattern = r'\s*(\[\d+\]\s*)+$'
    
    # Remove the matched pattern from the end of the string
    return re.sub(pattern, '', s)

async def encode_chunks(chunks): return await config.ENCODER.encode(chunks)

def get_context(question_embd, book_name):
  embd_fn = f"{config.EMBD_DIR}/{book_name}.{config.ENCODER.__name__}.bin"
  if os.path.exists(embd_fn): all_chunks, embeddings = helper.load_data(embd_fn)
  else:
    if book_name.endswith(".txt"):
      with open(f"{config.BOOK_DIR}/{book_name}", "r") as f: all_text = f.read()
    else:
      logger.info(f"embedding {book_name}")
      reader = PdfReader(f"{config.BOOK_DIR}/{book_name}.pdf")
      all_text = " ".join([
          x.strip()
          for page in reader.pages
          for x in page.extract_text().split("\n")
      ])

    all_sents = sent_tokenize(all_text)
    all_chunks = helper.get_chunks(all_sents, max_len=config.MAX_LEN)
    if config.ENCODER.__name__ == "encoder_service":
        embeddings = asyncio.run(encode_chunks(all_chunks, book_name))
    else:
        embeddings = config.ENCODER.encode(all_chunks)
    helper.save_data((all_chunks, embeddings), embd_fn)

  similarity_score = torch.tensor(cosine_similarity(question_embd, embeddings).squeeze())
  print(f"book: {book_name}, max score: {similarity_score.max()}")
  chunk_ids = similarity_score.argsort().tolist()[-config.PASSAGE_K :]
  return similarity_score[chunk_ids], [all_chunks[x] for x in chunk_ids]


async def rerank_chunks(question, chunks):
  async with aiohttp.ClientSession() as session:
    batch_size = 16
    reranked_scores = []
    tasks = []
    for i in range(0, len(chunks), batch_size):
      batch_chunks = chunks[i : i + batch_size]
      task = asyncio.create_task(
        session.post(
            "https://fun-reranker-84b02fb456da.herokuapp.com/rerank",
            json={"query": question, "passages": batch_chunks},
        )
      )
      tasks.append(task)
    responses = await asyncio.gather(*tasks)
    for res in responses:
      if res.status == 200: reranked_scores.extend((await res.json())["scores"])
      else:
        logger.error(f"Reranking failed with status code {res.status}")
        logger.error(await res.text())
        reranked_scores.extend([0] * len(batch_chunks))
    return reranked_scores


def rag(question, books, prev_context=None):
  # embed
  if config.ENCODER.encode.__name__ == "encoder_service": question_embd = asyncio.run(config.ENCODER.encode(question))
  else: question_embd = config.ENCODER.encode(question)
  logger.info('question embedding done')

  # retrieve
  scores, chunks, book_names = [], [], []
  for book_name in books:
    sc, chks = get_context(question_embd, book_name)
    scores.append(sc)
    chunks.extend(chks)
    book_names.extend([book_name] * len(chks))

  logger.info('retrieval done')

  # rerank
  rerank_scores = asyncio.run(rerank_chunks(question, chunks))
  rerank_scores = torch.tensor(rerank_scores)
  chunk_ids = rerank_scores.argsort().tolist()[::-1]
  logger.info('reranking done')
  # if prev_context is None: prev_context = []
  # return [chunks[x] for x in chunk_ids if chunks[x] not in prev_context][:config.SEARCH_K]
  chunk_ids = chunk_ids[:config.SEARCH_K]


  # create prompt
  # TODO (rohan): should I initialize the template in config?
  with open(config.PROMPT_TEMPLATE, "r") as f: prompt_template = f.read()
  passages = "\n-\n".join((f"[{i+1}] {chunks[x]}" for i, x in enumerate(chunk_ids)))
  prompt = prompt_template.format(passages=passages, question=question)

  # generate
  ans_obj = None
  for _ in range(5):
    llm_output = dsp.call_llm(prompt)
    logger.debug(llm_output)
    try:
      ans_obj = dsp.get_json(llm_output)
      break
    except Exception as e:
      logger.error(e)
      logger.error(llm_output)
      logger.error("Retrying...")
      continue

  if ans_obj is None: raise Exception("LLM failed to generate answer")
  return ans_obj["final_answer"]


def dsp_(question, books):
  context_list = []
  for _ in tqdm(range(20), leave=False):
    logger.debug(f'len(context_list): {len(context_list)}')
    logger.debug(f'context_list: {context_list}')
    if context_list:
      # context = [f"[{i+1}] <<{c}>>" for i, c in enumerate(context_list)]
      # context = "\n".join(context)
      context = []
      for i, (q, a) in enumerate(context_list):
        s = f'  - [{i+1}]:\n'
        s += f'    - subquestion: "{q}"\n'
        s += f'    - answer: "{remove_reference_ids(a)}"'
        context.append(s)

      context = "\n".join(context)

    else:
      context = "N/A"

    # ask llm is it enough info
    if context != "N/A":
      is_enough_info_prompt = dsp_prompts.is_enough_info_prompt.format(context=context, original_question=question)
      llm_output = dsp.call_llm(is_enough_info_prompt)
      output_dict = dsp.get_json(llm_output)
      logger.debug(output_dict)
      if output_dict['is_enough_info'] == True: break

    # ask llm for subquestion
    subquestion_prompt = dsp_prompts.subquestion_prompt.format(context=context, original_question=question)
    llm_output = dsp.call_llm(subquestion_prompt)
    logger.debug(llm_output)
    output_dict = dsp.get_json(llm_output)
    subquestion = output_dict['next_question']
    subq_ans = rag(subquestion, books)
    logger.debug(f'answer: {subq_ans}')

    context_list.append((subquestion, subq_ans))

    '''
    prompt = dsp_prompts.initial_prompt.format(context=context, query=question)
    output_obj = None
    for _ in range(5):
      generated_output = dsp.call_llm(prompt)
      logger.debug(generated_output)
      try:
        output_obj = dsp.get_json(generated_output)
        break
      except Exception as e:
        logger.error(e)
        logger.error(generated_output)
        logger.error("Retrying...")
        continue

    if output_dict['factoid_subquestion'] == 'N/A' and output_dict['answer'] != 'N/A': break
    # else: context_list.extend(rag(output_dict['factoid_subquestion'], books, context_list))
    else:
      subq_ans = rag(output_dict['factoid_subquestion'], books, context_list)
      context_list.append((output_dict['factoid_subquestion'], subq_ans))
    '''

  # print final answer
  final_answer_prompt = dsp_prompts.final_answer_prompt.format(context=context, original_question=question)
  llm_output = dsp.call_llm(final_answer_prompt)
  output_dict = dsp.get_json(llm_output)
  answer = output_dict['answer']
  print('Context: ', context)
  print('Original Question: ', question)
  print('Answer: ', answer)




if __name__ == "__main__":
  config.ENCODER.init()

  # books = [
  #     # "data_science_for_business",
  #     "the_starbucks_experience_book",
  # ]

  books = [
      "data_mgmt_for_multimedia_retrieval",
      "intro_to_info_retrieval",
      "search_engines_info_retrieval_in_practice",
      "mwdb_all_lectures_transcription.txt",
  ]

  # books = [
  #     "riscv_isa_privileged",
  #     "xv6_book",
  #     "the_little_os_book",
  #     "modern_os",
  #     "os_concepts",
  # ]  # TODO (rohan): this should be taken from yaml
  # books = ['swe_tb', 'sommerville_se']
  with open(config.QUESTION_FILE, "r") as f:
      question = (
          f.read()
      )  # TODO (rohan): this should be run in loop, to avoid reinitialization of encoder model

  dsp_(question, books)
  # ans = rag(question, books)
  # print(ans)
