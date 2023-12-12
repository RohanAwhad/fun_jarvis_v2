import g4f
from typing import Dict
import re
import json

import prompts

def call_llm(prompt: str) -> str:
  response = g4f.ChatCompletion.create(
    model=g4f.models.gpt_4,
    messages=[{"role": "user", "content": prompt}],
    stream=False,
  )
  return response

def get_json(text: str) -> Dict:
  json_string_extracted = re.search(r"\{.*\}", text, re.DOTALL).group()
  return json.loads(json_string_extracted)

def main(question: str) -> str:
  # first iteration
  first_call = prompts.initial_prompt.format(query=question)
  subquestion = get_json(call_llm(first_call))['factoid_subquestion']
  # search for answer for this subquestion
  #   embed
  #   retrieve
  #   rerank
  #   generate

  # continue with second iteration
  


if __name__ == "__main__":
  query = "Analyze the use of R-trees in mapping applications like Google Maps. Discuss the advantages of R-trees in spatial data handling and the challenges they might face in such applications"