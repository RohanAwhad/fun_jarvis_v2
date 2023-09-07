import torch

torch.set_grad_enabled(False)

K = 15
BOOK_DIR = 'books'
EMBD_DIR = 'embeddings'
QUESTION_FILE = 'question.txt'
PROMPT_TEMPLATE = 'prompt.template'
PROMPT_FILE = 'prompt.txt'
MAX_LEN = 1200

# encoder config
MODEL_PATH = '/Users/rohan/3_Resources/ai_models/all-mpnet-base-v2'
