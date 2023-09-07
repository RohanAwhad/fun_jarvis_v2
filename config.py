import torch
from encoders import mpnet_encoder

torch.set_grad_enabled(False)

SEARCH_K = 3
PASSAGE_K = 15
BOOK_DIR = 'books'
EMBD_DIR = 'embeddings'
QUESTION_FILE = 'question.txt'
PROMPT_TEMPLATE = 'prompt.template'
PROMPT_FILE = 'prompt.txt'
MAX_LEN = 1200

# encoder config
ENCODER = mpnet_encoder
