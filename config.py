import torch
from encoders import mpnet_encoder, msmarco_distilbert_cos, encoder_service

torch.set_grad_enabled(False)

SEARCH_K = 5
PASSAGE_K = 15
BOOK_DIR = "books"
EMBD_DIR = "embeddings"
QUESTION_FILE = "question.txt"
PROMPT_TEMPLATE = "prompt.template"
PROMPT_FILE = "prompt.txt"
MAX_LEN = 500

# encoder config
# ENCODER = msmarco_distilbert_cos
ENCODER = mpnet_encoder
# ENCODER = encoder_service
