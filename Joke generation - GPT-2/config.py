from transformers import GPT2Tokenizer


BATCH_SIZE = 512
EPOCHS = 4
LEARNING_RATE = 2e-5
MAX_LEN = 64
TRAIN_PATH = "./jokes.csv"  #ADD PATH TO YOUR DATASET HERE
MODEL_FOLDER = "Joke Generation" 
Tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
