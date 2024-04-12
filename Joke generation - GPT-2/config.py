from transformers import GPT2Tokenizer


<<<<<<< HEAD
BATCH_SIZE = 16
=======
BATCH_SIZE = 512
>>>>>>> 2f3bf53eff7a2e77eb3b2e1c1d7d6585406dd2e4
EPOCHS = 4
LEARNING_RATE = 2e-5
MAX_LEN = 64
TRAIN_PATH = "./jokes.csv"  #ADD PATH TO YOUR DATASET HERE
MODEL_FOLDER = "Joke Generation" 
Tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
