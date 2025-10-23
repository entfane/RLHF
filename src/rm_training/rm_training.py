from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from torch.utils.data import DataLoader


model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-uncased", num_labels = 1)
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

dataset = load_dataset("HumanLLMs/Human-Like-DPO-Dataset", split = "train")

def tokenize(example):
    prompt_good = example['prompt'] + example['chosen']
    prompt_bad = example['prompt'] + example['rejected']

    return {'good': tokenizer(prompt_good, return_tensors = "pt"), 'bad': tokenizer(prompt_bad, return_tensors = "pt")}

output = dataset.map(tokenize).remove_columns(['prompt', 'chosen', 'rejected'])
