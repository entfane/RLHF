from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm


from rm_loss import RMLoss



model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-uncased", num_labels = 1).to("cuda")
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

print(f"Model loaded to {model.device}")

def freeze_non_head_layers(model):
    for param in model.parameters():
        param.requires_grad = False
    model.classifier.weight.requires_grad = True
    model.classifier.bias.requires_grad = True

freeze_non_head_layers()

dataset = load_dataset("HumanLLMs/Human-Like-DPO-Dataset", split = "train")

def tokenize(example):
    prompt_good = example['prompt'] + example['chosen']
    prompt_bad = example['prompt'] + example['rejected']

    return {'good': prompt_good, 'bad': prompt_bad}

dataset = dataset.map(tokenize).remove_columns(['prompt', 'chosen', 'rejected'])


dataloader = DataLoader(dataset, batch_size=4)

criterion = RMLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4, betas = [0.8, 0.99])
for epoch in range(1):
    for batch in tqdm(dataloader):
        input_good = tokenizer(batch['good'], return_tensors = "pt", padding = "max_length", max_length=512, padding_side = "left", truncate = True).to(model.device)
        input_bad = tokenizer(batch['bad'], return_tensors = "pt", padding = "max_length", max_length=512, padding_side = "left", truncate = True).to(model.device)
        output_good = model(**input_good).logits
        output_bad = model(**input_bad).logits
        loss = criterion(output_good, output_bad)
        print(f"Epoch {epoch} loss: {loss.item()}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    torch.save(model.state_dict(), f"./saves/model{epoch}.pt")





