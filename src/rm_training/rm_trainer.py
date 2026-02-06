from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import wandb


from rm_loss import RMLoss

class RMTrainer:

    def __init__(self, model_name):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels = 1).to("cuda")
        print(f"Model loaded to {self.model.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def freeze_non_head_layers(self):
        for param in self.model.parameters():
            param.requires_grad = False

        last_module = list(self.model.children())[-1]

        for param in last_module.parameters():
            param.requires_grad = True

    def preprocess(self, example, prompt_col, chosen_col, rejected_col):
        chosen_messages = [{"role": "user", "content": example[prompt_col]}, {"role": "assistant", "content": example[chosen_col][1]['content']}]
        rejected_messages = [{"role": "user", "content": example[prompt_col]}, {"role": "assistant", "content": example[rejected_col][1]['content']}]
        example['text_chosen'] = self.tokenizer.apply_chat_template(chosen_messages, tokenize = False)
        example['text_rejected'] = self.tokenizer.apply_chat_template(rejected_messages, tokenize = False)
        return example

    def train(self, dataset_name, split_name, prompt_col, chosen_col, rejected_col, epochs, mini_batch_size, max_len, lr, log_wandb, wandb_project = "rm-training", wandb_run_name = None):
        self.freeze_non_head_layers()
        dataset = load_dataset(dataset_name, split = split_name)
        dataset = dataset.map(self.preprocess, fn_kwargs={
            "prompt_col": prompt_col, 
            "chosen_col": chosen_col, 
            "rejected_col": rejected_col
        }).remove_columns([prompt_col, chosen_col, rejected_col])

        if log_wandb:
            wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                config={
                "learning_rate": lr,
                "dataset": dataset_name,
                "epochs": epochs,
                "max_len": max_len,
                "mini_batch_size": mini_batch_size
            })
        if log_wandb:
            wandb.watch(self.model, log="gradients", log_freq=100)
        dataloader = DataLoader(dataset, batch_size=mini_batch_size)
        criterion = RMLoss()
        trainable_params = [param for param in self.model.parameters() if param.requires_grad]
        optimizer = torch.optim.Adam(trainable_params, lr = lr)
        for epoch in range(epochs):
            for batch in tqdm(dataloader):
                good = self.tokenizer(batch['text_chosen'], return_tensors = "pt", padding = "max_length", max_length=max_len, padding_side = "left", truncation = True).to(self.model.device)
                bad = self.tokenizer(batch['text_rejected'], return_tensors = "pt", padding = "max_length", max_length=max_len, padding_side = "left", truncation = True).to(self.model.device)
                output_good = self.model(**good).logits
                output_bad = self.model(**bad).logits
                loss = criterion(output_good, output_bad)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    wandb.log({"loss": loss.item(), "accuracy": (output_good > output_bad).float().mean()})
            torch.save(self.model.state_dict(), f"./saves/model{epoch}.pt")

        self.model.save_pretrained("./RM")

        wandb.finish()