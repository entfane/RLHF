from trl import AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer
from typing import List, Tuple

class Policy:

    def __init__(self, model_name: str):
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def freeze_params(self):
        for name, param in self.model.named_parameters():
            param.requires_grad = False

    def generate(self, inputs: List[str], num_completions_per_prompt: int = 1, max_tokens_to_generate: int = 100):
        formatted_inputs = []
        for prompt in inputs:
            chat_formatted_prompt_completion = [{'role': 'user', 'content': prompt}]
            chat_formatted_prompt_completion = self.tokenizer.apply_chat_template(chat_formatted_prompt_completion, tokenize = False)
            formatted_inputs.append(chat_formatted_prompt_completion)
        input = self.tokenizer(formatted_inputs, return_tensors = "pt", padding = True, padding_side = 'left')
        output = self.model.generate(**input, max_new_tokens=100, do_sample=True,
                                     top_p=0.95, num_return_sequences = num_completions_per_prompt)
        return self.tokenizer.batch_decode(output)

    def generate_logits_and_values(self, input: List[str]):
        input = self.tokenizer(input, return_tensors = "pt", padding = True)
        lm_logits, _, value = self.model(**input)
        return lm_logits, value

