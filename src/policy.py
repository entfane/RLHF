from trl import AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer
from typing import List, Tuple

VALUE_HEAD_NAME = "v_head"

class OfflinePolicy:

    def __init__(self, model_name: str):
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.__freeze_policy_without_value_head()

    def __freeze_policy_without_value_head(self):
        for name, param in self.model.named_parameters():
            if not VALUE_HEAD_NAME in name:
                param.requires_grad = False

    def generate(self, inputs: List[str], num_completions_per_prompt: int = 2):
        formatted_inputs = []
        for prompt in inputs:
            chat_formatted_prompt_completion = [{'role': 'user', 'content': prompt}]
            chat_formatted_prompt_completion = self.tokenizer.apply_chat_template(chat_formatted_prompt_completion, tokenize = False)
            formatted_inputs.append(chat_formatted_prompt_completion)
        input = self.tokenizer(formatted_inputs, return_tensors = "pt", padding = True, padding_side = 'left')
        output = self.model.generate(**input, max_new_tokens=100, do_sample=True, top_p=0.95, num_return_sequences = num_completions_per_prompt)
        return output

    def generate_logits_and_values(self, input: List[Tuple[str, str]]):
        formatted_inputs = []
        for prompt, completion in input:
            chat_formatted_prompt_completion = [{'role': 'user', 'content': prompt},
                                                {'role': 'assistant', 'content': completion}]
            formatted_inputs.append(self.tokenizer.apply_chat_template(chat_formatted_prompt_completion, tokenize = False))
        input = self.tokenizer(formatted_inputs, return_tensors = "pt", padding = True)
        lm_logits, _, value = self.model(**input)
        return lm_logits, value

