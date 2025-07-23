from trl import AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer

VALUE_HEAD_NAME = "v_head"

class OfflinePolicy:

    def __init__(self, model_name):
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.__freeze_policy_without_value_head()

    def __freeze_policy_without_value_head(self):
        for name, param in self.model.named_parameters():
            if not VALUE_HEAD_NAME in name:
                param.requires_grad = False

    def generate(self, input):
        formatted_inputs = []
        for prompt, completion in input:
            chat_formatted_prompt_completion = [{'role': 'user', 'content': prompt},
                                                {'role': 'assistant', 'content': completion}]
            formatted_inputs.append(self.tokenizer.apply_chat_template(chat_formatted_prompt_completion), tokenize = False)
        input = self.tokenizer(formatted_inputs, return_tensors = "pt")
        lm_logits, _, value = self.model(**input)
        return lm_logits, value

