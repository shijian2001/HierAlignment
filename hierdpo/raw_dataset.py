from torch.utils.data import Dataset
from typing import Dict
import json

def make_hier_pref(data, conv_key:str, soap_key:str, hw_key:str, instruction_templates:Dict[str]):

    conversation, soap_note, homework = data[conv_key], data[soap_key], data[hw_key]
    inst_conv, inst_soap, inst_hw = instruction_templates[conv_key], instruction_templates[soap_key], instruction_templates[hw_key]
    
    pref = {
        "soap_de": [
            {
                "content": inst_conv.format(conversation),
                "role": "user"
            },
            {
                "content": soap_note,
                "role": "assistant"
            }
        ],
        "soap_in": [
            {
                "content": inst_hw.format(homework),
                "role": "user"
            },
            {
                "content": soap_note,
                "role": "assistant"
            }
        ],
        "hw": [
            {
                "content": inst_soap.format(soap_note),
                "role": "user"
            },
            {
                "content": homework,
                "role": "assistant"
            }
        ]
    }
    return pref

def apply_template(pref, tokenizer, apply_chat_template=None):
    data = {}

    for key in ["soap_de", "soap_in", "hw"]:
        if apply_chat_template:
            chosen = apply_chat_template(pref[key], tokenize=False)
            prompt = apply_chat_template(pref[key][:-1], tokenize=False, add_generation_prompt=True)
            if chosen.startwith(tokenizer.bos_token):
                chosen  = chosen[len(tokenizer.bos_token):]
            if not chosen.endswith(tokenizer.eos_token):
                chosen += " " + tokenizer.eos_token
        else:
            prompt = ""
            chosen = pref[key]

        data[f"prompt_{key}"] = prompt
        data[f"chosen_{key}"] = chosen

    return data

class HierRewardDataset(Dataset):
    """
    Raw Dataset for the hierarchical alignment
    Column: conversation, soap_note, homework

    Args:
        dataset: dataset for clinical diagnosis
        tokenizer: tokenizer for the tuned LLM
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        strategy,
        num_processors=8,
        instruction_template_file:str=None
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer

        # chat_template
        if instruction_template_file is not None:
            self.instruction_templates = self._load_templates(instruction_template_file)
        else:
            self.instruction_templates = None

        self.conv_key = getattr(self.strategy.args, "conversation", None)
        self.soap_key = getattr(self.strategy.args, "soap_note", None)
        self.hw_key = getattr(self.strategy.args, "homework", None)
        self.apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)

        if self.apply_chat_template:
            self.apply_chat_template = self.tokenizer.apply_chat_template
            tokenizer_chat_template = getattr(self.strategy.args, "tokenizer_chat_template", None)
            if tokenizer_chat_template:
                self.tokenizer.chat_template = tokenizer_chat_template

        # Parallel processing datasets
        processed_dataset = dataset.map(
            self.process_data, remove_columns=dataset.column_names, num_proc=num_processors
        )

        self.prompts_soap_de = processed_dataset["prompt_soap_de"]
        self.chosens_soap_de = processed_dataset["chosen_soap_de"]

        self.prompts_soap_in = processed_dataset["prompt_soap_in"]
        self.chosens_soap_in = processed_dataset["chosen_soap_in"]

        self.prompts_hw = processed_dataset["prompt_hw"]        
        self.chosens_hw = processed_dataset["chosen_hw"]        
    
    def _load_templates(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"The specified file {file_path} does not exist")
        except json.JSONDecodeError:
            raise ValueError(f"The file {file_path} is not a valid JSON format")

    def process_data(self, data):
        # make hier data
        hier_pref = make_hier_pref(
            data,
            self.conv_key,
            self.soap_key,
            self.hw_key,
            self.instruction_templates
        )

        # apply chat templat
        processed_data = apply_template(hier_pref, self.tokenizer, self.apply_chat_template)

        return processed_data

    def __len__(self):
        length = len(self.chosens_soap_de)
        return length

    def __getitem__(self, idx):
        prompt_soap_de, chosen_soap_de, prompt_soap_in, chosen_soap_in, prompt_hw, chosen_hw = (
            self.prompts_soap_de[idx], self.chosens_soap_de[idx], self.prompts_soap_in[idx], 
            self.chosens_soap_in[idx], self.prompts_hw[idx], self.chosens_hw[idx]
        )

        return (
            prompt_soap_de, 
            chosen_soap_de, 
            prompt_soap_in, 
            chosen_soap_in, 
            prompt_hw, 
            chosen_hw
        )