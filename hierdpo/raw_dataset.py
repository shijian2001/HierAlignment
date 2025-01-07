from torch.utils.data import Dataset

def make_hier_data():
    pass

def apply_template():
    pass


class HierRawDataset(Dataset):
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
        input_template=None,
        num_processors=8
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer

        # chat_template
        self.input_template = input_template
        conv_key = getattr(self.strategy.args, "conversation", None)
        soap_key = getattr(self.strategy.args, "soap_note", None)
        hw_key = getattr(self.strategy.args, "homework", None)
        apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)

        if apply_chat_template:
            apply_chat_template = self.tokenizer.apply_chat_template

        # Parallel processing datasets
        processed_dataset = dataset.map(
            self.process_data, remove_columns=dataset.column_names, num_proc=num_processors
        )

        # Store the processed data in class attributes
        self.prompt_soap_in = processed_dataset["prompt_soap_in"]
        self.chosens_soap_in = processed_dataset["chosen_soap_in"]
        self.rejects_soap_in = processed_dataset["reject_soap_in"]

        self.prompt_soap_de = processed_dataset["prompt_soap_de"]
        self.chosens_soap_de = processed_dataset["chosen_soap_de"]
        self.rejects_soap_de = processed_dataset["reject_soap_de"]

        self.prompt_hw = processed_dataset["prompt_hw"]
        self.chosens_hw = processed_dataset["chosen_hw"]
        self.rejects_hw = processed_dataset["reject_hw"]
        
        self.extras = processed_dataset["extra"]

    def process_data(self):
        # make hier data

        # apply chat templat

        pass

    def __len__(self):
        length = len(self.chosens_soap_in)
        return length

    def __getitem__(self, idx):
        # ToDo
        return self.chosens_soap_in[idx]