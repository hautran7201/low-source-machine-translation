import pandas as pd
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import MBart50TokenizerFast

class TranslationDataset:
    def __init__(self,
        source_lng,
        target_lng,
        tokenizer,
        aug_data = None
    ):
        # Load data
        self.data = load_dataset(
            'mt_eng_vietnamese',
            'iwslt2015-en-vi',
        )

        # Augmentation
        if aug_data != None:
            data_df = pd.DataFrame(self.data['train'])
            aug_data_df = pd.DataFrame(aug_data)
            concat_data_df = pd.concat([data_df, aug_data_df])
            concat_data = Dataset.from_pandas(concat_data_df).remove_columns('__index_level_0__')
            self.data['train'] = concat_data

        # Tokenize config
        self.truncation = True
        self.max_length = 75
        self.source_lng = source_lng
        self.target_lng = target_lng

        # Tokenizer
        self.tokenizer = tokenizer

        # Tokenized dataset
        """self.tokenized_dataset = self.data.map(
            self.preprocess_function,
            batched=True,
            remove_columns=self.data['train'].column_names
        )"""
    
    def tokenize_split_data(self, split):
        tokenized_dataset = self.data[split].map(
            self.preprocess_function,
            batched=True,
            remove_columns=self.data['train'].column_names
        )

        return tokenized_dataset

    def preprocess_function(self, examples):
        inputs = [ex[self.source_lng] for ex in examples["translation"]]
        targets = [ex[self.target_lng] for ex in examples["translation"]]
        model_inputs = self.tokenizer(
            inputs,
            text_target=targets,
            padding='max_length',
            max_length=self.max_length,
            truncation=self.truncation,
            return_tensors='pt'
        )

        return model_inputs