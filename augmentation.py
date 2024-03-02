from tqdm.auto import tqdm
from model import TranslationModel
from transformers import AutoModelForSeq2SeqLM, MBart50TokenizerFast
from datasets import Dataset
from accelerate import Accelerator


# Model
accelerator = Accelerator()
model_checkpoint = 'facebook/mbart-large-50-many-to-many-mmt'
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)


# Tokenizer
tokenizer = MBart50TokenizerFast.from_pretrained(
    model_checkpoint,
    src_lang="vi_VN",
    tgt_lang="en_XX"
)


# Create model
inputs = self.tokenizer(
      text,
      padding='max_length',
      truncation=True,
      max_length=75,
      return_tensors='pt'
)
inputs = self.accelerator.prepare(inputs)


# Generate
outputs = self.accelerator.unwrap_model(self.model).generate(
    inputs.input_ids,
    attention_mask = inputs.attention_mask,
    max_new_tokens = 75,
    num_beams = 4
)
output_str = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

v2e_model = TranslationModel(model, tokenizer, accelerator=accelerator)


# Augmentation
path = r'data/phoml_test.vi'
vi_phml = []
with open(path, 'r') as file:
    for line in file:
        vi_phml.append(line)    

vi_phml = vi_phml

en_phomt_preds = []
process_bar = tqdm(range(len(vi_phml)))
for sent in vi_phml:
    # Tokenize
    inputs = self.tokenizer(
          text,
          padding='max_length',
          truncation=True,
          max_length=75,
          return_tensors='pt'
    )
    inputs = self.accelerator.prepare(inputs)

    # Generate
    outputs = self.accelerator.unwrap_model(self.model).generate(
        inputs.input_ids,
        attention_mask = inputs.attention_mask,
        max_new_tokens = 75,
        num_beams = 4
    )
    output = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)    
    en_phomt_preds.append(output[0])
    process_bar.update(1)


# Convert to dataset
aug_data = Dataset.from_dict(
    {
        'translation': [
            {'en': source, 'vi': target} for
            source, target in zip(en_phomt_preds, vi_phml)
        ]
    }
)

# Save data
path = 'data/augmentation_data'
aug_data.save_to_disk(path)