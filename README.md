
# Finetune translation model with back translation augmentation (EN 2 VI)




## Installation

Install my-project with npm

```bash
  cd low-source-machine-translation
  pip install -r requirements.txt
```
    
## Run Locally

Back translation (augmentation)
```bash
    python augmentation.py
```

Train model
```bash
    # Train
    model_path = 'facebook/mbart-large-50-many-to-many-mmt'
    aug_data_path = "data/augmentation_data"
    python main.py --train_only 1 --model_path {model_path} --aug_data_path {aug_data_path}
```

Test model 
```bash
    peft_model_id = 'hautran7201/mBart50-tl-en2vi-lora'
    python main.py --eval_only 1 --peft_model_id {peft_model_id}
```

Inference Model
```bash
    peft_model_id = 'hautran7201/mBart50-tl-en2vi-lora'
    python main.py --infer_only 1 --infer_data "Hello my house" --peft_model_id {peft_model_id}
```

