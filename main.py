import numpy as np
import evaluate
import os
import torch
from evaluate import evaluator
from datasets import load_from_disk 
from utils import compute_metrics, evaluate_model
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model 
from peft import PeftModel, PeftConfig
from opt import config_parser
from torch.optim import AdamW
from torch.utils.data import DataLoader
from model import TranslationModel
from data.translation_dataset import TranslationDataset
from accelerate import Accelerator
from transformers import (
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer,
    MBart50TokenizerFast,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq, 
    get_scheduler,
    AutoTokenizer
)

if __name__ == '__main__':
    # ====== Config ======
    args = config_parser()
    model_checkpoint = args.model_path

    # ====== Pre-trained Model ======
    accelerator = Accelerator()

    if args.train_only:
        # Metric
        metric = evaluate.load("sacrebleu")

        # Model and tokenizer
        model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
        tokenizer = MBart50TokenizerFast.from_pretrained(
            model_checkpoint,
            src_lang="en_XX",
            tgt_lang="vi_VN"
        )
        collator = DataCollatorForSeq2Seq(tokenizer, model=model)

        # Dataset
        source_lng = 'en'
        target_lng = 'vi'
        if args.aug_data_path:
            aug_data = load_from_disk(args.aug_data_path)
            dataset = TranslationDataset(source_lng, target_lng, tokenizer, aug_data=aug_data)
        else:
            dataset = TranslationDataset(source_lng, target_lng, tokenizer, aug_data=None)
        tokenized_train_dataset = dataset.tokenize_split_data('train')
        tokenized_train_dataset.set_format("torch")
        tokenized_test_dataset = dataset.tokenize_split_data('test')
        tokenized_test_dataset.set_format("torch")
        accelerator = Accelerator()
        train_data, test_data = accelerator.prepare(
            tokenized_train_dataset, tokenized_test_dataset
        )

        # Arguments
        arguments = Seq2SeqTrainingArguments (
            predict_with_generate = True ,
            evaluation_strategy = "steps",
            save_strategy ="steps",
            save_steps = args.save_steps,
            eval_steps = args.eval_steps,
            output_dir="./checkpoint/",
            per_device_train_batch_size = args.batch_size,
            per_device_eval_batch_size = args.batch_size,
            learning_rate = 5e-5,
            save_total_limit = 1,
            num_train_epochs = args.epochs,
            report_to="none",
            label_names=["labels"],
            # push_to_hub=True
        )

        # Using lora
        if args.lora:
            model = prepare_model_for_kbit_training(model)
            config = LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="SEQ_2_SEQ_LM"
            )
            model = get_peft_model(model, config)
            model.print_trainable_parameters()

        # Training
        trainer = Seq2SeqTrainer(
            model=model,
            args=arguments,
            data_collator=collator,
            train_dataset=train_data,
            eval_dataset=test_data,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )

        trainer, model = accelerator.prepare(
            trainer, model
        )

        trainer.train()

    if args.eval_only:
        # Metric
        metric = evaluate.load("sacrebleu")

        # Load model
        if args.peft_model_id:
            config = PeftConfig.from_pretrained(args.peft_model_id)
            base_model_path = config.base_model_name_or_path            
            model = AutoModelForSeq2SeqLM.from_pretrained(base_model_path)
            model = PeftModel.from_pretrained(model, args.peft_model_id)
            tokenizer = MBart50TokenizerFast.from_pretrained(
                config.base_model_name_or_path,
                src_lang="en_XX",
                tgt_lang="vi_VN"
            )
            model_path = args.peft_model_id
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
            tokenizer = MBart50TokenizerFast.from_pretrained(
                args.model_path,
                src_lang="en_XX",
                tgt_lang="vi_VN"
            )
            model_path = args.model_path

        # Get data
        source_lng = 'en'
        target_lng = 'vi'
        dataset = TranslationDataset(source_lng, target_lng, tokenizer)
        tokenized_test_dataset = dataset.tokenize_split_data('test')
        tokenized_test_dataset.set_format("torch")

        # Accelerator
        eval_results = evaluate_model(model, tokenizer, tokenized_test_dataset, metric)

        params = {"model": model_path}
        evaluate.save("./results/", **eval_results, **params)        

    if args.infer_only:
        # Load model
        if args.peft_model_id:
            config = PeftConfig.from_pretrained(args.peft_model_id)
            base_model_path = config.base_model_name_or_path            
            model = AutoModelForSeq2SeqLM.from_pretrained(base_model_path)
            model = PeftModel.from_pretrained(model, args.peft_model_id)
            tokenizer = MBart50TokenizerFast.from_pretrained(
                config.base_model_name_or_path,
                src_lang="en_XX",
                tgt_lang="vi_VN"
            )            
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
            tokenizer = MBart50TokenizerFast.from_pretrained(
                args.model_path,
                src_lang="en_XX",
                tgt_lang="vi_VN"
            )
        
        inputs = tokenizer(
            args.infer_data,
            padding='max_length',
            truncation=True,
            max_length=75,
            return_tensors='pt'
        )        

        model.eval()
        with torch.no_grad():
            outputs = accelerator.unwrap_model(model).generate(
                input_ids=inputs['input_ids']
            )

        output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        print(output_str)