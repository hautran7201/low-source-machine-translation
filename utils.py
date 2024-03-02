import torch
import numpy as np
import evaluate
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm.auto import tqdm
from transformers import DataCollatorForSeq2Seq


def evaluate_model(model, tokenizer, data, metric):
    process_bar = tqdm(range(len(data)))
    accelerator = Accelerator()
    
    collator = DataCollatorForSeq2Seq(tokenizer, model=model)    
    eval_loader = DataLoader(
        data, 
        collate_fn=collator, 
        batch_size=5
    )

    eval_loader, model = accelerator.prepare(
        eval_loader,
        model
    ) 

    model.eval()
    for batch in eval_loader:
        with torch.no_grad():
            generated_tokens  = accelerator.unwrap_model(model).generate(
                input_ids = batch['input_ids'],
                attention_mask = batch['attention_mask'],
                max_new_tokens=128,
                num_beams=4
            )

        labels = batch['labels']

        # Pad across processs
        generated_tokens = accelerator.pad_across_processes(
            generated_tokens,
            dim=1,
            pad_index=tokenizer.pad_token_id
        )

        labels =accelerator.pad_across_processes(
            generated_tokens,
            dim=-1,
            pad_index=-100
        )

        # Gather
        generated_tokens = accelerator.gather(generated_tokens)
        labels = accelerator.gather(labels)

        # decoded_preds, decoded_labels = postprocess(generated_tokens, labels)

        predictions = generated_tokens.cpu().numpy()
        decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_preds = [pred.strip() for pred in decoded_predictions]

        labels = labels.cpu().numpy()
        labels = np.where(labels != -100, labels, model.config.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_labels = [[label.strip()] for label in decoded_labels]

        metric.add_batch(predictions=decoded_preds, references=decoded_labels)
        process_bar.update(1)

    results = metric.compute()
    return results

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    """decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)"""

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess(preds, labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result    

def postprocess(preds, labels=None):
    predictions = preds.cpu().numpy()
    decode_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decode_preds = [pred.strip() for pred in decode_predictions]

    decode_labels = None
    if labels != None:
        labels = labels.cpu().numpy()
        labels = np.where(labels != -100, labels, model.config.pad_token_id)
        decode_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decode_labels = [[label.strip()] for label in decode_labels]

    return decode_preds, decode_labels    