import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm

class TranslationModel:
    def __init__(self, model, tokenizer, optimizer=None, lr_scheduler=None, accelerator=None, metric=None):
        self.accelerator = accelerator
        self.model = self.accelerator.prepare(model)
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.metric = metric

    def train(self, train_data, val_data=None, test_data=None, epochs=1, out_dir=''):
        num_training_steps = len(train_data) * epochs
        process_bar = tqdm(range(num_training_steps))

        self.optimizer, train_data, test_data = self.accelerator.prepare(
            self.optimizer,
            train_data,
            test_data,
        )

        self.model.train()
        for epoch in range(epochs):
            for step, batch in enumerate(train_data): # step, batch in enumerate(train_data):
                outputs = self.model(**batch)
                loss = outputs.loss
                self.accelerator.backward(loss)

                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                process_bar.update(1)

                if val_data != None:
                    if step % 1000 == 0:
                        self.model.eval()
                        with torch.no_grad():
                            val_batch = next(iter(val_data))
                            val_loss = self.model(**batch).loss
                            process_bar.set_description(f"Train loss {loss}, Val loss {val_loss}")

        if test_data != None:
            results = self.evaluate(test_data) # self.metric.compute()
            print(f"epoch {epoch}, BLEU score: {results['score']:.2f}")

        if out_dir != None:
            self.accelerator.wait_for_everyone()
            unwarped_model = self.accelerator.unwrap_model(self.model)
            unwarped_model.save_pretrained(out_dir, save_function=self.accelerator.save)
            if self.accelerator.is_main_process:
                self.tokenizer.save_pretrained(out_dir)

    def evaluate(self, data):
        process_bar = tqdm(range(len(data)))

        self.model.eval()
        for batch in data:
            with torch.no_grad():
                generated_tokens  = self.accelerator.unwrap_model(self.model).generate(
                    batch['input_ids'],
                    attention_mask = batch['attention_mask'],
                    max_new_tokens=128,
                    num_beams=4
                )

            labels = batch['labels']

            # Pad across processs
            generated_tokens = self.accelerator.pad_across_processes(
                generated_tokens,
                dim=1,
                pad_index=self.tokenizer.pad_token_id
            )

            labels =self.accelerator.pad_across_processes(
                generated_tokens,
                dim=-1,
                pad_index=-100
            )

            # Gather
            generated_tokens = self.accelerator.gather(generated_tokens)
            labels = self.accelerator.gather(labels)

            decoded_preds, decoded_labels = self.postprocess(generated_tokens, labels)
            self.metric.add_batch(predictions=decoded_preds, references=decoded_labels)

            process_bar.update(1)

        self.model.train()
        results = self.metric.compute()
        return results

    def inference(self, text):
        inputs = self.tokenizer(
              text,
              padding='max_length',
              truncation=True,
              max_length=75,
              return_tensors='pt'
        )

        inputs = self.accelerator.prepare(inputs)
        outputs = self.accelerator.unwrap_model(self.model).generate(
            inputs.input_ids,
            attention_mask = inputs.attention_mask,
            max_new_tokens = 75,
            num_beams = 4
        )

        output_str = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return output_str

    def postprocess(self, preds, labels=None):
        predictions = preds.cpu().numpy()
        decode_predictions = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decode_preds = [pred.strip() for pred in decode_predictions]

        decode_labels = None
        if labels != None:
            labels = labels.cpu().numpy()
            labels = np.where(labels != -100, labels, self.model.config.pad_token_id)
            decode_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
            decode_labels = [[label.strip()] for label in decode_labels]

        return decode_preds, decode_labels