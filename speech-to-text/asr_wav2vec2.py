from datasets import load_dataset, Audio, load_from_disk

# minds = load_dataset("PolyAI/minds14", name="en-US", split="train[:100]") # Dataset minds14 downloaded and prepared to /home/tslab/phusaeng/.cache/huggingface/datasets/PolyAI___minds14/en-US/1.0.0/65c7e0f3be79e18a6ffaf879a083daf706312d421ac90d25718459cbf3c42696. Subsequent calls will reuse this data.
minds = load_from_disk("minds14")
# minds.save_to_disk("minds14")

# split train and test set
minds = minds.train_test_split(test_size=0.2)

minds = minds.remove_columns(['intent_class', 'lang_id', 'english_transcription'])

# ignore special characters for speech
import re
chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�\…\–\—\(\)\[\]\{\}\<\>\=\+\@\#\$\&\*\^\~\_\`\’\/\’\‘\|]'

def remove_special_characters(batch, column_names='transcription'):
    batch[column_names] = re.sub(chars_to_ignore_regex, '', batch[column_names]).lower() + " "
    return batch


# preprocess the data
# what this processor can do?
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base")

# The MInDS-14 dataset has a sampling rate of 8000kHz (you can find this information in its dataset card), 
# which means you’ll need to resample the dataset to 16000kHz to use the pretrained Wav2Vec2 model:
from datasets import Audio

minds = minds.cast_column("audio", Audio(sampling_rate=16_000))

# The Wav2Vec2 tokenizer is only trained on uppercase characters 
# so you’ll need to make sure the text matches the tokenizer’s vocabulary:

def uppercase(example):
    return {"transcription": example["transcription"].upper()}

minds = minds.map(uppercase)

def prepare_dataset(batch):
    # call the audio column to get audio data

    audio = batch["audio"]
    batch = processor(audio["array"], sampling_rate=audio["sampling_rate"], text=batch["transcription"])
    batch["input_length"] = len(batch["input_values"][0]) # dict_keys(['input_values', 'labels', 'input_length'])
    return batch

# apply the preprocessing function to the entire dataset by using Datasets map function
encoded_minds = minds.map(prepare_dataset, remove_columns=minds.column_names['train'], num_proc=4)

import torch

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

# add dataclass decorator to instantiate the __init__, __repr__, and __eq__
@dataclass
class DataCollatorCTCWithPadding:
    processor: AutoProcessor
    padding: Union[bool, str] = "longest"

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"][0]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")
        labels_batch = self.processor.pad(labels=label_features, padding=self.padding, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels
        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor, padding="longest")

# call evaluation method
import evaluate

wer = evaluate.load('wer')

# create a method that compute the WER to evaluate between pred and gt
import numpy as np

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

from transformers import AutoModelForCTC, TrainingArguments, Trainer

model = AutoModelForCTC.from_pretrained(
    "facebook/wav2vec2-base",
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
)

training_args = TrainingArguments(
    output_dir="my_awesome_asr_mind_model",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=2000,
    gradient_checkpointing=True,
    fp16=True,
    group_by_length=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_minds["train"],
    eval_dataset=encoded_minds["test"],
    tokenizer=processor,
    data_collator=data_collator,
)

trainer.train()