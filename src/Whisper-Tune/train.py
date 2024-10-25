import comet_ml
import os
import argparse
import soundfile as sf
import torch
import evaluate
import warnings

from dotenv import load_dotenv
from typing import Any, Dict, List, Union
from dataclasses import dataclass
from datasets import load_dataset, DatasetDict
from transformers import (WhisperForConditionalGeneration,
                          WhisperFeatureExtractor, 
                          WhisperTokenizer, 
                          WhisperProcessor,
                          Seq2SeqTrainingArguments, 
                          Seq2SeqTrainer)
from transformers import logging as hf_logging

# Suppress all warnings
warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()

# Load API Keys from .env
load_dotenv()
comet_ml.login(
    api_key=os.getenv("COMET_API_KEY"),
    project_name="Whisper"
)


def get_whisper_model(whisper_model_name):
    whisper_models = {
        "tiny": "openai/whisper-tiny",
        "base": "openai/whisper-base",
        "small": "openai/whisper-small",
        "medium": "openai/whisper-medium",
        "large": "openai/whisper-large",
        "large-v2": "openai/whisper-large-v2",
        "large-v3": "openai/whisper-large-v3",
        "large-v3-turbo": "openai/whisper-large-v3-turbo",
    }
    return whisper_models.get(whisper_model_name.lower(), whisper_model_name)

# Argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Whisper model for ASR.")
    parser.add_argument('--train_json', type=str, required=True, help="Path to the train dataset JSON file")
    parser.add_argument('--test_json', type=str, required=True, help="Path to the test dataset JSON file")
    
    parser.add_argument('--whisper_model', '-model', type=str, required=True,
                        help="Choose from 'tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3', 'large-v3-turbo' or provide your own model path in format example: `openai/whisper-large`")
    
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for training and evaluation")
    parser.add_argument('--gradient_accumulation_steps', '-grad_steps', type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument('--learning_rate', '-lr', type=float, default=2e-5, help="Learning rate for training")
    parser.add_argument('--warmup_steps', type=int, default=500, help="Warmup steps for learning rate")
    parser.add_argument('--epochs', '-e', type=int, default=10, help="Number of epochs for training")
    
    parser.add_argument('--num_workers', '-w', type=int, default=2, help="Num of cpu workers")

    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()

    # Set the Whisper model
    whisper_model = get_whisper_model(args.whisper_model)

    # Load datasets
    train_dataset = load_dataset('json', data_files=args.train_json)
    test_dataset = load_dataset('json', data_files=args.test_json)

    dataset = DatasetDict({
        'train': train_dataset["train"],
        'test': test_dataset["train"]
    })


    # Load the Whisper model and processor
    model = WhisperForConditionalGeneration.from_pretrained(whisper_model)
    model.generation_config.language = "english"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None


    feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_model)
    tokenizer = WhisperTokenizer.from_pretrained(whisper_model, language="English", task="transcribe")
    processor = WhisperProcessor.from_pretrained(whisper_model, language="English", task="transcribe")


    def prepare_dataset(batch):
        # load and resample audio data from 48 to 16kHz
        audio, _ = sf.read(batch["path"])

        # compute log-Mel input features from input audio array
        batch["input_features"] = feature_extractor(audio, sampling_rate=16000).input_features[0]

        # encode target text to label ids
        batch["labels"] = tokenizer(batch["text"]).input_ids
        return batch

    dataset = dataset.map(prepare_dataset, 
                          remove_columns=dataset.column_names["train"], 
                          num_proc=args.num_workers)


    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        processor: Any
        decoder_start_token_id: int

        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            # Handle audio inputs by returning torch tensors
            input_features = [{"input_features": feature["input_features"]} for feature in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

            # Get the tokenized label sequences
            label_features = [{"input_ids": feature["labels"]} for feature in features]
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

            # Replace padding with -100 to ignore loss correctly
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

            # Cut bos token if present
            if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
                labels = labels[:, 1:]

            batch["labels"] = labels
            return batch


    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Replace -100 with the pad_token_id
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        # Decode predictions and labels
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        # Compute Word Error Rate (WER)
        wer = metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    metric = evaluate.load("wer")

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=f"{whisper_model}",                                   # Set the output directory for the model
        disable_tqdm=False,                                              # Enaisable the progress bar
        per_device_train_batch_size=args.batch_size,                     # Batch size for training
        gradient_accumulation_steps=args.gradient_accumulation_steps,    # Gradient accumulation steps
        learning_rate=args.learning_rate,                                # Learning rate for training
        warmup_steps=args.warmup_steps,                                  # Warmup steps for learning rate
        num_train_epochs=args.epochs,                                    # Number of epochs for training
        gradient_checkpointing=True,                                     # Enable gradient checkpointing
        fp16=True,                                                       # Enable mixed precision training
        eval_strategy="epoch",                                           # Evaluation strategy
        per_device_eval_batch_size=args.batch_size,                      # Batch size for evaluation
        predict_with_generate=True,                                      # Enable generate prediction
        generation_max_length=64,                                        # Maximum length of generated text
        save_strategy="epoch",                                           # Save strategy: epoch
        logging_strategy="epoch",                                        # Logging strategy: epoch
        report_to=["comet_ml", "tensorboard"],                           # Report to comet_ml and tensorboard
        load_best_model_at_end=True,                                     # Load best model at the end of training
        metric_for_best_model="wer",                                     # Metric: WER for best model 
        greater_is_better=False,                                         # Less WER is better
        push_to_hub=True,                                                # Push to the Hugging Face Hub
    )


    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    processor.save_pretrained(training_args.output_dir)
    trainer.train()
    trainer.evaluate()

    kwargs = {
        "dataset": "Personal - Mimic Recording",
        "language": "en",
        "model_name": whisper_model,
        "finetuned_from": whisper_model,
        "tasks": "automatic-speech-recognition",
    }

    trainer.push_to_hub(**kwargs)

if __name__ == "__main__":
    main()