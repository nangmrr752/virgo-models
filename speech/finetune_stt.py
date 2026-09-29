"""Fine-tune Virgo 1.0 speech to text (Whisper) so it understands Khmer better.

    python speech/finetune_stt.py                          # FLEURS Khmer, whisper-small
    python speech/finetune_stt.py --base openai/whisper-base --steps 2000

Default data: Google FLEURS "km_kh" (read Khmer speech with transcripts). Add your own recordings
by pointing --extra at a folder with audio files and a metadata.csv of `file_name,sentence`.
Needs a GPU (a free Colab T4 is fine); saves to speech/out/virgo-1.0-stt.
"""
import argparse
from dataclasses import dataclass

import torch
from datasets import Audio, concatenate_datasets, load_dataset
from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperForConditionalGeneration,
                          WhisperProcessor)


@dataclass
class Collator:
    processor: WhisperProcessor

    def __call__(self, features):
        inputs = self.processor.feature_extractor.pad([{"input_features": f["input_features"]} for f in features], return_tensors="pt")
        labels = self.processor.tokenizer.pad([{"input_ids": f["labels"]} for f in features], return_tensors="pt")
        ids = labels["input_ids"].masked_fill(labels.attention_mask.ne(1), -100)
        if (ids[:, 0] == self.processor.tokenizer.bos_token_id).all():
            ids = ids[:, 1:]
        inputs["labels"] = ids
        return inputs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="openai/whisper-small")
    p.add_argument("--out", default="speech/out/virgo-1.0-stt")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--extra", help="folder with your own audio + metadata.csv (file_name,sentence)")
    args = p.parse_args()

    processor = WhisperProcessor.from_pretrained(args.base, language="khmer", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.base)
    model.generation_config.language = "khmer"
    model.generation_config.task = "transcribe"

    data = load_dataset("google/fleurs", "km_kh", split="train", trust_remote_code=True).rename_column("transcription", "sentence")
    data = data.select_columns(["audio", "sentence"])
    if args.extra:
        mine = load_dataset("audiofolder", data_dir=args.extra, split="train").select_columns(["audio", "sentence"])
        data = concatenate_datasets([data, mine])
    data = data.cast_column("audio", Audio(sampling_rate=16000))

    def prepare(row):
        audio = row["audio"]
        row["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=16000).input_features[0]
        row["labels"] = processor.tokenizer(row["sentence"]).input_ids
        return row

    data = data.map(prepare, remove_columns=data.column_names)
    Seq2SeqTrainer(
        model=model,
        args=Seq2SeqTrainingArguments(
            output_dir=args.out, per_device_train_batch_size=16, learning_rate=1e-5, warmup_steps=300,
            max_steps=args.steps, fp16=torch.cuda.is_available(), gradient_checkpointing=True,
            logging_steps=25, save_steps=1000, report_to=[],
        ),
        train_dataset=data,
        data_collator=Collator(processor),
    ).train()
    model.save_pretrained(args.out)
    processor.save_pretrained(args.out)
    print(f"Saved Virgo 1.0 speech to text to {args.out}")


if __name__ == "__main__":
    main()
