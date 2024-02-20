import torch
import transformers
from dataclasses import dataclass, field
from typing import Union
from transformers import AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, HfArgumentParser, \
    AutoModelForSeq2SeqLM
import os
from datasets import load_dataset, load_from_disk, concatenate_datasets
from datasets import load_metric

metric = load_metric("rouge")



@dataclass
class ModelArguments:
    pretrained_model_name_or_path: str = field(
        default="KETI-AIR/ke-t5-small-ko", metadata={"help": "학습 및 평가에 사용할 모델 경로"}
    )
    use_auth_token: str = field(
        default=None, metadata={"help": "비공개 모델 사용에 필요한 인증 토큰"}
    )


@dataclass
class DataArguments:
    data_name_or_path: str = field(
        default="ohgnues/korean-qa-paraphrase"
    )
    use_auth_token_data: str = field(
        default=None, metadata={"help": "비공개 데이터 사용에 필요한 인증 토큰"}
    )
    train_split: str = field(
        default="train", metadata={"help": "학습 데이터 이름"}
    )
    eval_split: str = field(
        default="validation", metadata={"help": "평가 데이터 이름"}
    )
    shuffle: bool = field(
        default=True, metadata={"help": "데이터 셔플 여부"}
    )
    question1_column_name: str = field(
        default="question-1", metadata={"help": "질문 데이터 Column 이름"}
    )
    question2_column_name: str = field(
        default="question-2", metadata={"help": "타겟 데이터 Column 이름"}
    )
    text_max_length: int = field(
        default=32, metadata={"help": "입력의 최대 토큰 길이"}
    )
    answer_max_length: int = field(
        default=32, metadata={"help": "라벨의 최대 토큰 길이"}
    )


@dataclass
class TrainArguments(Seq2SeqTrainingArguments):
    output_dir: str = field(
        default="runs/", metadata={"help": "학습 결과 저장 경로"}
    )
    do_train: bool = field(
        default=True, metadata={"help": "학습 여부"}
    )
    do_eval: bool = field(
        default=True, metadata={"help": "평가 여부"}
    )
    per_device_train_batch_size: int = field(
        default=128, metadata={"help": "학습 배치 사이즈"}
    )
    per_device_eval_batch_size: int = field(
        default=128, metadata={"help": "평가 배치 사이즈"}
    )
    num_train_epochs: float = field(
        default=5.0, metadata={"help": "학습 Epoch 수"}
    )
    save_strategy: Union[transformers.trainer_utils.IntervalStrategy, str] = field(
        default='epoch'
    )
    logging_strategy: Union[transformers.trainer_utils.IntervalStrategy, str] = field(
        default='epoch'
    )
    evaluation_strategy: Union[transformers.trainer_utils.IntervalStrategy, str] = field(
        default='epoch'
    )


if __name__ == "__main__":

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainArguments))
    model_args, data_args, train_args = parser.parse_args_into_dataclasses()

    model = AutoModelForSeq2SeqLM.from_pretrained(**vars(model_args))
    tokenizer = AutoTokenizer.from_pretrained(**vars(model_args), padding_side='left')

    if os.path.isdir(data_args.data_name_or_path):
        dataset = load_from_disk(data_args.data_name_or_path)
    else:
        dataset = load_dataset(data_args.data_name_or_path, use_auth_token=data_args.use_auth_token_data)

    if data_args.shuffle:
        dataset = dataset.shuffle()


    def example_function(examples):

        q1 = examples[data_args.question1_column_name]
        q2 = examples[data_args.question2_column_name]

        tokenized_passsages = tokenizer(
            q1,
            truncation=True,
            padding="max_length",
            max_length=data_args.text_max_length
        )

        tokenized_answer = tokenizer(
            q2,
            truncation=True,
            padding="max_length",
            max_length=data_args.answer_max_length
        )

        tokenized_passsages["labels"] = tokenized_answer["input_ids"]

        return tokenized_passsages



    train_dataset = dataset[data_args.train_split]
    eval_dataset = dataset[data_args.eval_split] if data_args.eval_split in dataset else None

    train_dataset = train_dataset.map(example_function, remove_columns=train_dataset.column_names, num_proc=20)
    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(example_function, remove_columns=eval_dataset.column_names, num_proc=20)


    def preprocess_logits_for_metrics(logits, labels):
        """
        Original Seq2SeqTrainer may have a memory leak.
        This is a workaround to avoid storing too many tensors that are not needed.
        """
        pred_ids = torch.argmax(logits[0], dim=-1)
        return pred_ids, labels


    def rouge(eval_preds):
        preds, labels = eval_preds

        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels]

        _rouge = metric.compute(predictions=decoded_preds, references=decoded_labels)

        outputs = {}
        for key in ['rouge1', 'rouge2', 'rougeLsum']:
            outputs[key] = _rouge[key].mid.fmeasure * 100

        return {"rouge1": outputs["rouge1"], "rouge2": outputs["rouge2"], "rougeLsum": outputs["rougeLsum"]}


    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=rouge,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )

    trainer.train()
