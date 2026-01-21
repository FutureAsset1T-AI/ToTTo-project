import os
import json
import torch
import wandb
import numpy as np
import evaluate
from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed
)
from config import config
from T5_architecture import T5Dataset, get_model_and_tokenizer

# 데이터 로드 함수
def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {file_path}")
        
    print(f"Loading data: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        if file_path.endswith('.jsonl'):
            return [json.loads(line) for line in f]
        else:
            return json.load(f)

# 평가 지표 계산 함수 (BLEU Score)
metric = evaluate.load("sacrebleu")

def compute_metrics(eval_preds, tokenizer):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]
    
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": result["score"]}

def main():
    # 1. 시드 및 환경 설정
    set_seed(config.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 2. WandB 초기화
    wandb.init(
        project=config.WANDB_PROJECT,
        name=config.WANDB_NAME,
        config=config.__dict__
    )

    # 3. 모델 및 토크나이저 로드 (A100 최적화 포함)
    model, tokenizer = get_model_and_tokenizer(config.MODEL_NAME)
    model.to(device)

    # 4. 데이터셋 구축
    raw_train = load_data(config.TRAIN_FILE_PATH)
    raw_val = load_data(config.VAL_FILE_PATH)

    train_dataset = T5Dataset(raw_train, tokenizer, config.MAX_INPUT_LEN, config.MAX_TARGET_LEN)
    val_dataset = T5Dataset(raw_val, tokenizer, config.MAX_INPUT_LEN, config.MAX_TARGET_LEN)

    # 5. 데이터 콜레이터
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8
    )

    # 6. 학습 인자 설정 (A100 최적화 및 학습률 전략 적용)
    training_args = Seq2SeqTrainingArguments(
        output_dir=config.OUTPUT_DIR,
        learning_rate=config.LEARNING_RATE,
        
        # [추가된 학습률 최적화 전략]
        # 새로운 스페셜 토큰 임베딩의 안정적인 학습을 위해 Warmup을 설정합니다.
        lr_scheduler_type="linear",      # 학습률을 선형적으로 감소시킵니다.
        warmup_ratio=0.1,                # 전체 스텝의 10% 동안 학습률을 서서히 올립니다.
        weight_decay=0.01,               # 가중치 감쇠를 통해 과적합을 방지합니다.
        
        # [A100 핵심 최적화 설정]
        bf16=True,                       # bfloat16 활성화
        fp16=False,                      
        tf32=True,                       # TensorFloat-32 가속
        
        # 배치 및 성능 설정
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        dataloader_num_workers=4, 
        group_by_length=True,     
        
        num_train_epochs=config.NUM_EPOCHS,
        
        # 평가 및 저장 전략
        evaluation_strategy="steps",
        eval_steps=config.EVAL_STEPS,
        save_strategy="steps",
        save_steps=config.SAVE_STEPS,
        
        # Best Model 설정
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=config.SAVE_TOTAL_LIMIT,
        
        logging_steps=config.LOGGING_STEPS,
        report_to="wandb",
        predict_with_generate=True,
        push_to_hub=False,
    )

    # 7. Trainer 초기화
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_metrics(p, tokenizer),
    )

    # 8. 체크포인트 탐색 및 학습 재개 로직
    last_checkpoint = None
    if os.path.isdir(config.OUTPUT_DIR):
        from transformers.trainer_utils import get_last_checkpoint
        last_checkpoint = get_last_checkpoint(config.OUTPUT_DIR)
        if last_checkpoint:
            print(f"Found checkpoint. Resuming from: {last_checkpoint}")

    # 9. 학습 실행
    trainer.train(resume_from_checkpoint=last_checkpoint)

    # 10. Best Model 저장
    print(f"Finalizing... Saving the best model to {config.BEST_MODEL_DIR}")
    trainer.save_model(config.BEST_MODEL_DIR)
    tokenizer.save_pretrained(config.BEST_MODEL_DIR)
    
    wandb.finish()

if __name__ == "__main__":
    main()