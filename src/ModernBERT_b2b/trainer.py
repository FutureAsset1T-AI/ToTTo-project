import os
import json
import torch
from datasets import load_dataset
import transformers
from transformers import (
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    EncoderDecoderModel,
    AutoTokenizer
)
from transformers.trainer_callback import PrinterCallback
from transformers.trainer_utils import get_last_checkpoint
from config import Config

# [중요] 이 함수는 (model, tokenizer_enc, tokenizer_dec) 3개를 반환하도록 수정되어 있어야 합니다.
from ModernBERT_B2B_architecture import build_modernbert_b2b_model

def preprocess_function(examples, tokenizer_enc, tokenizer_dec, config):
    """
    이중 토크나이저를 적용한 전처리 함수
    - 입력(Input): ModernBERT Tokenizer 사용
    - 출력(Label): RoBERTa(또는 BERT) Tokenizer 사용
    """
    inputs = examples["input"]
    targets = examples["target"]

    # 1. 인코더 입력 처리
    model_inputs = tokenizer_enc(
        inputs,
        max_length=config.MAX_INPUT_LENGTH,
        truncation=True,
        padding=False # DataCollator가 배치 단위로 패딩하므로 여기선 False
    )

    # 2. 디코더 정답(Label) 처리
    labels = tokenizer_dec(
        text_target=targets,
        max_length=config.MAX_TARGET_LENGTH,
        truncation=True,
        padding=False
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def main():
    transformers.logging.set_verbosity_error()

    # ==============================================================================
    # [WandB 설정]
    # 프로젝트 이름과 로그 설정을 환경변수로 지정합니다.
    # ==============================================================================
    os.environ["WANDB_PROJECT"] = "ModernBERT-Encoder-Decoder" # 프로젝트 이름 (원하는 대로 변경)
    os.environ["WANDB_WATCH"] = "false" # 모델 그라디언트 로깅 끔 (속도 향상)

    # 0. GPU 및 가속 설정 확인
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = torch.cuda.is_bf16_supported()
    use_fp16 = not use_bf16 and torch.cuda.is_available()

    print(f"==================================================")
    print(f"Device Status: {device}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"==================================================")

    # 1. 설정 로드
    cfg = Config()
    
    # 체크포인트 경로 설정
    base_output_dir = cfg.OUTPUT_DIR
    checkpoints_dir = os.path.join(base_output_dir, "checkpoints")
    
    last_checkpoint = None
    
    # (1) checkpoints 폴더 탐색
    if os.path.isdir(checkpoints_dir):
        last_checkpoint = get_last_checkpoint(checkpoints_dir)
        if last_checkpoint:
            print(f"✅ Found checkpoint in sub-folder: {last_checkpoint}")

    # (2) 기본 폴더 탐색
    if last_checkpoint is None and os.path.isdir(base_output_dir):
        last_checkpoint = get_last_checkpoint(base_output_dir)
        if last_checkpoint:
            print(f"✅ Found checkpoint in base folder: {last_checkpoint}")

    best_model_path = os.path.join(base_output_dir, "best_model")
    
    # 변수 초기화
    model = None
    tokenizer_enc = None
    tokenizer_dec = None

    # ==============================================================================
    # 모델 빌드 및 로드 (Dual Tokenizer 적용)
    # ==============================================================================

    # CASE 1: 체크포인트 발견 (Resume)
    if last_checkpoint is not None:
        print(">> Resuming training from optimizer/scheduler state.")
        # [수정] 반환값 3개 언패킹
        model, tokenizer_enc, tokenizer_dec = build_modernbert_b2b_model(cfg)
        
    # CASE 2: Smart Start (Best Model 가중치 로드)
    elif os.path.exists(best_model_path) and os.path.exists(os.path.join(best_model_path, "model.safetensors")):
        print(f"⚠️ No checkpoint found, but found 'best_model' at: {best_model_path}")
        print(">> Loading weights from best_model to continue training (Smart Start).")
        
        # 모델 로드
        model = EncoderDecoderModel.from_pretrained(best_model_path)
        
        # [수정] 토크나이저 재로드 (안전성을 위해 명시적 로드)
        # 인코더용: Config의 원본 모델명 사용 (또는 별도 저장된 경로)
        tokenizer_enc = AutoTokenizer.from_pretrained(cfg.ENCODER_NAME)
        # 디코더용: 학습된 모델 폴더에서 로드 (Trainer가 저장한 것)
        tokenizer_dec = AutoTokenizer.from_pretrained(best_model_path)
        
        # 특수 토큰 재적용 (필요 시)
        if cfg.SPECIAL_TOKENS:
            special_tokens = {'additional_special_tokens': cfg.SPECIAL_TOKENS}
            tokenizer_enc.add_special_tokens(special_tokens)
            tokenizer_dec.add_special_tokens(special_tokens)
            model.resize_token_embeddings(len(tokenizer_dec)) # 디코더 기준 리사이징 재확인

        # 모델 설정 복구
        model.config.decoder_start_token_id = tokenizer_dec.cls_token_id
        model.config.pad_token_id = tokenizer_dec.pad_token_id
        model.config.vocab_size = len(tokenizer_dec)
        
        last_checkpoint = None

    # CASE 3: 처음부터 시작
    else:
        print("❌ No checkpoint or best_model found.")
        print(">> Starting training from scratch.")
        # [수정] 반환값 3개 언패킹
        model, tokenizer_enc, tokenizer_dec = build_modernbert_b2b_model(cfg)

    model.to(device)
    
    # 3. 데이터셋 로드
    data_files = {
        "train": cfg.TRAIN_DATA_PATH,
        "validation": cfg.VAL_DATA_PATH
    }
    print(f"Loading datasets from: {data_files}")
    dataset = load_dataset("json", data_files=data_files)
    
    # 4. 전처리 (Mapping)
    cpu_cores = os.cpu_count()
    # [수정] map 함수에 tokenizer_enc, tokenizer_dec 모두 전달
    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer_enc, tokenizer_dec, cfg),
        batched=True,
        remove_columns=dataset["train"].column_names,
        num_proc=cpu_cores
    )

    # 5. Data Collator
    # [중요] 평가(Generation) 시 디코딩을 위해 '디코더 토크나이저'를 전달
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer_dec,
        model=model,
        label_pad_token_id=-100
    )

    # 6. Training Arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=cfg.OUTPUT_DIR,
        per_device_train_batch_size=cfg.BATCH_SIZE,
        per_device_eval_batch_size=cfg.BATCH_SIZE,
        gradient_accumulation_steps=cfg.GRADIENT_ACCUMULATION,
        learning_rate=cfg.LEARNING_RATE,
        num_train_epochs=cfg.NUM_EPOCHS,
        weight_decay=cfg.WEIGHT_DECAY,
        warmup_ratio=cfg.WARMUP_RATIO,
        label_smoothing_factor=cfg.LABEL_SMOOTHING,
        
        logging_dir=f"{cfg.OUTPUT_DIR}/logs",
        logging_steps=50,
        
        save_strategy=cfg.SAVE_STRATEGY,
        eval_strategy=cfg.EVAL_STRATEGY,
        eval_steps=500,
        save_steps=500,
        
        save_total_limit=1,            
        load_best_model_at_end=True,  
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        bf16=use_bf16,
        fp16=use_fp16,
        
        dataloader_num_workers=4,
        predict_with_generate=True, # 생성 모델 평가 켜기
        
        # [수정] WandB 활성화
        report_to="wandb",  
        run_name=f"run-{cfg.ENCODER_NAME.split('/')[-1]}-B2B", # 실행 이름 자동 지정
        
        disable_tqdm=False,
        log_level="error"
    )

    # 8. Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        # [중요] Trainer에 디코더 토크나이저 전달 (Metric 계산용)
        tokenizer=tokenizer_dec,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    print("Removing PrinterCallback to keep console clean...")
    trainer.remove_callback(PrinterCallback)

    # 9. 학습 시작
    print("\nStarting Training...")
    trainer.train(resume_from_checkpoint=last_checkpoint)

    # 10. 모델 및 토크나이저 저장
    print(f"\nSaving best model to {cfg.OUTPUT_DIR}/best_model")
    
    # (1) 모델과 디코더 토크나이저는 Trainer가 저장
    trainer.save_model(f"{cfg.OUTPUT_DIR}/best_model")
    
    # (2) [수정] 인코더 토크나이저는 Trainer가 모르므로 수동 저장
    # 나중에 추론할 때 반드시 필요함
    encoder_tok_path = os.path.join(cfg.OUTPUT_DIR, "best_model", "encoder_tokenizer")
    tokenizer_enc.save_pretrained(encoder_tok_path)
    print(f"✅ Saved Encoder Tokenizer manually to: {encoder_tok_path}")

    # 학습 기록 저장
    history_path = os.path.join(cfg.OUTPUT_DIR, "training_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, indent=4)

    print("Training finished successfully!")

if __name__ == "__main__":
    main()