import json
import torch
import os
import transformers
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, 
    AutoConfig, 
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback
)
# [Added] 기본 로그 출력을 제어하기 위해 Import
from transformers.trainer_callback import PrinterCallback
from transformers.trainer_utils import get_last_checkpoint

# 사용자 정의 모듈 Import
from config import Config
from T5_LATTICE_architecture import LatticeT5ForConditionalGeneration, TottoLatticeDataCollator

# [Part 1] Custom Dataset (이전과 동일)
class TottoDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_input_len, max_target_len):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len
        self.data = self._load_data()

    def _load_data(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {self.data_path}")
            
        with open(self.data_path, 'r', encoding='utf-8') as f:
            if self.data_path.endswith('.jsonl'):
                return [json.loads(line) for line in f if line.strip()]
            return json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        segments = item.get("segments", [])
        segment_coords = item.get("segment_coords", [])
        target_text = item.get("target", "")
        
        full_input_ids = []
        full_coords = []
        
        for text, coord in zip(segments, segment_coords):
            token_ids = self.tokenizer.encode(str(text), add_special_tokens=False)
            full_input_ids.extend(token_ids)
            full_coords.extend([coord] * len(token_ids))
            
        if len(full_input_ids) > self.max_input_len:
            full_input_ids = full_input_ids[:self.max_input_len]
            full_coords = full_coords[:self.max_input_len]
            
        labels = self.tokenizer(
            target_text, 
            max_length=self.max_target_len, 
            truncation=True
        ).input_ids
        
        return {
            "input_ids": torch.tensor(full_input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "cell_coords": full_coords 
        }

# =============================================================================
# [Part 2] Main Training Function
# =============================================================================
def main():
    print(f"Loading Model: {Config.MODEL_NAME}")
    transformers.logging.set_verbosity_error()
    
    last_checkpoint = None
    if os.path.isdir(Config.OUTPUT_DIR):
        last_checkpoint = get_last_checkpoint(Config.OUTPUT_DIR)
        if last_checkpoint is not None:
            print(f"Found checkpoint: {last_checkpoint}. Resuming training from here.")

    # 1. 토크나이저 및 모델 설정
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    tokenizer.add_special_tokens({"additional_special_tokens": Config.SPECIAL_TOKENS})
    
    config = AutoConfig.from_pretrained(Config.MODEL_NAME)
    if hasattr(Config, 'LATTICE_PENALTY'):
        config.LATTICE_PENALTY = Config.LATTICE_PENALTY

    model = LatticeT5ForConditionalGeneration.from_pretrained(Config.MODEL_NAME, config=config)
    model.resize_token_embeddings(len(tokenizer))

    # 2. 데이터셋 로드
    print("Loading Datasets...")
    train_dataset = TottoDataset(Config.TRAIN_FILE_PATH, tokenizer, Config.MAX_INPUT_LENGTH, Config.MAX_TARGET_LENGTH)
    
    eval_dataset = None
    if hasattr(Config, 'DEV_FILE_PATH') and os.path.exists(Config.DEV_FILE_PATH):
        print(f"Validation Dataset Found: {Config.DEV_FILE_PATH}")
        eval_dataset = TottoDataset(Config.DEV_FILE_PATH, tokenizer, Config.MAX_INPUT_LENGTH, Config.MAX_TARGET_LENGTH)
    else:
        print("Warning: Validation dataset not found.")

    data_collator = TottoLatticeDataCollator(tokenizer=tokenizer, model=model, label_pad_token_id=-100)

    # 3. Precision 설정
    is_bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_bf16 = is_bf16_supported
    use_fp16 = torch.cuda.is_available() and not is_bf16_supported
    print(f"\n[Hardware Check] Precision Mode: {'BF16' if use_bf16 else 'FP16' if use_fp16 else 'FP32'}")

    # 4. 학습 설정
    training_args = Seq2SeqTrainingArguments(
        output_dir=Config.OUTPUT_DIR,
        report_to="wandb", 
        run_name="t5-lattice-finetune-stable",
        overwrite_output_dir=True,
        learning_rate=Config.LEARNING_RATE,
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.BATCH_SIZE,
        gradient_accumulation_steps=getattr(Config, 'GRADIENT_ACCUMULATION_STEPS', 1),
        num_train_epochs=Config.NUM_EPOCHS,
        weight_decay=Config.WEIGHT_DECAY,
        warmup_ratio=getattr(Config, 'WARMUP_RATIO', 0.0),
        warmup_steps=0, 
        max_grad_norm=1.0,

        eval_strategy="steps",  
        eval_steps=Config.EVAL_STEPS,
        save_strategy="steps",
        save_steps=Config.SAVE_STEPS,
        save_total_limit=getattr(Config, 'SAVE_TOTAL_LIMIT', 2),
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        logging_steps=Config.LOGGING_STEPS,
        
        predict_with_generate=True,
        log_level="error",     
        disable_tqdm=False,    
        
        fp16=use_fp16,
        bf16=use_bf16,
        remove_unused_columns=False,
    )

    # 5. Trainer 초기화
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] 
    )

    # [Added] 로그 딕셔너리 출력을 끄는 핵심 코드
    # 기본값으로 활성화된 PrinterCallback을 제거하여 터미널에 {'loss':...}가 찍히는 것을 방지합니다.
    # 진행바(tqdm)와 WandB 로그만 남게 됩니다.
    trainer.remove_callback(PrinterCallback)

    print("Starting Training...")
    trainer.train(resume_from_checkpoint=last_checkpoint)

    print(f"Saving Best Model to {Config.OUTPUT_DIR}")
    trainer.save_model(Config.OUTPUT_DIR)
    tokenizer.save_pretrained(Config.OUTPUT_DIR)

if __name__ == "__main__":
    main()