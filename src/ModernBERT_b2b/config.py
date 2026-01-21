from dataclasses import dataclass

@dataclass
class Config:
    # 1. 경로 및 모델 설정
    ENCODER_NAME: str = "answerdotai/ModernBERT-base" 
    DECODER_NAME: str = "roberta-base"              
    
    TRAIN_DATA_PATH: str = "/content/train_data/totto_modernbert_train.json"
    VAL_DATA_PATH: str = "/content/train_data/totto_modernbert_dev.json"
    # 구글 드라이브 경로 확인 (마운트가 되어 있어야 함)
    OUTPUT_DIR: str = "/content/drive/MyDrive/models"
    
    # 2. 특수 토큰
    SPECIAL_TOKENS = [
        "[PAGE]", "[SEC]", "[TEXT]", 
        "[CELL]", "[TYPE]", "[R_HEAD]", "[C_HEAD]", 
        "[H]", "[/H]"
    ]

    # 3. 데이터 관련 설정
    MAX_INPUT_LENGTH: int = 1024 
    MAX_TARGET_LENGTH: int = 128 
    
    # 4. 학습 하이퍼 파라미터 (Effective Batch Size = 64 * 4 = 256)
    BATCH_SIZE: int = 64 
    GRADIENT_ACCUMULATION: int = 4 
    LEARNING_RATE: float = 5e-5
    NUM_EPOCHS: int = 25
    WARMUP_RATIO: float = 0.1  # 높은 LR에 대비해 웜업 비율 상향 (Good)
    WEIGHT_DECAY: float = 0.1
    
    # 5. 생성 및 손실 함수 관련
    LABEL_SMOOTHING: float = 0.05
    NUM_BEAMS: int = 4 
    
    # 6. 로깅 및 저장
    LOGGING_STEPS: int = 25       # 배치가 커졌으므로 로그를 좀 더 자주 확인 (100 -> 50)
    
    # [수정 제안] Epoch 단위는 너무 기므로 Step 단위로 변경하여 안전하게 저장
    SAVE_STRATEGY: str = "steps"
    EVAL_STRATEGY: str = "steps"
    SAVE_STEPS: int = 500         # 약 20~30분마다 저장/평가
    
    # [수정 제안] A100/H100 등 최신 GPU라면 BF16 사용 (Config에 필드 추가)
    FP16: bool = False            # BF16을 쓴다면 False로
    BF16: bool = True             # A100이라면 True 권장