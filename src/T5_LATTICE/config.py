# config.py

class Config:
    # 1. 경로 설정
    MODEL_NAME = "t5-base"
    TRAIN_FILE_PATH = "train_data/totto_train_LATTICE.json"  # 실제 데이터 경로로 수정 필요
    DEV_FILE_PATH = "train_data/totto_dev_LATTICE.json"
    OUTPUT_DIR = "/content/drive/MyDrive/models_T5L"
    
    # 2. 스페셜 토큰 정의
    SPECIAL_TOKENS = [
        "[CLEAN]", "[DRAFT]", "[PAGE]", "[SEC]", 
        "[TEXT]", "[CELL]", "[TYPE]", "[R_HEAD]", 
        "[C_HEAD]", "|" 
    ]
    
    # 3. 데이터셋 설정
    MAX_INPUT_LENGTH = 512
    MAX_TARGET_LENGTH = 128
    
    # 4. 학습 하이퍼파라미터 (안정성 강화)
    # [Modified] A100에서는 64 배치를 한번에 처리하는 것이 속도/성능 면에서 유리합니다.
    BATCH_SIZE = 64
    
    # [Modified] 누적을 1로 줄여서 실질 배치를 64로 맞춥니다. (기존 256은 너무 컸음)
    GRADIENT_ACCUMULATION_STEPS = 1 
    
    # [Modified] 배치를 줄였으니 LR을 다시 표준값(5e-5 ~ 1e-4)으로 올리는 것을 추천합니다.
    # 1e-5는 너무 작아서 학습이 더디거나 Local Minima에 갇힐 수 있습니다.
    LEARNING_RATE = 5e-5  
    
    NUM_EPOCHS = 8
    
    # [Modified] 전체 스텝의 6~10%를 웜업으로 사용 (Ratio 권장)
    # Train.py에서 getattr(Config, 'WARMUP_RATIO', 0.0)으로 읽어야 함
    WARMUP_RATIO = 0.1  
    # WARMUP_STEPS = 800 # (주석 처리)
    
    WEIGHT_DECAY = 0.01
    
    # 5. 로깅 및 저장 설정
    LOGGING_STEPS = 50   # [Modified] 초반 추이를 더 자주 보기 위해 줄임
    EVAL_STEPS = 500
    SAVE_STEPS = 500
    SAVE_TOTAL_LIMIT = 2 # Best Model과 최신 체크포인트 유지를 위해 2개 권장
    
    # 6. Lattice Attention 설정
    LATTICE_PENALTY = -10.0