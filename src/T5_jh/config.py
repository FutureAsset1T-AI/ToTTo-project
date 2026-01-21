import os

class Config:
    # 1. 데이터 경로 설정
    # (사용자의 환경에 맞게 수정하세요)
    TRAIN_FILE_PATH = "./train_data/totto_train_aggumented.json"
    VAL_FILE_PATH = "./train_data/totto_dev_aggumented.json"
    
    # 2. 모델 저장 및 로그 경로
    OUTPUT_DIR = "/content/drive/MyDrive/models/t5_totto_checkpoints"
    BEST_MODEL_DIR = "/content/drive/MyDrive/models/t5_totto_best_model"
    
    # 3. 모델 하이퍼파라미터
    MODEL_NAME = "t5-base"
    MAX_INPUT_LEN = 512
    MAX_TARGET_LEN = 128
    
    # 4. 학습 파라미터
    BATCH_SIZE = 32 # GPU 메모리에 따라 조절 (OOM 발생 시 줄이세요)
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 3
    GRADIENT_ACCUMULATION_STEPS = 4
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.01
    
    # 5. 로깅 및 저장 설정
    LOGGING_STEPS = 100
    EVAL_STEPS = 500
    SAVE_STEPS = 500
    SAVE_TOTAL_LIMIT = 2  # 저장할 체크포인트 최대 개수 (공간 절약)
    
    # 6. WandB 설정
    WANDB_PROJECT = "totto-t5-project"
    WANDB_ENTITY = None # 팀 계정일 경우 입력, 개인일 경우 None
    WANDB_NAME = "t5-base-training-run"

    # 시드 설정
    SEED = 42

config = Config()