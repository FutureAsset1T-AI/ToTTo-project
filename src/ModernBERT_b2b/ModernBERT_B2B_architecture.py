import torch
from transformers import EncoderDecoderModel, AutoTokenizer
from config import Config

# [유지] 요청하신 대로 함수 이름은 변경하지 않았습니다.
def build_modernbert_b2b_model(config: Config = Config()):
    """
    ModernBERT(Encoder) + RoBERTa(Decoder) 하이브리드 모델 빌드 함수
    (Dual Tokenizer Strategy 적용됨)
    """
    print(f"[Build] Loading Dual Tokenizers for Hybrid Model...")
    
    # [수정 1] 인코더와 디코더의 토크나이저를 각각 별도로 로드
    # 이유: 두 모델의 어휘 집합(Vocab)이 서로 직교(Orthogonal)하므로 분리해야 함
    tokenizer_enc = AutoTokenizer.from_pretrained(config.ENCODER_NAME)
    tokenizer_dec = AutoTokenizer.from_pretrained(config.DECODER_NAME)

    # [수정 2] 특수 토큰 추가 로직 분리
    if config.SPECIAL_TOKENS:
        special_tokens_dict = {'additional_special_tokens': config.SPECIAL_TOKENS}
        # 인코더와 디코더 양쪽 모두에 새로운 특수 토큰을 등록
        tokenizer_enc.add_special_tokens(special_tokens_dict)
        tokenizer_dec.add_special_tokens(special_tokens_dict)
        print(f"[Build] Added special tokens. Enc Vocab: {len(tokenizer_enc)}, Dec Vocab: {len(tokenizer_dec)}")

    # 3. Encoder-Decoder 모델 로드
    print(f"[Build] Loading Hybrid Model... Encoder: {config.ENCODER_NAME} / Decoder: {config.DECODER_NAME}")
    
    model = EncoderDecoderModel.from_encoder_decoder_pretrained(
        encoder_pretrained_model_name_or_path=config.ENCODER_NAME,
        decoder_pretrained_model_name_or_path=config.DECODER_NAME,
        # [수정 3] 임베딩 공유 비활성화 (tie_encoder_decoder=False)
        # 이유: 서로 다른 Token ID 체계를 가지므로 가중치를 공유하면 학습이 불가능함
        tie_encoder_decoder=False
    )

    # [수정 4] 임베딩 리사이징을 각 모델의 토크나이저 크기에 맞춰 개별 수행
    print(f"[Build] Resizing embeddings independently...")
    model.encoder.resize_token_embeddings(len(tokenizer_enc))
    model.decoder.resize_token_embeddings(len(tokenizer_dec))

    # [수정 5] 모델 설정(Config)은 '디코더 토크나이저'를 기준으로 동기화
    # 디코더가 텍스트를 생성할 때 사용하는 ID는 tokenizer_dec의 것이기 때문
    model.config.decoder_start_token_id = tokenizer_dec.cls_token_id  # RoBERTa의 경우 <s>
    model.config.pad_token_id = tokenizer_dec.pad_token_id
    model.config.eos_token_id = tokenizer_dec.sep_token_id
    
    # Vocab Size 설정도 각각의 크기로 지정
    model.config.vocab_size = len(tokenizer_dec)
    model.config.encoder.vocab_size = len(tokenizer_enc)
    model.config.decoder.vocab_size = len(tokenizer_dec)

    # [유지] 생성(Generation) 하이퍼파라미터 설정
    model.config.max_length = config.MAX_TARGET_LENGTH
    model.config.min_length = 5
    model.config.no_repeat_ngram_size = 3
    model.config.early_stopping = True
    model.config.length_penalty = 2.0
    model.config.num_beams = config.NUM_BEAMS

    print("[Build] Model initialized successfully with Dual Tokenizers.")
    
    # [수정 6] 반환값 변경: (model, tokenizer) -> (model, tokenizer_enc, tokenizer_dec)
    # 데이터셋 클래스에서 입력과 라벨을 처리할 때 서로 다른 토크나이저가 필요하므로 둘 다 반환
    return model, tokenizer_enc, tokenizer_dec