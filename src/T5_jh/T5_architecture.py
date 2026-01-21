import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class T5Dataset(Dataset):
    def __init__(self, data, tokenizer, max_input_len=512, max_target_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 입력 데이터 (input)
        input_text = item.get("input", "")
        # 정답 데이터 (target)
        target_text = item.get("target", "")

        # 토크나이징 (Input)
        model_inputs = self.tokenizer(
            input_text,
            max_length=self.max_input_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # 토크나이징 (Target/Label)
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                target_text,
                max_length=self.max_target_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

        # squeeze(0)으로 배치 차원 제거 (DataLoader가 배치를 만들기 때문)
        inputs = {k: v.squeeze(0) for k, v in model_inputs.items()}
        
        # Label의 패딩 토큰은 loss 계산 시 무시되도록 -100으로 설정
        labels_ids = labels["input_ids"].squeeze(0)
        labels_ids[labels_ids == self.tokenizer.pad_token_id] = -100
        inputs["labels"] = labels_ids

        return inputs

def get_model_and_tokenizer(model_name):
    """
    [수정됨] T5는 Flash Attention 2를 지원하지 않으므로 해당 옵션 제거.
    하지만 A100 성능을 위해 bf16 로딩은 유지합니다.
    """
    print(f"Loading model: {model_name} (Optimization: bf16 enabled)...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 1. 추가할 스페셜 토큰 정의
    special_tokens_list = [
        "[CLEAN]", "[DRAFT]",                        # Task Prefix
        "[PAGE]", "[SEC]", "[TEXT]",                 # Context Tags
        "[CELL]", "[TYPE]", "[R_HEAD]", "[C_HEAD]"   # Table Tags
    ]

    # 2. 토크나이저에 새로운 토큰 추가
    num_added_toks = tokenizer.add_tokens(special_tokens_list)
    print(f"Added {num_added_toks} special tokens: {special_tokens_list}")

    # 3. 모델 로드 (bf16만 유지)
    # attn_implementation="flash_attention_2" 삭제됨
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16  # A100을 위한 bf16은 유지
    )

    # 4. 모델의 임베딩 레이어 크기 조정
    if num_added_toks > 0:
        model.resize_token_embeddings(len(tokenizer))
        print(f"Resized token embeddings to {len(tokenizer)}")
        
    return model, tokenizer