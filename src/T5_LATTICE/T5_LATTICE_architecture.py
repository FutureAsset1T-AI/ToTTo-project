import torch
import torch.nn as nn
from transformers import DataCollatorForSeq2Seq, T5ForConditionalGeneration
from transformers.models.t5.modeling_t5 import T5Attention

# =============================================================================
# [Part 1] Data Collator (로그 출력 제거 및 최적화)
# =============================================================================
class TottoLatticeDataCollator(DataCollatorForSeq2Seq):
    def __init__(self, tokenizer, model=None, padding=True, max_length=None, pad_to_multiple_of=None, label_pad_token_id=-100):
        super().__init__(tokenizer, model, padding, max_length, pad_to_multiple_of, label_pad_token_id)
        
    def __call__(self, features, return_tensors=None):
        # [Removed] 이전 코드에서 print(batch["labels"])와 같은 출력문이 있었다면 삭제되었습니다.
        
        # 1. cell_coords 분리
        cell_coords_batch = [f.pop("cell_coords", []) for f in features]

        # 2. 기본 HuggingFace Collator 수행
        batch = super().__call__(features, return_tensors=return_tensors)
        
        # 3. Lattice 정보 텐서화
        batch_size = batch['input_ids'].shape[0]
        max_seq_len = batch['input_ids'].shape[1]
        
        lattice_rows = torch.full((batch_size, max_seq_len), -1, dtype=torch.long)
        lattice_cols = torch.full((batch_size, max_seq_len), -1, dtype=torch.long)
        
        for i, coords in enumerate(cell_coords_batch):
            length = min(len(coords), max_seq_len)
            for j in range(length):
                # 학술적 정의: Lattice 구조 정보(Row, Col)를 텐서로 변환하여 
                # Attention Mask 연산 시 효율적인 Broadcasting이 가능하도록 준비합니다.
                if coords[j][0] != -1: 
                    lattice_rows[i, j] = coords[j][0]
                    lattice_cols[i, j] = coords[j][1]
        
        batch["lattice_rows"] = lattice_rows
        batch["lattice_cols"] = lattice_cols
        
        return batch


# =============================================================================
# [Part 2] Lattice Attention
# =============================================================================
class LatticeT5Attention(T5Attention):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__(config, has_relative_attention_bias)
        self.w_row = nn.Parameter(torch.tensor(0.0)) 
        self.w_col = nn.Parameter(torch.tensor(0.0))
        self.penalty = getattr(config, 'LATTICE_PENALTY', -10.0)
        
        self.current_lattice_rows = None
        self.current_lattice_cols = None

    def compute_lattice_bias(self, batch_size, seq_len, lattice_rows, lattice_cols, device):
        row_expanded = lattice_rows.unsqueeze(2) # (B, L, 1)
        row_transposed = lattice_rows.unsqueeze(1) # (B, 1, L)
        
        valid_mask = (row_expanded != -1) & (row_transposed != -1)
        is_same_row = (row_expanded == row_transposed) & valid_mask

        col_expanded = lattice_cols.unsqueeze(2)
        col_transposed = lattice_cols.unsqueeze(1)
        is_same_col = (col_expanded == col_transposed) & valid_mask

        lattice_bias = torch.full((batch_size, seq_len, seq_len), self.penalty, device=device)
        is_related = is_same_row | is_same_col
        
        # 학술적 정의: 구조적 편향(Structural Bias)을 Attention Score에 가산하여 
        # 모델이 테이블의 기하학적 관계(동일 행/열)를 인지하도록 유도합니다.
        base_score = torch.where(is_related, torch.tensor(0.0, device=device), torch.tensor(self.penalty, device=device))
        final_bias = base_score + (is_same_row.float() * self.w_row) + (is_same_col.float() * self.w_col)
        
        return final_bias.unsqueeze(1) 

    def forward(self, hidden_states, mask=None, key_value_states=None, position_bias=None, **kwargs):
        lattice_rows = getattr(self, "current_lattice_rows", None)
        lattice_cols = getattr(self, "current_lattice_cols", None)
        
        if lattice_rows is not None and lattice_cols is not None:
            lattice_rows = lattice_rows.to(hidden_states.device)
            lattice_cols = lattice_cols.to(hidden_states.device)
            batch_size, seq_len = hidden_states.shape[:2]
            
            struct_bias = self.compute_lattice_bias(
                batch_size, seq_len, lattice_rows, lattice_cols, hidden_states.device
            )
            
            if mask is None:
                mask = struct_bias
            else:
                mask = mask + struct_bias
        
        return super().forward(
            hidden_states, mask=mask, key_value_states=key_value_states, 
            position_bias=position_bias, **kwargs
        )


# =============================================================================
# [Part 3] Wrapper Model (로그 출력 제거 및 메모리 정리)
# =============================================================================
class LatticeT5ForConditionalGeneration(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.swap_attention_layers()

    def swap_attention_layers(self):
        for i, block in enumerate(self.encoder.block):
            old_attn = block.layer[0].SelfAttention
            new_attn = LatticeT5Attention(self.config, has_relative_attention_bias=old_attn.has_relative_attention_bias)
            new_attn.load_state_dict(old_attn.state_dict(), strict=False)
            block.layer[0].SelfAttention = new_attn

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None, labels=None, **kwargs):
        # [Removed] 여기에 print(labels) 등의 출력 코드가 있었다면 제거되었습니다.

        lattice_rows = kwargs.pop("lattice_rows", None)
        lattice_cols = kwargs.pop("lattice_cols", None)
        kwargs.pop("num_items_in_batch", None)
        
        if lattice_rows is not None and lattice_cols is not None:
            for block in self.encoder.block:
                if hasattr(block.layer[0], 'SelfAttention'):
                    attn_layer = block.layer[0].SelfAttention
                    if isinstance(attn_layer, LatticeT5Attention):
                        attn_layer.current_lattice_rows = lattice_rows
                        attn_layer.current_lattice_cols = lattice_cols

        # 학술적 정의: Superclass Forwarding은 모델 아키텍처의 핵심 로직을 유지하면서 
        # 입력 데이터만 커스터마이징하여 전달하는 상속 기반의 모델 확장 방식입니다.
        outputs = super().forward(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            decoder_input_ids=decoder_input_ids,
            labels=labels, 
            **kwargs
        )

        # [Added] 메모리 누수 방지: 매 스텝 종료 후 주입된 Lattice 정보 초기화
        for block in self.encoder.block:
            attn_layer = block.layer[0].SelfAttention
            if isinstance(attn_layer, LatticeT5Attention):
                attn_layer.current_lattice_rows = None
                attn_layer.current_lattice_cols = None

        return outputs