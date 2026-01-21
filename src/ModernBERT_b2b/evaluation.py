import os
import json
import torch
from tqdm import tqdm
import sacrebleu
from transformers import AutoTokenizer, EncoderDecoderModel
from config import Config

# [유지] 직접 만든 PARENT 지표 모듈 임포트
from parent_metric import parent_score

class ModelEvaluator:
    def __init__(self, config):
        self.cfg = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 모델 가중치 파일 경로 설정
        BASE_DIR = "/content/drive/MyDrive/models"
        self.model_path = os.path.join(BASE_DIR, "best_model")
        # 결과 저장할 디렉토리 설정
        self.result_dir = os.path.join(BASE_DIR, "results")
        os.makedirs(self.result_dir, exist_ok=True)

        print(f"[Info] Loading best model from {self.model_path}...")
        
        self.model = EncoderDecoderModel.from_pretrained(self.model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # [설정] 시작 토큰 강제 지정 (에러 방지)
        if self.tokenizer.cls_token_id is not None:
            self.model.config.decoder_start_token_id = self.tokenizer.cls_token_id
        elif self.tokenizer.bos_token_id is not None:
            self.model.config.decoder_start_token_id = self.tokenizer.bos_token_id
        else:
            self.model.config.decoder_start_token_id = self.tokenizer.pad_token_id
            
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.vocab_size = self.model.config.encoder.vocab_size

        print(f"[Info] Set decoder_start_token_id to: {self.model.config.decoder_start_token_id}")
        self.model.eval()

    def generate_batch(self, batch_inputs):
        # [최적화] 여러 문장을 한 번에 토큰화 (Padding 적용)
        inputs = self.tokenizer(
            batch_inputs,
            max_length=self.cfg.MAX_INPUT_LENGTH,
            padding=True, # 배치 처리를 위해 필수
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                inputs["input_ids"],
                max_length=self.cfg.MAX_TARGET_LENGTH,
                num_beams=self.cfg.NUM_BEAMS,
                early_stopping=True,
                decoder_start_token_id=self.model.config.decoder_start_token_id,
                pad_token_id=self.tokenizer.pad_token_id
            )

        decoded_preds = self.tokenizer.batch_decode(
            generated_ids, 
            skip_special_tokens=True
        )
        return decoded_preds

    def run_evaluation(self):
        print(f"[Info] Loading validation data from {self.cfg.VAL_DATA_PATH}")
        with open(self.cfg.VAL_DATA_PATH, "r", encoding="utf-8") as f:
            val_data = json.load(f)

        # 전체 결과를 저장할 리스트
        all_predictions = []
        all_references = []
        all_tables = []

        # [최적화] 배치 처리를 위한 임시 버퍼
        batch_inputs = []
        batch_refs = []
        batch_tables = []
        
        batch_size = self.cfg.BATCH_SIZE
        print(f"[Info] Starting generation for {len(val_data)} examples (Batch Size: {batch_size})...")
        
        for i, example in enumerate(tqdm(val_data)):
            # 1. 데이터 파싱
            input_text = example["input"]
            target = example["target"]
            
            # Reference 처리
            if isinstance(target, list):
                ref_list = target 
            else:
                ref_list = [target]
                
            # Table 데이터 추출
            raw_table = example.get("table", [])
            table_values = []
            for cell in raw_table:
                val = cell.get("value", "")
                if val:
                    table_values.append(str(val))
            
            # 2. 배치 버퍼에 추가
            batch_inputs.append(input_text)
            batch_refs.append(ref_list)
            batch_tables.append(table_values)

            # 3. 배치가 꽉 찼거나 마지막 데이터인 경우 실행
            if len(batch_inputs) >= batch_size or i == len(val_data) - 1:
                # 배치 단위 생성
                preds = self.generate_batch(batch_inputs)
                
                # 결과 저장
                all_predictions.extend(preds)
                all_references.extend(batch_refs)
                all_tables.extend(batch_tables)
                
                # 버퍼 초기화
                batch_inputs = []
                batch_refs = []
                batch_tables = []

        # --- 지표 계산 ---
        print("\n[Info] Calculating metrics...")
        
        # 1. BLEU Score
        ref_for_bleu = [[r[0] for r in all_references]] 
        bleu = sacrebleu.corpus_bleu(all_predictions, ref_for_bleu)
        
        # 2. PARENT Score
        precision, recall, f1 = parent_score(all_predictions, all_references, all_tables)
        
        # 결과 저장
        evaluation_results = {
            "metadata": {
                "model_path": self.model_path,
                "dataset_path": self.cfg.VAL_DATA_PATH,
                "num_samples": len(val_data)
            },
            "metrics": {
                "bleu": round(bleu.score, 2),
                "parent": {
                    "precision": round(precision * 100, 2),
                    "recall": round(recall * 100, 2),
                    "f1": round(f1 * 100, 2)
                }
            }
        }

        score_json_path = os.path.join(self.result_dir, "evaluation_results.json")
        
        with open(score_json_path, "w", encoding="utf-8") as f_json:
            json.dump(evaluation_results, f_json, indent=4, ensure_ascii=False)
            
        print("\n" + "="*50)
        print(f"Evaluation Results Saved to: {score_json_path}")
        print(f"BLEU Score: {evaluation_results['metrics']['bleu']}")
        print(f"PARENT F1 : {evaluation_results['metrics']['parent']['f1']}")
        print("="*50 + "\n")

        return evaluation_results

if __name__ == "__main__":
    cfg = Config()
    evaluator = ModelEvaluator(cfg)
    evaluator.run_evaluation()