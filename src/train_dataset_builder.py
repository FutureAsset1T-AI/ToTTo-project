import json
import pandas as pd
import os
from ast import literal_eval
from typing import List, Dict, Any, Optional

class ToTToIntegratedProcessor:
    def __init__(self, input_path: str):
        self.input_path = input_path
        self.df = None

    def load_and_filter(self, quantile_cutoff: Optional[float] = 0.99):
        """데이터를 로드하고 설정에 따라 필터링을 수행합니다."""
        if not os.path.exists(self.input_path):
            print(f"오류: 파일을 찾을 수 없습니다 -> {self.input_path}")
            return

        print(f"데이터 로딩 중: {self.input_path}")
        self.df = pd.read_json(self.input_path, lines=True)
        
        if quantile_cutoff is None:
            print(f"▶ 필터링 건너뜀 (모든 데이터 사용)")
            print(f"▶ 데이터 총 개수: {len(self.df)}")
            return

        def get_count(x):
            if isinstance(x, str):
                try: x = literal_eval(x)
                except: return 0
            return len(x) if isinstance(x, list) else 0

        highlight_counts = self.df["highlighted_cells"].apply(get_count)
        cutoff = highlight_counts.quantile(quantile_cutoff)
        
        print(f"필터링 기준 (상위 {int((1-quantile_cutoff)*100)}%): {cutoff}개 이상의 셀 제외")
        self.df = self.df[highlight_counts <= cutoff].copy()
        print(f"필터링 후 남은 데이터 수: {len(self.df)}")

    def _get_hierarchical_headers(self, table: List[List[Dict]], row_idx: int, col_idx: int, mode: str) -> List[str]:
        """계층적 헤더 텍스트 추출"""
        headers = []
        if mode == "col":
            for r in range(row_idx):
                if r < len(table) and col_idx < len(table[r]):
                    if table[r][col_idx].get("is_header"):
                        headers.append(str(table[r][col_idx]["value"]))
        else:
            if row_idx < len(table):
                for c in range(col_idx):
                    if c < len(table[row_idx]):
                        if table[row_idx][c].get("is_header"):
                            headers.append(str(table[row_idx][c]["value"]))
        
        # 중복 제거 (순서 유지)
        seen = set()
        unique_headers = []
        for h in headers:
            if h not in seen:
                seen.add(h)
                unique_headers.append(h)
        return unique_headers

    def process_row(self, row: pd.Series) -> Optional[str]:
        """
        [Standard T5] 한 행을 단일 입력 문자열로 변환합니다.
        태그 없이 순수한 선형화 텍스트만 반환합니다.
        """
        table = row['table']
        page_title = str(row.get('table_page_title', 'None'))
        sec_title = str(row.get('table_section_title', 'None'))
        
        # 메타데이터
        input_parts = ["[PAGE]", page_title, "[SEC]", sec_title]

        highlighted = row['highlighted_cells']
        if isinstance(highlighted, str):
            highlighted = literal_eval(highlighted)

        # 좌표 유효성 검사
        for r_idx, c_idx in highlighted:
            if r_idx >= len(table) or c_idx >= len(table[r_idx]):
                return None 

        # Highlighted Cells 순회
        for r_idx, c_idx in highlighted:
            cell = table[r_idx][c_idx]
            val = str(cell['value'])
            is_hdr_bool = cell.get('is_header', False)
            is_hdr_str = "T" if is_hdr_bool else "F"

            # [CELL] 정보 추가
            input_parts.extend(["[CELL]", val, "[TYPE]", is_hdr_str])

            if is_hdr_bool:
                input_parts.extend(["[R_HEAD]", "None", "[C_HEAD]", "None"])
            else:
                # Row Headers
                r_headers = self._get_hierarchical_headers(table, r_idx, c_idx, "row")
                input_parts.append("[R_HEAD]")
                input_parts.append(" | ".join(r_headers) if r_headers else "None")

                # Col Headers
                c_headers = self._get_hierarchical_headers(table, r_idx, c_idx, "col")
                input_parts.append("[C_HEAD]")
                input_parts.append(" | ".join(c_headers) if c_headers else "None")

        return " ".join(input_parts)

    def print_sample(self):
        """샘플 출력"""
        if self.df is None or self.df.empty:
            return

        print("\n" + "="*80)
        print(" [SIMPLE T5 PREPROCESSED SAMPLE CHECK] ")
        print("="*80)
        
        for _, row in self.df.iterrows():
            input_str = self.process_row(row)
            if input_str is not None:
                annotations = row.get('sentence_annotations', [])
                target_text = annotations[0].get('final_sentence', 'N/A') if annotations else 'N/A'

                print(f"▶ INPUT:\n{input_str}")
                print("-" * 80)
                print(f"▶ TARGET:\n{target_text}")
                print("="*80 + "\n")
                return

    def save_to_json(self, output_path: str):
        """
        [수정됨] 
        1. [CLEAN]/[DRAFT] 태그 제거
        2. 데이터 증강(ambiguity sentence) 로직 제거
        3. 오직 Final Sentence만 타겟으로 사용
        """
        if self.df is None:
            print("오류: 데이터가 로드되지 않았습니다.")
            return

        print(f"데이터 처리 및 파일 저장 중...")
        processed_data = []
        error_count = 0 

        for _, row in self.df.iterrows():
            base_input_str = self.process_row(row)
            
            if base_input_str is None:
                error_count += 1
                continue

            annotations = row.get('sentence_annotations')
            if isinstance(annotations, str):
                annotations = literal_eval(annotations)
            
            if not annotations:
                continue

            for anno in annotations:
                final_target = anno.get('final_sentence', '').strip()
                
                # [수정] 태그 없이, 증강 없이, 오직 정답 문장만 저장
                if final_target:
                    processed_data.append({
                        "id": str(row.get("example_id")),
                        "input": base_input_str, # 태그 없음
                        "target": final_target
                    })

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=4)
        
        print(f"▶ 저장 완료: {output_path}")
        print(f"▶ 원본 데이터: {len(self.df)} -> 최종 생성: {len(processed_data)} (오류 제외: {error_count})")

if __name__ == "__main__":
    TRAIN_INPUT = "totto_data/totto_train_data.jsonl"
    DEV_INPUT = "totto_data/totto_dev_data.jsonl"
    
    # 1. Dev 처리
    processor_dev = ToTToIntegratedProcessor(DEV_INPUT)
    processor_dev.load_and_filter(quantile_cutoff=None)
    processor_dev.save_to_json("train_data/totto_dev_standard.json")

    # 2. Train 처리
    processor_train = ToTToIntegratedProcessor(TRAIN_INPUT)
    processor_train.load_and_filter(quantile_cutoff=0.99)
    processor_train.print_sample()
    processor_train.save_to_json("train_data/totto_train_standard.json")