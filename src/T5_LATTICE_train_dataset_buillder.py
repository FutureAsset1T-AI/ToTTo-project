import json
import pandas as pd
import os
from ast import literal_eval
from typing import List, Dict, Any, Optional, Tuple

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

    def _get_hierarchical_headers(self, table: List[List[Dict]], row_idx: int, col_idx: int, mode: str) -> str:
        """계층적 헤더 정보를 추출합니다."""
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
        
        res = " | ".join(dict.fromkeys(headers))
        return res if res else "None"

    def linearize_row(self, row: pd.Series) -> Optional[Tuple[str, List[List[int]]]]:
        """
        한 행의 데이터를 선형화하고, 사용된 셀의 좌표 리스트를 함께 반환합니다.
        
        Returns:
            Tuple[str, List[List[int]]]: (선형화된 텍스트, 대응하는 [row, col] 좌표 리스트)
        """
        table = row['table']
        page_title = str(row.get('table_page_title', 'None'))
        sec_title = str(row.get('table_section_title', 'None'))
        sec_text = row.get('table_section_text', 'None')
        
        if pd.isna(sec_text) or not sec_text:
            sec_text = "None"
        else:
            sec_text = str(sec_text)

        linearized = f"[PAGE] {page_title} [SEC] {sec_title} [TEXT] {sec_text}"

        highlighted = row['highlighted_cells']
        if isinstance(highlighted, str):
            highlighted = literal_eval(highlighted)

        # [수정됨] 행 우선(Row-Major) 순서로 정렬하여 텍스트 생성 순서 일관성 보장
        # 모델이 위에서 아래로, 왼쪽에서 오른쪽으로 읽는 흐름을 학습하도록 유도
        highlighted.sort(key=lambda x: (x[0], x[1]))

        # [수정됨] 좌표 추적을 위한 리스트 초기화
        used_coords = []

        # 1차 검증: 인덱스 범위 확인
        for r_idx, c_idx in highlighted:
            if r_idx >= len(table) or c_idx >= len(table[r_idx]):
                return None 

        # 데이터 생성 및 좌표 수집
        for r_idx, c_idx in highlighted:
            cell = table[r_idx][c_idx]
            val = str(cell['value'])
            is_hdr_bool = cell.get('is_header', False)
            is_hdr_str = "T" if is_hdr_bool else "F"

            if is_hdr_bool:
                r_head, c_head = "None", "None"
            else:
                r_head = self._get_hierarchical_headers(table, r_idx, c_idx, "row")
                c_head = self._get_hierarchical_headers(table, r_idx, c_idx, "col")

            # [H] 태그 제거 버전 포맷
            cell_info = f" [CELL] {val} [TYPE] {is_hdr_str} [R_HEAD] {r_head} [C_HEAD] {c_head}"
            linearized += cell_info
            
            # [수정됨] 현재 셀의 좌표 저장 (LATTICE Bias 생성용)
            used_coords.append([r_idx, c_idx])

        return linearized, used_coords

    def print_sample(self):
        """변환된 샘플을 출력하여 포맷을 확인합니다."""
        if self.df is None or self.df.empty:
            return

        for _, row in self.df.iterrows():
            result = self.linearize_row(row)
            
            # [수정됨] 튜플 언패킹 처리
            if result is not None:
                input_text, cell_coords = result
                annotations = row.get('sentence_annotations', [])
                target_text = annotations[0].get('final_sentence', 'N/A') if annotations else 'N/A'

                print("\n" + "="*80)
                print(" [PREPROCESSED SAMPLE CHECK] ")
                print("="*80)
                print(f"▶ INPUT (Linearized):\n{input_text}")
                print("-" * 80)
                print(f"▶ COORDS (For Lattice):\n{cell_coords}")
                print("-" * 80)
                print(f"▶ TARGET (Final Sentence):\n{target_text}")
                print("="*80 + "\n")
                return

    def save_to_json(self, output_path: str, is_train: bool = False):
        """
        최종 학습/검증용 JSON 생성
        [수정됨] cell_coords 필드를 포함하여 저장하도록 변경
        """
        if self.df is None:
            print(f"오류: 데이터가 로드되지 않았습니다. save_to_json을 중단합니다.")
            return

        print(f"선형화 진행 및 파일 저장 중... (Mode: {'TRAIN' if is_train else 'VALIDATION'})")
        processed_data = []
        error_count = 0 
        skip_count = 0 

        PREFIX_CLEAN = "[CLEAN] "
        PREFIX_DRAFT = "[DRAFT] "

        for _, row in self.df.iterrows():
            # [수정됨] 선형화 텍스트와 좌표 리스트를 함께 받아옴
            result = self.linearize_row(row)
            
            if result is None:
                error_count += 1
                continue
            
            base_linearized_text, cell_coords = result

            annotations = row.get('sentence_annotations')
            if isinstance(annotations, str):
                annotations = literal_eval(annotations)
            
            if not annotations:
                continue

            for anno in annotations:
                # (1) Final Sentence 처리
                final_target = anno.get('final_sentence', '').strip()
                
                if final_target:
                    processed_data.append({
                        "id": str(row.get("example_id")),
                        "input": PREFIX_CLEAN + base_linearized_text,
                        "target": final_target,
                        "cell_coords": cell_coords  # [수정됨] 좌표 정보 저장
                    })

                # (2) Ambiguity Sentence 처리 (Train 모드)
                if is_train:
                    ambiguity_target = anno.get('sentence_after_ambiguity', '').strip()
                    
                    if ambiguity_target:
                        if ambiguity_target != final_target:
                            processed_data.append({
                                "id": str(row.get("example_id")) + "_aug",
                                "input": PREFIX_DRAFT + base_linearized_text,
                                "target": ambiguity_target,
                                "cell_coords": cell_coords  # [수정됨] 좌표 정보 저장
                            })
                        else:
                            skip_count += 1

        # 파일 저장
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=4)
        
        print("\n" + "="*50)
        print(" [PROCESSING COMPLETE] ")
        print("="*50)
        print(f"▶ 모드 설정     : {'TRAIN (Multi-Task Augmentation)' if is_train else 'VALIDATION (Clean Only)'}")
        print(f"▶ 원본 데이터 수: {len(self.df)} 건")
        print(f"▶ 좌표 오류 제외: {error_count} 건")
        print(f"▶ 중복 증강 제외: {skip_count} 건 (Ambiguity == Final)")
        print(f"▶ 최종 생성 쌍  : {len(processed_data)} 건")
        print(f"▶ 저장 경로     : {output_path}")
        print("="*50)

if __name__ == "__main__":
    # 경로 설정
    TRAIN_INPUT = "totto_data/totto_train_data.jsonl"
    DEV_INPUT = "totto_data/totto_dev_data.jsonl"
    
    # 1. Dev 처리
    processor_dev = ToTToIntegratedProcessor(DEV_INPUT)
    processor_dev.load_and_filter(quantile_cutoff=None)
    processor_dev.save_to_json("train_data/totto_dev_LATICE.json", is_train=False)

    # 2. Train 처리
    processor_train = ToTToIntegratedProcessor(TRAIN_INPUT)
    processor_train.load_and_filter(quantile_cutoff=0.99)
    processor_train.print_sample()
    processor_train.save_to_json("train_data/totto_train_LATICE.json", is_train=True)