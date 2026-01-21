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

    def _get_hierarchical_headers_list(self, table: List[List[Dict]], row_idx: int, col_idx: int, mode: str) -> List[Tuple[str, int, int]]:
        """
        계층적 헤더 정보와 그 좌표를 추출합니다.
        Returns: List of (Header_Text, Row_Index, Col_Index)
        """
        headers = []
        
        if mode == "col":
            # 현재 셀(row_idx, col_idx) 위의 모든 행을 탐색
            for r in range(row_idx):
                if r < len(table) and col_idx < len(table[r]):
                    if table[r][col_idx].get("is_header"):
                        # (텍스트, 실제 행 r, 실제 열 col_idx)
                        headers.append((str(table[r][col_idx]["value"]), r, col_idx))
        else:
            # 현재 셀(row_idx, col_idx) 왼쪽의 모든 열을 탐색
            if row_idx < len(table):
                for c in range(col_idx):
                    if c < len(table[row_idx]):
                        if table[row_idx][c].get("is_header"):
                            # (텍스트, 실제 행 row_idx, 실제 열 c)
                            headers.append((str(table[row_idx][c]["value"]), row_idx, c))
        
        # 중복 제거 (내용이 같더라도 위치가 다르면 다른 헤더로 볼 수 있으나, 여기서는 순서 유지하며 텍스트 기준 중복 제거)
        # 단, Lattice 구조 유지를 위해 좌표가 중요하다면 중복 제거에 주의해야 함.
        # ToTTo 베이스라인들은 보통 텍스트 기준 중복제거를 하므로 유지하되, 좌표는 첫 등장 기준을 따름.
        seen = set()
        unique_headers = []
        for h_text, r, c in headers:
            if h_text not in seen:
                seen.add(h_text)
                unique_headers.append((h_text, r, c))
                
        return unique_headers

    def process_row(self, row: pd.Series) -> Optional[Dict[str, Any]]:
        """
        한 행의 데이터를 Lattice T5 포맷(Segments + Coords)으로 변환합니다.
        """
        table = row['table']
        page_title = str(row.get('table_page_title', 'None'))
        sec_title = str(row.get('table_section_title', 'None'))
        
        # 1. 메타데이터 (PAGE, SEC) - 좌표 없음(Dummy: [-1, -1])
        segments = ["[PAGE]", page_title, "[SEC]", sec_title]
        coords = [[-1, -1]] * 4

        highlighted = row['highlighted_cells']
        if isinstance(highlighted, str):
            highlighted = literal_eval(highlighted)

        # 좌표 유효성 검사
        for r_idx, c_idx in highlighted:
            if r_idx >= len(table) or c_idx >= len(table[r_idx]):
                return None 

        # 2. Highlighted Cells 순회
        for r_idx, c_idx in highlighted:
            cell = table[r_idx][c_idx]
            val = str(cell['value'])
            is_hdr_bool = cell.get('is_header', False)
            is_hdr_str = "T" if is_hdr_bool else "F"

            # (1) [CELL] 태그 및 값
            segments.extend(["[CELL]", val, "[TYPE]", is_hdr_str])
            coords.extend([[-1, -1], [r_idx, c_idx], [-1, -1], [-1, -1]]) # 값만 실제 좌표 가짐

            if is_hdr_bool:
                # 헤더 셀 자체가 선택된 경우, 상위 헤더는 없다고 가정 (혹은 필요시 로직 추가)
                segments.extend(["[R_HEAD]", "None", "[C_HEAD]", "None"])
                coords.extend([[-1, -1], [-1, -1], [-1, -1], [-1, -1]])
            else:
                # (2) Row Headers
                r_headers = self._get_hierarchical_headers_list(table, r_idx, c_idx, "row")
                segments.append("[R_HEAD]")
                coords.append([-1, -1])
                
                if not r_headers:
                    segments.append("None")
                    coords.append([-1, -1])
                else:
                    for i, (h_text, h_r, h_c) in enumerate(r_headers):
                        if i > 0:
                            segments.append("|") # 구분자
                            coords.append([-1, -1])
                        segments.append(h_text)
                        coords.append([h_r, h_c]) # 헤더의 실제 좌표 사용

                # (3) Col Headers
                c_headers = self._get_hierarchical_headers_list(table, r_idx, c_idx, "col")
                segments.append("[C_HEAD]")
                coords.append([-1, -1])
                
                if not c_headers:
                    segments.append("None")
                    coords.append([-1, -1])
                else:
                    for i, (h_text, h_r, h_c) in enumerate(c_headers):
                        if i > 0:
                            segments.append("|") # 구분자
                            coords.append([-1, -1])
                        segments.append(h_text)
                        coords.append([h_r, h_c]) # 헤더의 실제 좌표 사용

        # 결과 반환
        return {
            "segments": segments,
            "segment_coords": coords
        }

    def print_sample(self):
        """변환된 샘플을 출력하여 포맷을 확인합니다."""
        if self.df is None or self.df.empty:
            return

        print("\n" + "="*80)
        print(" [LATTICE PREPROCESSED SAMPLE CHECK] ")
        print("="*80)
        
        for _, row in self.df.iterrows():
            processed = self.process_row(row)
            if processed is not None:
                segments = processed["segments"]
                coords = processed["segment_coords"]
                
                # 보기 좋게 텍스트로 합쳐서 출력
                input_str = " ".join(segments)
                
                annotations = row.get('sentence_annotations', [])
                target_text = annotations[0].get('final_sentence', 'N/A') if annotations else 'N/A'

                print(f"▶ INPUT SEGMENTS (Joined):\n{input_str}")
                print("-" * 80)
                print(f"▶ COORDS (First 20):\n{coords[:20]} ...")
                print("-" * 80)
                print(f"▶ TARGET (Final Sentence):\n{target_text}")
                print("="*80 + "\n")
                return

    def save_to_json(self, output_path: str, is_train: bool = False):
        """
        최종 학습/검증용 JSON 생성
        segments와 segment_coords를 포함하여 저장
        """
        if self.df is None:
            print(f"오류: 데이터가 로드되지 않았습니다.")
            return

        print(f"데이터 처리 및 파일 저장 중... (Mode: {'TRAIN' if is_train else 'VALIDATION'})")
        processed_data = []
        error_count = 0 
        skip_count = 0 

        # 태스크 토큰 정의 (세그먼트 단위)
        SEG_CLEAN = ["[CLEAN]"]
        SEG_DRAFT = ["[DRAFT]"]
        COORD_DUMMY = [[-1, -1]] # 태스크 토큰용 좌표

        for _, row in self.df.iterrows():
            # 1. 행 처리 (Lattice 구조 생성)
            processed_row = self.process_row(row)
            
            if processed_row is None:
                error_count += 1
                continue

            base_segments = processed_row["segments"]
            base_coords = processed_row["segment_coords"]
            
            # 입력 확인용 문자열 (디버깅/가독성용)
            input_display_str = " ".join(base_segments)

            annotations = row.get('sentence_annotations')
            if isinstance(annotations, str):
                annotations = literal_eval(annotations)
            
            if not annotations:
                continue

            # 2. Annotation 순회하며 데이터 생성
            for anno in annotations:
                final_target = anno.get('final_sentence', '').strip()
                
                if final_target:
                    # [CLEAN] 모드 데이터 생성
                    processed_data.append({
                        "id": str(row.get("example_id")),
                        "input": "[CLEAN] " + input_display_str,   # 시각적 확인용
                        "target": final_target,
                        # [중요] 실제 학습에 사용될 구조화된 데이터
                        "segments": SEG_CLEAN + base_segments,
                        "segment_coords": COORD_DUMMY + base_coords
                    })

                # [DRAFT] 모드 데이터 생성 (Train Only + Augmentation)
                if is_train:
                    ambiguity_target = anno.get('sentence_after_ambiguity', '').strip()
                    
                    if ambiguity_target:
                        if ambiguity_target != final_target:
                            processed_data.append({
                                "id": str(row.get("example_id")) + "_aug",
                                "input": "[DRAFT] " + input_display_str,
                                "target": ambiguity_target,
                                # [중요] 구조화된 데이터
                                "segments": SEG_DRAFT + base_segments,
                                "segment_coords": COORD_DUMMY + base_coords
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
        print(f"▶ 중복 증강 제외: {skip_count} 건")
        print(f"▶ 최종 생성 쌍  : {len(processed_data)} 건")
        print(f"▶ 저장 경로     : {output_path}")
        print("="*50)

if __name__ == "__main__":
    # [수정] 업로드해주신 원본 파일의 경로 설정 유지
    TRAIN_INPUT = "totto_data/totto_train_data.jsonl"
    DEV_INPUT = "totto_data/totto_dev_data.jsonl"
    
    # 1. Dev 처리
    processor_dev = ToTToIntegratedProcessor(DEV_INPUT)
    processor_dev.load_and_filter(quantile_cutoff=None)
    # 저장 경로 유지: train_data/totto_dev_aggumented.json
    processor_dev.save_to_json("train_data/totto_dev_.LATTIC.json", is_train=False)

    # 2. Train 처리
    processor_train = ToTToIntegratedProcessor(TRAIN_INPUT)
    processor_train.load_and_filter(quantile_cutoff=0.99)
    processor_train.print_sample()
    # 저장 경로 유지: train_data/totto_train_aggumented.json
    processor_train.save_to_json("train_data/totto_train_LATTIC.json", is_train=True)