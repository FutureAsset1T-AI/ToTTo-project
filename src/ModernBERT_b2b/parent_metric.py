# parent_metric.py
import collections
from collections import Counter

def get_ngrams(segment, max_order):
    """텍스트에서 1~max_order까지의 n-gram을 추출하여 카운트합니다."""
    ngram_counts = collections.Counter()
    for order in range(1, max_order + 1):
        for i in range(0, len(segment) - order + 1):
            ngram = tuple(segment[i:i+order])
            ngram_counts[ngram] += 1
    return ngram_counts

def parent_score(predictions, references, tables, lambda_weight=0.5):
    """
    PARENT 지표를 계산합니다.
    Args:
        predictions: 생성된 문장 리스트 (List[str])
        references: 정답 문장 리스트의 리스트 (List[List[str]])
        tables: 테이블 데이터 리스트 (List[List[str]]) - 각 테이블은 셀 값(문자열)들의 리스트
        lambda_weight: Precision 계산 시 테이블과 레퍼런스 비중 (기본 0.5)
    Returns:
        precision, recall, f1 (float, float, float)
    """
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    
    # n-gram 최대 길이 (보통 4-gram까지 사용)
    max_order = 4
    smoothing = 1e-13 # 0 나누기 방지

    for pred_text, ref_texts, table_texts in zip(predictions, references, tables):
        # 1. 토큰화 (단순 공백 기준 split)
        pred_tokens = pred_text.strip().split()
        # Reference가 여러 개일 수 있으므로 각각 토큰화
        ref_tokens_list = [ref.strip().split() for ref in ref_texts]
        # 테이블 값들도 토큰화하여 하나의 큰 리스트로 만듦
        table_tokens = []
        for cell_value in table_texts:
            table_tokens.extend(str(cell_value).split())

        # 2. N-gram 추출
        pred_ngrams = get_ngrams(pred_tokens, max_order)
        ref_ngrams_list = [get_ngrams(ref, max_order) for ref in ref_tokens_list]
        table_ngrams = get_ngrams(table_tokens, max_order)

        # --- Precision 계산 ---
        # 분자: 생성된 n-gram이 (테이블 OR 레퍼런스)에 존재하는 확률
        numerator_prec = 0.0
        denominator_prec = sum(pred_ngrams.values()) + smoothing

        for ngram, count in pred_ngrams.items():
            # 테이블에 존재 확률: 1 if exist else 0
            prob_in_table = 1.0 if ngram in table_ngrams else 0.0
            # 레퍼런스에 존재 확률: max(count_in_ref / count_in_pred)
            prob_in_ref = 0.0
            for ref_ngrams in ref_ngrams_list:
                prob_in_ref = max(prob_in_ref, min(1.0, ref_ngrams.get(ngram, 0) / count))
            
            # PARENT Precision 공식: (1-lambda)*P(T) + lambda*P(R)
            prob_entailment = prob_in_table + prob_in_ref - (prob_in_table * prob_in_ref)
            # 또는 가중합: prob_entailment = (prob_in_table * (1 - lambda_weight)) + (prob_in_ref * lambda_weight)
            # 논문 공식(Entailment Probability): 테이블이나 레퍼런스 둘 중 하나라도 있으면 됨을 의미 (Soft OR)
            
            # 여기서는 표준적인 Word Overlap 방식 사용: 
            # w_prob = P(ngram \in T) + P(ngram \in R) * (1 - P(ngram \in T)) 
            # 이는 합집합 확률 P(T U R)과 동일
            w_prob = prob_in_table + prob_in_ref * (1.0 - prob_in_table)
            numerator_prec += count * w_prob

        precision = numerator_prec / denominator_prec

        # --- Recall 계산 ---
        # 분자: (레퍼런스 AND 테이블)에 있는 n-gram이 생성문에 얼마나 있나
        # 분모: (레퍼런스 AND 테이블)에 있는 n-gram의 총 개수
        numerator_rec = 0.0
        denominator_rec = 0.0 + smoothing

        # 다중 레퍼런스 중 가장 점수가 높은 것 선택 (Best Match)
        best_recall = 0.0
        
        for ref_ngrams in ref_ngrams_list:
            curr_num = 0.0
            curr_denom = 0.0 + smoothing
            
            for ngram, count in ref_ngrams.items():
                # 테이블에 있는 n-gram인지 확인
                if ngram in table_ngrams:
                    curr_denom += count
                    # 생성된 문장에도 있는지 확인
                    if ngram in pred_ngrams:
                        curr_num += min(count, pred_ngrams[ngram])
            
            if curr_denom > smoothing:
                best_recall = max(best_recall, curr_num / curr_denom)
            else:
                # 테이블과 겹치는 n-gram이 없는 레퍼런스의 경우 단순 n-gram recall 사용 가능하나, PARENT 정의상 0
                pass
                
        recall = best_recall

        # --- F1 계산 ---
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        total_precision += precision
        total_recall += recall
        total_f1 += f1

    # 평균 반환
    n = len(predictions)
    return (total_precision / n), (total_recall / n), (total_f1 / n)