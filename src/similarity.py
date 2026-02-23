"""
特許類似度計算モジュール
論文参考: コサイン類似度によるランキング、段落（文）単位の類似度計算
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity


def compute_cosine_similarity(
    query_vector: np.ndarray,
    candidate_vectors: np.ndarray,
) -> np.ndarray:
    """
    クエリベクトルと候補ベクトル群のコサイン類似度を計算

    Args:
        query_vector: クエリベクトル (1D or 2D)
        candidate_vectors: 候補ベクトル行列 (n_candidates, n_features)

    Returns:
        類似度スコア配列 (n_candidates,)
    """
    query_vector = np.atleast_2d(query_vector)
    if query_vector.ndim > 2:
        query_vector = query_vector.reshape(1, -1)

    scores = cosine_similarity(query_vector, candidate_vectors)[0]
    return scores


def rank_patents_by_similarity(
    patent_df: pd.DataFrame,
    scores: np.ndarray,
    top_n: int = 50,
    score_threshold: float = 0.0,
) -> pd.DataFrame:
    """
    類似度スコアで特許をランキング

    Args:
        patent_df: 特許データフレーム
        scores: 各特許の類似度スコア
        top_n: 上位N件を返す
        score_threshold: この値以上のスコアのみ返す

    Returns:
        類似度スコア付きの特許データフレーム（降順ソート）
    """
    result_df = patent_df.copy()
    result_df['similarity_score'] = scores
    result_df = result_df[result_df['similarity_score'] >= score_threshold]
    result_df = result_df.sort_values('similarity_score', ascending=False)
    return result_df.head(top_n).reset_index(drop=True)


def compute_passage_similarities(
    query_text_preprocessed: str,
    patent_sentences: List[Dict],
    query_vector: np.ndarray,
    sentence_vectors: np.ndarray,
) -> List[Dict]:
    """
    段落（文）単位の類似度計算
    論文参考: パッセージ検索 - 特許の各文に対してスコアを付与

    Args:
        query_text_preprocessed: クエリの前処理済みテキスト
        patent_sentences: 特許文のリスト（各要素: {'patent_id', 'section', 'sent_id', 'text'}）
        query_vector: クエリベクトル
        sentence_vectors: 各文のベクトル

    Returns:
        類似度スコア付き文リスト（降順ソート）
    """
    if len(patent_sentences) == 0:
        return []

    scores = compute_cosine_similarity(query_vector, sentence_vectors)

    results = []
    for i, (sent_info, score) in enumerate(zip(patent_sentences, scores)):
        results.append({
            **sent_info,
            'similarity_score': float(score),
        })

    results.sort(key=lambda x: x['similarity_score'], reverse=True)
    return results


def compute_claim_element_similarities(
    claim_elements: List[Dict],
    element_vectors: np.ndarray,
    candidate_sentences: List[Dict],
    sentence_vectors: np.ndarray,
) -> pd.DataFrame:
    """
    構成要素単位の類似度計算
    論文参考: 発明の構成要素毎の根拠箇所（文）抽出

    各構成要素に対して最も類似した候補特許の文を抽出する

    Returns:
        構成要素ごとの最類似文テーブル
    """
    results = []

    for elem, elem_vec in zip(claim_elements, element_vectors):
        if len(candidate_sentences) == 0:
            continue

        scores = compute_cosine_similarity(elem_vec, sentence_vectors)
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        best_sent = candidate_sentences[best_idx]

        results.append({
            '構成要素ID': elem.get('element_id', '?'),
            '請求項番号': elem.get('claim_no', '?'),
            '構成要素テキスト': elem.get('text', ''),
            '根拠文': best_sent.get('text', ''),
            '根拠箇所': f"{best_sent.get('patent_id', '')}_{best_sent.get('section', '')}{best_sent.get('sent_id', '')}",
            '類似度スコア': round(best_score, 4),
        })

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    return df


def calculate_precision_recall(
    ranked_results: pd.DataFrame,
    label_column: str = 'label',
    score_column: str = 'similarity_score',
    top_ks: List[int] = None,
) -> pd.DataFrame:
    """
    精度（Precision）・再現率（Recall）を計算
    論文参考: クリアランス調査での評価指標

    Args:
        ranked_results: ランキング済みの結果データフレーム
        label_column: 正解ラベルの列名 (1=正解, 0=ノイズ)
        score_column: スコアの列名
        top_ks: 評価するK値のリスト

    Returns:
        各K値でのPrecision/Recallテーブル
    """
    if label_column not in ranked_results.columns:
        return pd.DataFrame({'エラー': ['ラベル列が見つかりません']})

    if top_ks is None:
        n = len(ranked_results)
        top_ks = [10, 20, 30, 50, 100, n]
        top_ks = [k for k in top_ks if k <= n]

    labels = ranked_results[label_column].values
    total_positives = int(np.sum(labels == 1))

    if total_positives == 0:
        return pd.DataFrame({'エラー': ['正解ラベル(1)が存在しません']})

    rows = []
    for k in top_ks:
        top_labels = labels[:k]
        tp = int(np.sum(top_labels == 1))
        precision = tp / k if k > 0 else 0
        recall = tp / total_positives if total_positives > 0 else 0
        rows.append({
            '確認数 (K)': k,
            '精度 (Precision)': round(precision, 4),
            '再現率 (Recall)': round(recall, 4),
            'F1スコア': round(2 * precision * recall / (precision + recall + 1e-9), 4),
            '発見した正解数': tp,
            '正解総数': total_positives,
        })

    return pd.DataFrame(rows)


def build_sentence_index(
    patent_df: pd.DataFrame,
    preprocessor,
    sections: List[str] = None,
) -> Tuple[List[Dict], List[str]]:
    """
    特許データから文単位のインデックスを構築
    論文参考: 「PatNo_記載部略号_文番号」形式のタグ付け

    Args:
        patent_df: 特許データフレーム
        preprocessor: JapanesePreprocessorインスタンス
        sections: 処理するセクション ('title', 'abstract', 'claims', 'description')

    Returns:
        (文情報リスト, 前処理済み文リスト)
    """
    if sections is None:
        sections = ['title', 'abstract', 'claims', 'description']

    section_codes = {
        'title': 'T',
        'abstract': 'A',
        'claims': 'C',
        'description': 'E',
    }

    sentence_infos = []
    preprocessed_sentences = []

    for _, row in patent_df.iterrows():
        patent_id = str(row.get('patent_id', ''))

        for section in sections:
            if section not in row or not row[section]:
                continue

            text = str(row[section])
            sentences = preprocessor.split_into_sentences(text)
            sec_code = section_codes.get(section, section[0].upper())

            for sent_idx, sent in enumerate(sentences, 1):
                preprocessed = preprocessor.preprocess(sent)
                if not preprocessed:
                    continue

                sentence_infos.append({
                    'patent_id': patent_id,
                    'section': sec_code,
                    'sent_id': sent_idx,
                    'tag': f"{patent_id}_{sec_code}{sent_idx}",
                    'text': sent,
                })
                preprocessed_sentences.append(preprocessed)

    return sentence_infos, preprocessed_sentences
