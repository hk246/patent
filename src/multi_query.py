"""
複数クエリ（自社特許）による類似度計算モジュール

【自社特許が複数ある場合の戦略】

1. 最大スコア (Max) ← クリアランス調査に推奨
   各候補特許について「全自社特許との類似度の最大値」を取る
   → 1件でも自社特許に近ければ要注意と判断できる（保守的・安全側）

2. 平均スコア (Mean)
   全自社特許との類似度の平均値
   → 自社ポートフォリオ全体への近さを示す

3. 結合ベクトル (Combined)
   全自社特許のベクトルを平均してから類似度計算
   → 自社技術の「重心」からの距離を見る

4. 個別スコア (Individual)
   各自社特許に対する類似度を個別に表示
   → 最も詳細・透明性が高い
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity


def compute_multi_query_similarity(
    query_vectors: np.ndarray,
    candidate_vectors: np.ndarray,
    strategy: str = 'max',
) -> np.ndarray:
    """
    複数クエリベクトルと候補ベクトル群の類似度を計算する

    Args:
        query_vectors:     自社特許ベクトル行列 shape=(n_queries, n_features)
        candidate_vectors: 候補特許ベクトル行列 shape=(n_candidates, n_features)
        strategy:          集約戦略 'max' | 'mean' | 'combined'

    Returns:
        候補特許ごとの類似度スコア shape=(n_candidates,)
    """
    if query_vectors.ndim == 1:
        query_vectors = query_vectors.reshape(1, -1)

    if strategy == 'combined':
        # 全クエリベクトルを平均してから類似度計算
        combined = np.mean(query_vectors, axis=0, keepdims=True)
        scores = cosine_similarity(combined, candidate_vectors)[0]
        return scores

    # 全クエリ × 全候補 の類似度行列 shape=(n_queries, n_candidates)
    sim_matrix = cosine_similarity(query_vectors, candidate_vectors)

    if strategy == 'max':
        return np.max(sim_matrix, axis=0)
    elif strategy == 'mean':
        return np.mean(sim_matrix, axis=0)
    else:
        return np.max(sim_matrix, axis=0)


def compute_individual_query_similarities(
    query_vectors: np.ndarray,
    candidate_vectors: np.ndarray,
    query_ids: List[str],
) -> pd.DataFrame:
    """
    各自社特許に対する候補特許の類似度を個別に計算する

    Returns:
        columns = query_id ごとの類似度スコア
        index = candidate インデックス
    """
    if query_vectors.ndim == 1:
        query_vectors = query_vectors.reshape(1, -1)

    sim_matrix = cosine_similarity(query_vectors, candidate_vectors)
    # shape: (n_queries, n_candidates) → transposeして (n_candidates, n_queries)
    df = pd.DataFrame(
        sim_matrix.T,
        columns=[f"score_{qid}" for qid in query_ids],
    )
    return df


def rank_candidates_multi_query(
    patent_df: pd.DataFrame,
    query_vectors: np.ndarray,
    candidate_vectors: np.ndarray,
    query_patents_df: pd.DataFrame,
    strategy: str = 'max',
    top_n: int = 50,
    score_threshold: float = 0.0,
) -> pd.DataFrame:
    """
    複数クエリで候補特許をランキングする

    Args:
        patent_df:       候補特許DataFrame
        query_vectors:   自社特許ベクトル (n_queries, n_features)
        candidate_vectors: 候補特許ベクトル (n_candidates, n_features)
        query_patents_df: 自社特許DataFrame (patent_id列が必要)
        strategy:        集約戦略
        top_n:           上位N件
        score_threshold: 最低スコア閾値

    Returns:
        類似度スコア付き候補特許DataFrame（降順ソート）
        追加列: similarity_score, max_similar_query_id, individual scores
    """
    if query_vectors.ndim == 1:
        query_vectors = query_vectors.reshape(1, -1)

    query_ids = query_patents_df['patent_id'].tolist()

    # 集約スコアを計算
    scores = compute_multi_query_similarity(query_vectors, candidate_vectors, strategy)

    # 個別スコア
    individual_df = compute_individual_query_similarities(
        query_vectors, candidate_vectors, query_ids
    )

    # 最も類似した自社特許を特定（max戦略用）
    if query_vectors.shape[0] > 1:
        sim_matrix = cosine_similarity(query_vectors, candidate_vectors)
        most_similar_query_idx = np.argmax(sim_matrix, axis=0)
        most_similar_query_ids = [query_ids[i] for i in most_similar_query_idx]
    else:
        most_similar_query_ids = [query_ids[0]] * len(scores)

    result_df = patent_df.copy().reset_index(drop=True)
    result_df['similarity_score'] = scores
    result_df['most_similar_query'] = most_similar_query_ids

    # 個別スコアを追加
    for col in individual_df.columns:
        result_df[col] = individual_df[col].values

    # フィルタ & ソート
    result_df = result_df[result_df['similarity_score'] >= score_threshold]
    result_df = result_df.sort_values('similarity_score', ascending=False)

    return result_df.head(top_n).reset_index(drop=True)


def summarize_multi_query_results(
    results_df: pd.DataFrame,
    query_patents_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    自社特許ごとに「最も類似した候補特許」のサマリーを作成

    Returns:
        自社特許ごとの最高類似度・上位候補の一覧
    """
    query_ids = query_patents_df['patent_id'].tolist()
    summary_rows = []

    for qid in query_ids:
        col = f"score_{qid}"
        q_info = query_patents_df[query_patents_df['patent_id'] == qid].iloc[0]

        if col in results_df.columns:
            top_result = results_df.loc[results_df[col].idxmax()]
            max_score = results_df[col].max()
            above_05 = int((results_df[col] >= 0.5).sum())
            above_03 = int((results_df[col] >= 0.3).sum())
        else:
            max_score = 0.0
            above_05 = 0
            above_03 = 0
            top_result = pd.Series()

        summary_rows.append({
            '自社特許番号': qid,
            '自社特許タイトル': str(q_info.get('title', ''))[:50],
            '最高類似度スコア': round(max_score, 4),
            '類似度≥0.5の候補数': above_05,
            '類似度≥0.3の候補数': above_03,
            '最類似候補番号': str(top_result.get('patent_id', '')) if not top_result.empty else '',
            '最類似候補タイトル': str(top_result.get('title', ''))[:50] if not top_result.empty else '',
        })

    return pd.DataFrame(summary_rows)


STRATEGY_DESCRIPTIONS = {
    'max': '最大スコア（推奨: いずれかの自社特許に類似したら要注意）',
    'mean': '平均スコア（自社ポートフォリオ全体への近さ）',
    'combined': '結合ベクトル（自社技術の重心からの距離）',
}
