"""
ベクトル化手法の比較ベンチマークモジュール
各ベクトル化手法×集約戦略の組み合わせで検索精度と分類器精度を比較する
"""

import numpy as np
import pandas as pd
import time
from typing import List, Dict, Optional, Callable

from sklearn.model_selection import train_test_split

from vectorizer import VectorizerFactory
from similarity import calculate_precision_recall
from multi_query import rank_candidates_multi_query
from classifier import compare_classifiers, get_algorithm_list


def _vectorize_once(method, company_texts, db_texts, kwargs):
    """手法ごとに1回だけベクトル化する（戦略ループの外で使う）"""
    t0 = time.time()
    vec = VectorizerFactory.create(method, **kwargs)
    all_texts = company_texts + db_texts
    vec.fit(all_texts)
    company_vecs = vec.transform(company_texts)
    db_vecs = vec.transform(db_texts)
    vectorize_time = time.time() - t0
    dim = company_vecs.shape[1] if company_vecs.ndim > 1 else 1
    return company_vecs, db_vecs, vectorize_time, dim


def benchmark_similarity_with_vectors(
    method: str,
    strategy: str,
    company_vecs,
    db_vecs,
    db_df: pd.DataFrame,
    company_df: pd.DataFrame,
    vectorize_time: float,
    vector_dim: int,
) -> Dict:
    """既にベクトル化済みのデータで類似度検索を実行し、Precision/Recall指標を返す。"""
    results = rank_candidates_multi_query(
        patent_df=db_df,
        query_vectors=company_vecs,
        candidate_vectors=db_vecs,
        query_patents_df=company_df,
        strategy=strategy,
        top_n=len(db_df),
        score_threshold=0.0,
    )

    pr_df = pd.DataFrame()
    if 'label' in results.columns:
        pr_df = calculate_precision_recall(results)

    return {
        'method': method,
        'strategy': strategy,
        'vectorize_time': round(vectorize_time, 4),
        'precision_recall': pr_df,
        'ranked_results': results,
        'vector_dim': vector_dim,
    }


def benchmark_classifier_with_vectors(
    method: str,
    db_vecs,
    db_df: pd.DataFrame,
    vectorize_time: float,
    test_ratio: float = 0.2,
    cv_folds: int = 5,
    classifier_algorithms: Optional[List[str]] = None,
) -> Dict:
    """既にベクトル化済みのデータで分類器比較を実行する。"""
    if 'label' not in db_df.columns:
        return {'method': method, 'error': 'label列が必要です'}

    labeled_mask = db_df['label'].notna() & (db_df['label'].astype(str).str.strip() != '')
    labeled_df = db_df[labeled_mask].copy()
    labeled_df['label'] = labeled_df['label'].astype(int)

    if len(labeled_df) < 4:
        return {
            'method': method,
            'vectorize_time': round(vectorize_time, 4),
            'error': 'ラベル付きデータが少なすぎます（最低4件）',
        }

    labeled_pos = [db_df.index.tolist().index(i) for i in labeled_df.index]
    X_l = db_vecs[labeled_pos]
    y_l = labeled_df['label'].values

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_l, y_l, test_size=test_ratio,
        stratify=y_l if len(np.unique(y_l)) > 1 else None,
        random_state=42,
    )

    if classifier_algorithms is None:
        classifier_algorithms = [
            'エイダブースト', 'ランダムフォレスト', 'SVM (RBFカーネル)',
            'ロジスティック回帰', 'ニューラルネット (MLP)',
        ]

    cmp_df = compare_classifiers(
        X_tr, y_tr, X_te, y_te,
        algorithms=classifier_algorithms, cv=cv_folds,
    )

    best_row = cmp_df.loc[cmp_df['F1スコア(1)'].idxmax()]
    return {
        'method': method,
        'vectorize_time': round(vectorize_time, 4),
        'best_classifier': best_row['分類器'],
        'best_f1': float(best_row['F1スコア(1)']),
        'best_accuracy': float(best_row['正解率']),
        'best_auc': float(best_row.get('AUC-ROC', 0)),
        'comparison_df': cmp_df,
    }


def run_full_benchmark(
    methods: List[str],
    strategies: List[str],
    company_texts: List[str],
    db_texts: List[str],
    db_df: pd.DataFrame,
    company_df: pd.DataFrame,
    test_ratio: float = 0.2,
    cv_folds: int = 5,
    classifier_algorithms: Optional[List[str]] = None,
    vectorizer_kwargs: Optional[Dict] = None,
    progress_callback: Optional[Callable] = None,
) -> Dict:
    """
    複数ベクトル化手法×複数集約戦略を一括比較する。

    ベクトル化は手法ごとに1回だけ実行し、各戦略で類似度検索を回す。
    分類器比較は戦略に依存しないため手法ごとに1回だけ。

    progress_callback(current_index, total, description) で進捗通知。
    """
    t_start = time.time()
    kwargs = dict(vectorizer_kwargs or {})

    # 手法×戦略の全組み合わせ数
    total_steps = len(methods) * len(strategies)
    step = 0

    has_labels = 'label' in db_df.columns

    # 結果格納
    sim_results = {}     # key: "method|strategy"
    clf_results = {}     # key: method
    summary_rows = []

    for method in methods:
        # ─── ベクトル化（手法ごとに1回） ───
        company_vecs, db_vecs, vec_time, vec_dim = _vectorize_once(
            method, company_texts, db_texts, kwargs)

        # ─── 分類器比較（手法ごとに1回、戦略非依存） ───
        clf = {}
        if has_labels:
            clf = benchmark_classifier_with_vectors(
                method, db_vecs, db_df,
                vectorize_time=vec_time,
                test_ratio=test_ratio, cv_folds=cv_folds,
                classifier_algorithms=classifier_algorithms,
            )
        clf_results[method] = clf

        # ─── 各戦略で類似度検索 ───
        for strategy in strategies:
            combo_key = f"{method}|{strategy}"

            if progress_callback:
                progress_callback(step, total_steps, f"{method} × {strategy}")
            step += 1

            sim = benchmark_similarity_with_vectors(
                method, strategy, company_vecs, db_vecs,
                db_df, company_df, vec_time, vec_dim,
            )
            sim_results[combo_key] = sim

            # サマリー行（手法×戦略ごとに1行）
            strat_label = {'max': '最大スコア', 'mean': '平均スコア',
                           'combined': '結合ベクトル'}.get(strategy, strategy)
            row = {
                'ベクトル化手法': method,
                '集約戦略': strat_label,
                'ベクトル次元': vec_dim,
                'ベクトル化時間(秒)': round(vec_time, 4),
            }

            pr = sim.get('precision_recall')
            if pr is not None and not pr.empty and 'エラー' not in pr.columns:
                for _, pr_row in pr.iterrows():
                    k = int(pr_row['確認数 (K)'])
                    row[f'P@{k}'] = round(float(pr_row['精度 (Precision)']), 4)
                    row[f'R@{k}'] = round(float(pr_row['再現率 (Recall)']), 4)
                    row[f'F1@{k}'] = round(float(pr_row['F1スコア']), 4)

            if clf and 'error' not in clf:
                row['最良分類器'] = clf.get('best_classifier', '')
                row['最良F1'] = round(clf.get('best_f1', 0), 4)
                row['最良正解率'] = round(clf.get('best_accuracy', 0), 4)
                row['最良AUC'] = round(clf.get('best_auc', 0), 4)

            summary_rows.append(row)

    if progress_callback:
        progress_callback(total_steps, total_steps, '完了')

    return {
        'similarity_results': sim_results,
        'classifier_results': clf_results,
        'summary_df': pd.DataFrame(summary_rows),
        'total_time': round(time.time() - t_start, 2),
        'strategies': strategies,
        'methods': methods,
    }
