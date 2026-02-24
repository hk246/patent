"""Tab 3: クリアランス調査"""
import sys
import os
import io
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio

from flask import Blueprint, render_template, request, jsonify, g, session, send_file

from multi_query import (
    rank_candidates_multi_query,
    summarize_multi_query_results,
    STRATEGY_DESCRIPTIONS,
)
from similarity import (
    calculate_precision_recall,
    build_sentence_index,
    compute_claim_element_similarities,
)
from config import DEFAULT_SETTINGS

bp = Blueprint('clearance', __name__, url_prefix='/clearance')


def fig_to_html(fig, chart_id=None):
    return pio.to_html(fig, full_html=False, include_plotlyjs=False,
                       div_id=chart_id, config={'displayModeBar': True, 'locale': 'ja'})


@bp.route('/')
def index():
    state = g.state
    cp = state.get_company_patents()
    db = state.get_patent_db()
    has_company_vectors = state.has('company_vectors')
    has_db_vectors = state.has('db_vectors')

    ready = (cp is not None and not cp.empty and has_company_vectors
             and db is not None and has_db_vectors)

    patent_ids = cp['patent_id'].tolist() if (cp is not None and not cp.empty) else []
    settings = session.get('settings', DEFAULT_SETTINGS.copy())
    current_strategy = settings.get('multi_query_strategy', 'max')

    return render_template('clearance.html',
                           ready=ready,
                           patent_ids=patent_ids,
                           current_strategy=current_strategy,
                           strategy_desc=STRATEGY_DESCRIPTIONS.get(current_strategy, ''))


@bp.route('/run', methods=['POST'])
def run():
    """クリアランス調査実行（AJAX）"""
    try:
        state = g.state
        data = request.get_json() or {}

        cp = state.get_company_patents()
        db = state.get_patent_db()
        company_vecs = state.get('company_vectors')
        db_vecs = state.get('db_vectors')

        if cp is None or cp.empty or company_vecs is None:
            return jsonify({'error': '自社特許をベクトル化してください。'})
        if db is None or db_vecs is None:
            return jsonify({'error': '候補特許DBをベクトル化してください。'})

        selected_ids = data.get('selected_ids', cp['patent_id'].tolist())
        top_n = int(data.get('top_n', 30))
        score_threshold = float(data.get('score_threshold', 0.0))

        settings = session.get('settings', DEFAULT_SETTINGS.copy())
        current_strategy = settings.get('multi_query_strategy', 'max')

        selected_mask = cp['patent_id'].isin(selected_ids)
        selected_cp = cp[selected_mask].reset_index(drop=True)
        selected_vecs = company_vecs[selected_mask.values]

        results = rank_candidates_multi_query(
            patent_df=db,
            query_vectors=selected_vecs,
            candidate_vectors=db_vecs,
            query_patents_df=selected_cp,
            strategy=current_strategy,
            top_n=top_n,
            score_threshold=score_threshold,
        )

        state.set('search_results', results)

        # Build response
        resp = {'status': 'ok', 'count': len(results), 'charts': {}, 'tables': {}}

        # Summary table (multi-query)
        if len(selected_cp) > 1:
            summary_df = summarize_multi_query_results(results, selected_cp)
            resp['tables']['summary'] = summary_df.to_html(
                index=False, classes='w-full text-sm text-gray-300', border=0)

        # Precision/Recall
        if 'label' in results.columns:
            pr_df = calculate_precision_recall(results)
            if not pr_df.empty and 'エラー' not in pr_df.columns:
                resp['tables']['precision_recall'] = pr_df.to_html(
                    index=False, classes='w-full text-sm text-gray-300', border=0)

        # Score bar chart
        plot_results = results.reset_index()
        fig_bar = px.bar(
            plot_results, x='index', y='similarity_score',
            color='label' if 'label' in results.columns else None,
            color_discrete_map={1: '#10b981', 0: '#ef4444'},
            hover_data=['patent_id', 'title'] +
                       (['most_similar_query'] if 'most_similar_query' in results.columns else []),
            title=f'類似度スコア分布（集約: {STRATEGY_DESCRIPTIONS[current_strategy]}）',
            labels={'index': '順位', 'similarity_score': 'コサイン類似度'},
        )
        fig_bar.update_layout(height=400)
        resp['charts']['bar'] = fig_to_html(fig_bar, 'chart-bar')

        # Heatmap
        score_cols = [c for c in results.columns if c.startswith('score_')]
        if len(score_cols) > 1:
            heat = results[['patent_id'] + score_cols].head(30).set_index('patent_id')
            heat.columns = [c.replace('score_', '') for c in heat.columns]
            fig_h = px.imshow(
                heat, color_continuous_scale='RdYlGn', aspect='auto',
                title='候補特許(縦) x 自社特許(横) 類似度スコア',
                labels=dict(x='自社特許', y='候補特許', color='類似度'),
            )
            fig_h.update_layout(height=500)
            resp['charts']['heatmap'] = fig_to_html(fig_h, 'chart-heatmap')

        # Results table
        disp_cols = ['patent_id', 'title', 'similarity_score']
        if 'most_similar_query' in results.columns:
            disp_cols.append('most_similar_query')
        if 'label' in results.columns:
            disp_cols.append('label')
        resp['tables']['results'] = results[disp_cols].to_html(
            index=False, classes='w-full text-sm text-gray-300', border=0,
            float_format=lambda x: f'{x:.4f}' if isinstance(x, float) else x)

        # Top 10 detail cards
        detail_cards = []
        for _, row in results.head(10).iterrows():
            card = {
                'patent_id': str(row.get('patent_id', '')),
                'title': str(row.get('title', ''))[:60],
                'score': f"{row['similarity_score']:.4f}",
                'claims': str(row.get('claims', ''))[:400],
            }
            if 'most_similar_query' in row:
                card['most_similar_query'] = str(row['most_similar_query'])
            if 'label' in row and pd.notna(row['label']):
                card['label'] = int(row['label'])
            # Individual scores
            ind = {c.replace('score_', ''): round(row[c], 4)
                   for c in score_cols if c in row.index}
            if ind:
                card['individual_scores'] = ind
            detail_cards.append(card)
        resp['detail_cards'] = detail_cards

        return jsonify(resp)

    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()})


@bp.route('/claim-elements', methods=['POST'])
def claim_elements():
    """構成要素分析（AJAX）"""
    try:
        state = g.state
        data = request.get_json() or {}
        query_patent_id = data.get('patent_id', '')

        cp = state.get_company_patents()
        results = state.get('search_results')
        preprocessor = state.get('preprocessor')
        vec = state.get('vectorizer')

        if cp is None or cp.empty:
            return jsonify({'error': '自社特許が登録されていません。'})
        if results is None or results.empty:
            return jsonify({'error': '先にクリアランス調査を実行してください。'})
        if vec is None:
            return jsonify({'error': 'ベクトル化が必要です。'})

        q_row = cp[cp['patent_id'] == query_patent_id]
        if q_row.empty:
            return jsonify({'error': f'特許 {query_patent_id} が見つかりません。'})
        q_row = q_row.iloc[0]

        if preprocessor is None:
            from preprocessor import JapanesePreprocessor
            preprocessor = JapanesePreprocessor(
                pos_filter=['名詞'], min_length=2,
                use_stop_words=True, use_patent_stop_words=True)

        elements = preprocessor.extract_claim_elements(str(q_row.get('claims', '') or ''))
        if not elements:
            return jsonify({'error': '請求項から構成要素を抽出できませんでした。'})

        elem_vecs = vec.transform([e['preprocessed'] for e in elements])
        sent_infos, sent_texts = build_sentence_index(results.head(10), preprocessor)

        resp = {'status': 'ok', 'element_count': len(elements)}

        # Elements table
        elem_df = pd.DataFrame(elements)[['element_id', 'claim_no', 'text']]
        resp['tables'] = {
            'elements': elem_df.to_html(index=False, classes='w-full text-sm text-gray-300', border=0)
        }

        if sent_texts:
            sent_vecs = vec.transform(sent_texts)
            elem_result = compute_claim_element_similarities(
                elements, elem_vecs, sent_infos, sent_vecs)
            if not elem_result.empty:
                resp['tables']['analysis'] = elem_result.to_html(
                    index=False, classes='w-full text-sm text-gray-300', border=0)

        return jsonify(resp)

    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()})


@bp.route('/download-results')
def download_results():
    results = g.state.get('search_results')
    if results is None or results.empty:
        return jsonify({'error': 'ダウンロードする結果がありません。'})

    csv_bytes = results.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
    buf = io.BytesIO(csv_bytes)
    return send_file(buf, mimetype='text/csv', as_attachment=True,
                     download_name='clearance_results.csv')
