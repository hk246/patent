"""Tab 5: 可視化"""
import sys
import os
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio

from flask import Blueprint, render_template, request, jsonify, g

from vectorizer import reduce_dimensions

bp = Blueprint('visualization', __name__, url_prefix='/visualization')


def fig_to_html(fig, chart_id=None):
    return pio.to_html(fig, full_html=False, include_plotlyjs=False,
                       div_id=chart_id, config={'displayModeBar': True, 'locale': 'ja'})


@bp.route('/')
def index():
    state = g.state
    db = state.get_patent_db()
    has_db_vectors = state.has('db_vectors')
    cp = state.get_company_patents()
    has_company_vectors = state.has('company_vectors')

    ready = db is not None and has_db_vectors
    has_results = state.has('search_results')

    vec = state.get('vectorizer')
    has_top_features = vec is not None and hasattr(vec, 'get_top_features')

    patent_ids = cp['patent_id'].tolist() if (cp is not None and not cp.empty) else []

    return render_template('visualization.html',
                           ready=ready,
                           has_results=has_results,
                           has_top_features=has_top_features,
                           patent_ids=patent_ids)


@bp.route('/scatter', methods=['POST'])
def scatter():
    """散布図生成（AJAX）"""
    try:
        state = g.state
        data = request.get_json() or {}

        db = state.get_patent_db()
        db_vecs = state.get('db_vectors')
        cp = state.get_company_patents()
        company_vecs = state.get('company_vectors')

        if db is None or db_vecs is None:
            return jsonify({'error': 'ベクトル化を実行してください。'})

        dim_method = data.get('dim_method', 'SVD')
        color_by = data.get('color_by', 'type')
        show_labels = data.get('show_labels', False)

        method_map = {'SVD (LSA)': 'SVD', 'PCA': 'PCA', 'SVD': 'SVD'}
        method = method_map.get(dim_method, 'SVD')

        all_vecs = db_vecs.copy()
        all_df = db.copy()
        all_df['kind'] = '候補特許'

        if cp is not None and not cp.empty and company_vecs is not None:
            all_vecs = np.vstack([db_vecs, company_vecs])
            cp_copy = cp.copy()
            cp_copy['kind'] = '自社特許'
            all_df = pd.concat([all_df, cp_copy], ignore_index=True)

        reduced = reduce_dimensions(all_vecs, n_components=2, method=method)
        all_df['x'] = reduced[:, 0]
        all_df['y'] = reduced[:, 1]

        results = state.get('search_results')
        if results is not None and 'similarity_score' in results.columns:
            all_df = all_df.merge(results[['patent_id', 'similarity_score']],
                                  on='patent_id', how='left')

        color_col, color_map = None, None
        if color_by == 'type' and 'kind' in all_df.columns:
            color_col = 'kind'
            color_map = {'自社特許': '#f59e0b', '候補特許': '#3b82f6'}
        elif color_by == 'label' and 'label' in all_df.columns:
            all_df['label_str'] = all_df['label'].apply(
                lambda x: '正解(1)' if x == 1 else ('ノイズ(0)' if x == 0 else '未ラベル'))
            color_col = 'label_str'
            color_map = {'正解(1)': '#10b981', 'ノイズ(0)': '#ef4444', '未ラベル': '#6b7280'}
        elif color_by == 'score' and 'similarity_score' in all_df.columns:
            color_col = 'similarity_score'

        hover = ['patent_id', 'title']
        for c in ['label', 'similarity_score', 'kind']:
            if c in all_df.columns:
                hover.append(c)

        fig = px.scatter(all_df, x='x', y='y',
                         color=color_col,
                         color_discrete_map=color_map if isinstance(color_map, dict) else None,
                         color_continuous_scale='RdYlGn' if color_col == 'similarity_score' else None,
                         hover_data=hover,
                         text='patent_id' if show_labels else None,
                         symbol='kind' if 'kind' in all_df.columns else None,
                         symbol_map={'自社特許': 'star', '候補特許': 'circle'},
                         title=f'特許文書散布図（{dim_method}）',
                         labels={'x': f'{dim_method}成分1', 'y': f'{dim_method}成分2'})

        if show_labels:
            fig.update_traces(textposition='top center', textfont_size=8)
        fig.update_layout(height=600)

        return jsonify({
            'status': 'ok',
            'charts': {'scatter': fig_to_html(fig, 'chart-scatter')},
        })

    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()})


@bp.route('/histogram', methods=['POST'])
def histogram():
    """類似度スコアヒストグラム（AJAX）"""
    try:
        state = g.state
        results = state.get('search_results')
        if results is None or 'similarity_score' not in results.columns:
            return jsonify({'error': '先にクリアランス調査を実行してください。'})

        fig_h = px.histogram(results, x='similarity_score',
                             color='label' if 'label' in results.columns else None,
                             color_discrete_map={1: '#10b981', 0: '#ef4444'},
                             title='類似度スコアのヒストグラム', nbins=20,
                             barmode='overlay', opacity=0.7)
        fig_h.update_layout(height=400)

        return jsonify({
            'status': 'ok',
            'charts': {'histogram': fig_to_html(fig_h, 'chart-histogram')},
        })

    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()})


@bp.route('/features', methods=['POST'])
def features():
    """TF-IDF重要語（AJAX）"""
    try:
        state = g.state
        data = request.get_json() or {}
        patent_id = data.get('patent_id', '')

        cp = state.get_company_patents()
        vec = state.get('vectorizer')
        preprocessor = state.get('preprocessor')

        if cp is None or cp.empty:
            return jsonify({'error': '自社特許を登録してください。'})
        if vec is None or not hasattr(vec, 'get_top_features'):
            return jsonify({'error': 'TF-IDF手法でベクトル化してください。'})

        q_row = cp[cp['patent_id'] == patent_id]
        if q_row.empty:
            return jsonify({'error': f'特許 {patent_id} が見つかりません。'})
        q_row = q_row.iloc[0]

        if preprocessor is None:
            from preprocessor import JapanesePreprocessor
            preprocessor = JapanesePreprocessor(
                pos_filter=['名詞'], min_length=2,
                use_stop_words=True, use_patent_stop_words=True)

        q_text = preprocessor.preprocess_patent_sections(
            title=str(q_row.get('title', '') or ''),
            abstract=str(q_row.get('abstract', '') or ''),
            claims=str(q_row.get('claims', '') or ''),
            description=str(q_row.get('description', '') or ''),
        )

        feats = vec.get_top_features(q_text, top_n=30)
        if not feats:
            return jsonify({'error': '特徴量を取得できませんでした。'})

        fd = pd.DataFrame(feats, columns=['語', 'TF-IDFスコア'])
        fig_f = px.bar(fd, x='TF-IDFスコア', y='語', orientation='h',
                       color='TF-IDFスコア', color_continuous_scale='Blues',
                       title=f'{patent_id} の重要語 Top30')
        fig_f.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500)

        return jsonify({
            'status': 'ok',
            'charts': {'features': fig_to_html(fig_f, 'chart-features')},
        })

    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()})
