"""Tab: ベクトル化手法比較ベンチマーク"""
import sys
import os
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from flask import Blueprint, render_template, request, jsonify, g, session

from vectorizer import VectorizerFactory
from classifier import get_algorithm_list
from config import DEFAULT_SETTINGS
from benchmark import run_full_benchmark
from preprocessor import JapanesePreprocessor

bp = Blueprint('benchmark', __name__, url_prefix='/benchmark')


def fig_to_html(fig, chart_id=None):
    return pio.to_html(fig, full_html=False, include_plotlyjs=False,
                       div_id=chart_id, config={'displayModeBar': True, 'locale': 'ja'})


def get_preprocessor(use_stop_words=True):
    return JapanesePreprocessor(
        pos_filter=['名詞'], min_length=2,
        use_stop_words=use_stop_words, use_patent_stop_words=True,
    )


def preprocess_rows(df, preprocessor, weights=None):
    out = []
    for _, row in df.iterrows():
        t = preprocessor.preprocess_patent_sections(
            title=str(row.get('title', '') or ''),
            abstract=str(row.get('abstract', '') or ''),
            claims=str(row.get('claims', '') or ''),
            description=str(row.get('description', '') or ''),
            weights=weights,
        )
        out.append(t)
    return out


@bp.route('/')
def index():
    state = g.state
    cp = state.get_company_patents()
    db = state.get_patent_db()
    has_company = cp is not None and not cp.empty
    has_db = db is not None
    has_labels = has_db and 'label' in db.columns

    methods = list(VectorizerFactory.METHODS.keys())
    gensim_ok = VectorizerFactory.is_gensim_available()
    algorithms = get_algorithm_list()

    return render_template('benchmark.html',
                           ready=has_company and has_db,
                           has_labels=has_labels,
                           methods=methods,
                           gensim_ok=gensim_ok,
                           algorithms=algorithms)


@bp.route('/run', methods=['POST'])
def run():
    """ベンチマーク実行（AJAX）"""
    try:
        state = g.state
        data = request.get_json() or {}

        cp = state.get_company_patents()
        db = state.get_patent_db()

        if cp is None or cp.empty or db is None:
            return jsonify({'error': 'データが未登録です。'})

        methods = data.get('methods', ['TF-IDF', 'TF-IDF + LSA'])
        strategy = data.get('strategy', 'max')
        test_ratio = float(data.get('test_ratio', 0.2))
        cv_folds = int(data.get('cv_folds', 5))
        classifiers = data.get('classifiers', None)

        settings = session.get('settings', DEFAULT_SETTINGS.copy())
        use_stop_words = settings.get('use_stop_words', True)
        use_claim_weight = settings.get('use_claim_weight', True)
        max_features = int(settings.get('max_features', 10000))

        preprocessor = get_preprocessor(use_stop_words)
        weights = {'title': 3, 'abstract': 2, 'claims': 5, 'description': 1} if use_claim_weight else None
        company_texts = preprocess_rows(cp, preprocessor, weights)
        db_texts = preprocess_rows(db, preprocessor, weights)

        result = run_full_benchmark(
            methods=methods,
            company_texts=company_texts,
            db_texts=db_texts,
            db_df=db,
            company_df=cp,
            strategy=strategy,
            test_ratio=test_ratio,
            cv_folds=cv_folds,
            classifier_algorithms=classifiers if classifiers else None,
            vectorizer_kwargs={'max_features': max_features},
        )

        summary = result['summary_df']
        resp = {
            'status': 'ok',
            'total_time': result['total_time'],
            'tables': {},
            'charts': {},
        }

        # Summary table
        resp['tables']['summary'] = summary.to_html(
            index=False, classes='w-full text-sm text-gray-300', border=0,
            float_format=lambda x: f'{x:.4f}' if isinstance(x, float) else str(x))

        # Precision@K grouped bar
        p_cols = [c for c in summary.columns if c.startswith('P@')]
        if p_cols:
            p_data = summary.melt(id_vars='ベクトル化手法', value_vars=p_cols,
                                   var_name='K', value_name='Precision')
            fig_p = px.bar(p_data, x='K', y='Precision', color='ベクトル化手法',
                          barmode='group', title='Precision@K 比較', text_auto='.3f')
            fig_p.update_layout(yaxis_range=[0, 1.05], height=400)
            resp['charts']['precision'] = fig_to_html(fig_p, 'chart-precision')

        # Recall@K grouped bar
        r_cols = [c for c in summary.columns if c.startswith('R@')]
        if r_cols:
            r_data = summary.melt(id_vars='ベクトル化手法', value_vars=r_cols,
                                   var_name='K', value_name='Recall')
            fig_r = px.bar(r_data, x='K', y='Recall', color='ベクトル化手法',
                          barmode='group', title='Recall@K 比較', text_auto='.3f')
            fig_r.update_layout(yaxis_range=[0, 1.05], height=400)
            resp['charts']['recall'] = fig_to_html(fig_r, 'chart-recall')

        # Best F1 per method
        if '最良F1' in summary.columns:
            fig_f1 = px.bar(summary.sort_values('最良F1', ascending=True),
                            x='最良F1', y='ベクトル化手法', orientation='h',
                            color='最良F1', color_continuous_scale='RdYlGn',
                            text='最良分類器', title='ベクトル化手法別 最良分類器F1')
            n = len(summary)
            fig_f1.update_layout(xaxis_range=[0, 1.05], height=max(300, n * 60 + 120),
                                 yaxis=dict(automargin=True))
            resp['charts']['f1'] = fig_to_html(fig_f1, 'chart-f1')

        # Time comparison
        fig_t = px.bar(summary, x='ベクトル化手法', y='ベクトル化時間(秒)',
                       color='ベクトル化手法', title='ベクトル化所要時間')
        resp['charts']['time'] = fig_to_html(fig_t, 'chart-time')

        # Radar chart
        radar_fig = go.Figure()
        max_time = summary['ベクトル化時間(秒)'].max()
        for _, row in summary.iterrows():
            metrics = {}
            if 'R@20' in row: metrics['Recall@20'] = row['R@20']
            elif 'R@10' in row: metrics['Recall@10'] = row['R@10']
            if 'P@20' in row: metrics['Precision@20'] = row['P@20']
            elif 'P@10' in row: metrics['Precision@10'] = row['P@10']
            if '最良F1' in row.index and pd.notna(row.get('最良F1')): metrics['分類器F1'] = row['最良F1']
            if '最良AUC' in row.index and pd.notna(row.get('最良AUC')): metrics['AUC'] = row['最良AUC']
            metrics['速度'] = 1 - (row['ベクトル化時間(秒)'] / max_time) if max_time > 0 else 1.0
            if metrics:
                cats = list(metrics.keys())
                vals = list(metrics.values()) + [list(metrics.values())[0]]
                radar_fig.add_trace(go.Scatterpolar(
                    r=vals, theta=cats + [cats[0]],
                    fill='toself', name=row['ベクトル化手法'], opacity=0.6,
                ))
        radar_fig.update_layout(polar=dict(radialaxis=dict(range=[0, 1])),
                                 title='ベクトル化手法 総合比較', height=500)
        resp['charts']['radar'] = fig_to_html(radar_fig, 'chart-radar')

        # Per-method classifier detail tables
        detail_tables = {}
        for method, clf_res in result['classifier_results'].items():
            cmp_df = clf_res.get('comparison_df')
            if cmp_df is not None and not cmp_df.empty:
                detail_tables[method] = cmp_df.to_html(
                    index=False, classes='w-full text-sm text-gray-300', border=0,
                    float_format=lambda x: f'{x:.4f}' if isinstance(x, float) else str(x))
        resp['tables']['details'] = detail_tables

        return jsonify(resp)

    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()})
