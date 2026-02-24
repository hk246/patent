"""Tab 4: 分類器学習"""
import sys
import os
import io
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio

from flask import Blueprint, render_template, request, jsonify, g, send_file

from classifier import (
    PatentClassifier, compare_classifiers,
    get_algorithm_list, get_algorithm_description,
)

bp = Blueprint('classifier', __name__, url_prefix='/classifier')


def fig_to_html(fig, chart_id=None):
    return pio.to_html(fig, full_html=False, include_plotlyjs=False,
                       div_id=chart_id, config={'displayModeBar': True, 'locale': 'ja'})


@bp.route('/')
def index():
    state = g.state
    db = state.get_patent_db()
    has_db_vectors = state.has('db_vectors')

    ready = db is not None and has_db_vectors
    has_labels = ready and 'label' in db.columns

    algorithms = get_algorithm_list()
    algo_descriptions = {a: get_algorithm_description(a) for a in algorithms}

    label_info = None
    if has_labels:
        labeled_mask = db['label'].notna() & (db['label'].astype(str).str.strip() != '')
        labeled_df = db[labeled_mask].copy()
        labeled_df['label'] = labeled_df['label'].astype(int)
        label_info = {
            'labeled': len(labeled_df),
            'positive': int((labeled_df['label'] == 1).sum()),
            'negative': int((labeled_df['label'] == 0).sum()),
            'unlabeled': len(db) - len(labeled_df),
        }

    has_classifier = state.has('classifier')

    return render_template('classifier.html',
                           ready=ready,
                           has_labels=has_labels,
                           algorithms=algorithms,
                           algo_descriptions=algo_descriptions,
                           label_info=label_info,
                           has_classifier=has_classifier)


@bp.route('/train', methods=['POST'])
def train():
    """分類器学習（AJAX）"""
    try:
        state = g.state
        data = request.get_json() or {}

        db = state.get_patent_db()
        db_vecs = state.get('db_vectors')

        if db is None or db_vecs is None:
            return jsonify({'error': '候補特許DBをベクトル化してください。'})
        if 'label' not in db.columns:
            return jsonify({'error': 'label列が必要です。'})

        algorithm = data.get('algorithm', 'ランダムフォレスト')
        test_ratio = float(data.get('test_ratio', 0.2))
        cv_folds = int(data.get('cv_folds', 5))

        labeled_mask = db['label'].notna() & (db['label'].astype(str).str.strip() != '')
        labeled_df = db[labeled_mask].copy()
        labeled_df['label'] = labeled_df['label'].astype(int)

        if len(labeled_df) < 4:
            return jsonify({'error': 'ラベル付きデータが少なすぎます（最低4件）。'})

        from sklearn.model_selection import train_test_split
        labeled_pos = [db.index.tolist().index(i) for i in labeled_df.index]
        X_l = db_vecs[labeled_pos]
        y_l = labeled_df['label'].values

        X_tr, X_te, y_tr, y_te = train_test_split(
            X_l, y_l, test_size=test_ratio,
            stratify=y_l if len(np.unique(y_l)) > 1 else None,
            random_state=42)

        clf = PatentClassifier(algorithm=algorithm)
        clf.fit(X_tr, y_tr)
        state.set('classifier', clf)

        ev = clf.evaluate(X_te, y_te)
        report = ev['classification_report']
        cv = clf.cross_validate(X_l, y_l, cv=cv_folds)

        resp = {
            'status': 'ok',
            'metrics': {
                'accuracy': f"{ev['accuracy']:.4f}",
                'precision': f"{report.get('1', {}).get('precision', 0):.4f}",
                'recall': f"{report.get('1', {}).get('recall', 0):.4f}",
                'cv_f1': f"{cv['mean']:.4f}\u00b1{cv['std']:.4f}",
            },
            'charts': {},
        }

        # Confusion matrix
        cm = np.array(ev['confusion_matrix'])
        fig_cm = px.imshow(cm, text_auto=True,
                           x=['ノイズ(0)', '正解(1)'], y=['ノイズ(0)', '正解(1)'],
                           title='混同行列', color_continuous_scale='Blues')
        fig_cm.update_layout(height=350)
        resp['charts']['confusion_matrix'] = fig_to_html(fig_cm, 'chart-cm')

        return jsonify(resp)

    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()})


@bp.route('/compare', methods=['POST'])
def compare():
    """アルゴリズム比較（AJAX）"""
    try:
        state = g.state
        data = request.get_json() or {}

        db = state.get_patent_db()
        db_vecs = state.get('db_vectors')

        if db is None or db_vecs is None:
            return jsonify({'error': '候補特許DBをベクトル化してください。'})

        sel_algos = data.get('algorithms', [])
        test_ratio = float(data.get('test_ratio', 0.2))
        cv_folds = int(data.get('cv_folds', 5))

        if len(sel_algos) < 2:
            return jsonify({'error': '2件以上のアルゴリズムを選択してください。'})

        labeled_mask = db['label'].notna() & (db['label'].astype(str).str.strip() != '')
        labeled_df = db[labeled_mask].copy()
        labeled_df['label'] = labeled_df['label'].astype(int)

        from sklearn.model_selection import train_test_split
        labeled_pos = [db.index.tolist().index(i) for i in labeled_df.index]
        X_l = db_vecs[labeled_pos]
        y_l = labeled_df['label'].values

        X_tr, X_te, y_tr, y_te = train_test_split(
            X_l, y_l, test_size=test_ratio,
            stratify=y_l if len(np.unique(y_l)) > 1 else None,
            random_state=42)

        cmp_df = compare_classifiers(X_tr, y_tr, X_te, y_te,
                                     algorithms=sel_algos, cv=cv_folds)

        resp = {
            'status': 'ok',
            'tables': {
                'comparison': cmp_df.to_html(index=False,
                    classes='w-full text-sm text-gray-300', border=0,
                    float_format=lambda x: f'{x:.4f}' if isinstance(x, float) else x),
            },
            'charts': {},
        }

        if 'F1スコア(1)' in cmp_df.columns:
            # F1スコア昇順でソート（横棒グラフは下→上に表示されるため）
            plot_df = cmp_df.sort_values('F1スコア(1)', ascending=True)
            fig = px.bar(plot_df, x='F1スコア(1)', y='分類器',
                         orientation='h',
                         color='F1スコア(1)', color_continuous_scale='RdYlGn',
                         title='分類器別 F1スコア比較', text_auto='.4f')
            n_classifiers = len(plot_df)
            chart_height = max(400, n_classifiers * 40 + 120)
            fig.update_layout(
                height=chart_height,
                yaxis=dict(
                    tickfont=dict(size=12),
                    automargin=True,
                ),
                xaxis=dict(
                    title='F1スコア(1)',
                    range=[0, 1.05],
                ),
                margin=dict(l=10),
            )
            resp['charts']['f1_comparison'] = fig_to_html(fig, 'chart-f1')

        return jsonify(resp)

    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()})


@bp.route('/predict', methods=['POST'])
def predict():
    """未ラベル候補特許の予測（AJAX）"""
    try:
        state = g.state
        db = state.get_patent_db()
        db_vecs = state.get('db_vectors')
        clf = state.get('classifier')

        if clf is None:
            return jsonify({'error': '先に分類器を学習してください。'})

        labeled_mask = db['label'].notna() & (db['label'].astype(str).str.strip() != '')
        unlabeled_df = db[~labeled_mask].copy()

        if len(unlabeled_df) == 0:
            return jsonify({'error': '未ラベルの特許がありません。'})

        ul_pos = [db.index.tolist().index(i) for i in unlabeled_df.index]
        X_ul = db_vecs[ul_pos]

        preds = clf.predict(X_ul)
        scores = clf.predict_relevance_score(X_ul)

        pred_df = unlabeled_df.copy()
        pred_df['predicted_label'] = preds
        pred_df['relevance_score'] = scores
        pred_df = pred_df.sort_values('relevance_score', ascending=False)

        state.set('prediction_results', pred_df)

        disp_df = pred_df[['patent_id', 'title', 'predicted_label', 'relevance_score']]

        resp = {
            'status': 'ok',
            'positive_count': int((preds == 1).sum()),
            'negative_count': int((preds == 0).sum()),
            'tables': {
                'predictions': disp_df.to_html(index=False,
                    classes='w-full text-sm text-gray-300', border=0,
                    float_format=lambda x: f'{x:.4f}' if isinstance(x, float) else x),
            },
        }

        return jsonify(resp)

    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()})


@bp.route('/download-predictions')
def download_predictions():
    pred_df = g.state.get('prediction_results')
    if pred_df is None:
        return jsonify({'error': 'ダウンロードする予測結果がありません。'})

    csv_bytes = pred_df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
    buf = io.BytesIO(csv_bytes)
    return send_file(buf, mimetype='text/csv', as_attachment=True,
                     download_name='prediction_results.csv')
