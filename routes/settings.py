"""設定保存 + ベクトル化実行"""
import sys
import os
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from flask import Blueprint, request, session, jsonify, g
import pandas as pd
import numpy as np

from config import DEFAULT_SETTINGS
from preprocessor import JapanesePreprocessor
from vectorizer import VectorizerFactory

bp = Blueprint('settings', __name__, url_prefix='/settings')


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


@bp.route('/update', methods=['POST'])
def update():
    """設定値の保存（AJAX）"""
    data = request.get_json() or request.form.to_dict()
    settings = session.get('settings', DEFAULT_SETTINGS.copy())
    for key in DEFAULT_SETTINGS:
        if key in data:
            val = data[key]
            if key in ('max_features', 'lsa_dims'):
                val = int(val)
            elif key in ('use_lsa', 'use_stop_words', 'use_claim_weight'):
                val = val in (True, 'true', '1', 'on')
            else:
                val = str(val)
            settings[key] = val
    session['settings'] = settings
    return jsonify({'status': 'ok', 'settings': settings})


@bp.route('/vectorize', methods=['POST'])
def vectorize():
    """ベクトル化実行（AJAX）— app.py run_vectorize() の移植"""
    try:
        state = g.state
        settings = session.get('settings', DEFAULT_SETTINGS.copy())

        cp = state.get_company_patents()
        db = state.get_patent_db()

        if (cp is None or cp.empty) and db is None:
            return jsonify({'status': 'error',
                            'error': '自社特許・候補DBのいずれかを読み込んでください。'})

        use_stop_words = settings.get('use_stop_words', True)
        use_claim_weight = settings.get('use_claim_weight', True)
        vectorizer_method = settings.get('vectorizer_method', 'TF-IDF')
        max_features = int(settings.get('max_features', 10000))
        use_lsa = settings.get('use_lsa', False)
        lsa_dims = int(settings.get('lsa_dims', 100))

        preprocessor = get_preprocessor(use_stop_words)
        state.set('preprocessor', preprocessor)

        weights = {'title': 3, 'abstract': 2, 'claims': 5, 'description': 1} if use_claim_weight else None

        company_texts = preprocess_rows(cp, preprocessor, weights) if (cp is not None and not cp.empty) else []
        db_texts = preprocess_rows(db, preprocessor, weights) if db is not None else []

        state.set('company_preprocessed', company_texts)
        state.set('preprocessed_db', db_texts)

        all_texts = company_texts + db_texts
        if not all_texts:
            return jsonify({'status': 'error', 'error': '前処理後のテキストが空です。'})

        kwargs = {'max_features': max_features}
        if use_lsa:
            kwargs['lsa_components'] = lsa_dims

        vec = VectorizerFactory.create(vectorizer_method, **kwargs)
        vec.fit(all_texts)
        state.set('vectorizer', vec)

        company_count = 0
        db_count = 0

        if company_texts:
            company_vectors = vec.transform(company_texts)
            state.set('company_vectors', company_vectors)
            company_count = len(company_texts)

        if db_texts:
            db_vectors = vec.transform(db_texts)
            state.set('db_vectors', db_vectors)
            db_count = len(db_texts)

        return jsonify({
            'status': 'ok',
            'company_count': company_count,
            'db_count': db_count,
        })

    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e),
                        'traceback': traceback.format_exc()})
