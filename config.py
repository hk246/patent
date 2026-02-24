"""Flask アプリケーション設定"""
import os
import secrets

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SECRET_KEY = os.environ.get('FLASK_SECRET_KEY', secrets.token_hex(32))
SESSION_DATA_DIR = os.path.join(BASE_DIR, 'session_data')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB

DEFAULT_SETTINGS = {
    'vectorizer_method': 'TF-IDF',
    'max_features': 10000,
    'use_lsa': False,
    'lsa_dims': 100,
    'use_stop_words': True,
    'use_claim_weight': True,
    'multi_query_strategy': 'max',
}
