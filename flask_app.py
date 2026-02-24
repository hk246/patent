"""
特許クリアランス調査システム v2.1 — Flask版
論文参考: 文字ベクトル化と機械学習を用いた効率的な特許調査
(tokugikon 2018.11.26 no.291 安藤俊幸)

起動:
  pip install -r requirements.txt
  python flask_app.py
"""
import sys
import os
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from flask import Flask, g, session, redirect, url_for
import plotly.io as pio

from config import SECRET_KEY, MAX_CONTENT_LENGTH, UPLOAD_FOLDER, DEFAULT_SETTINGS
from state_manager import SessionStateManager
from routes import register_blueprints


def create_app():
    app = Flask(__name__)
    app.secret_key = SECRET_KEY
    app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

    # Plotly ダークテンプレート
    dark_template = {
        'layout': {
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'plot_bgcolor': 'rgba(17,24,39,1)',
            'font': {'color': '#d1d5db'},
            'xaxis': {'gridcolor': '#374151', 'zerolinecolor': '#374151'},
            'yaxis': {'gridcolor': '#374151', 'zerolinecolor': '#374151'},
            'colorway': ['#6366f1', '#10b981', '#f59e0b', '#ef4444',
                         '#8b5cf6', '#ec4899', '#14b8a6', '#f97316'],
        }
    }
    pio.templates['patent_dark'] = dark_template
    pio.templates.default = 'patent_dark'

    # セッション状態の初期化
    @app.before_request
    def load_state_manager():
        if 'session_id' not in session:
            session['session_id'] = str(uuid.uuid4())
        if 'settings' not in session:
            session['settings'] = DEFAULT_SETTINGS.copy()
        g.state = SessionStateManager(session['session_id'])

    # テンプレートにsettingsを自動注入
    @app.context_processor
    def inject_globals():
        from vectorizer import VectorizerFactory
        from multi_query import STRATEGY_DESCRIPTIONS
        settings = session.get('settings', DEFAULT_SETTINGS.copy())
        state = getattr(g, 'state', None)

        company_count = 0
        db_count = 0
        has_company_vectors = False
        has_db_vectors = False

        if state:
            cp = state.get_company_patents()
            if cp is not None and not cp.empty:
                company_count = len(cp)
            db = state.get_patent_db()
            if db is not None:
                db_count = len(db)
            has_company_vectors = state.has('company_vectors')
            has_db_vectors = state.has('db_vectors')

        return {
            'settings': settings,
            'vectorizer_methods': VectorizerFactory.METHODS,
            'strategy_descriptions': STRATEGY_DESCRIPTIONS,
            'company_count': company_count,
            'db_count': db_count,
            'has_company_vectors': has_company_vectors,
            'has_db_vectors': has_db_vectors,
        }

    # ルート登録
    register_blueprints(app)

    # トップページリダイレクト
    @app.route('/')
    def index():
        return redirect(url_for('company_patents.index'))

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, port=int(os.environ.get('PORT', 5000)))
