"""
特許クリアランス調査システム v1.01 — Flask版

起動:
  start.bat  (推奨)
  または: python flask_app.py
"""
import sys
import os
import uuid
import socket
import webbrowser
import threading

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


def find_free_port(start=5000, end=5099):
    """空きポートを探す（start〜endの範囲）"""
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('127.0.0.1', port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"ポート {start}〜{end} がすべて使用中です")


def open_browser(port):
    """サーバー起動後にブラウザを自動で開く"""
    import time
    time.sleep(1.5)
    webbrowser.open(f'http://127.0.0.1:{port}/')


if __name__ == '__main__':
    app = create_app()

    # ポート決定: 環境変数 → 自動検出
    env_port = os.environ.get('PORT')
    if env_port:
        port = int(env_port)
    else:
        port = find_free_port()

    # ブラウザ自動起動（--no-browser で無効化可能）
    if '--no-browser' not in sys.argv:
        threading.Thread(target=open_browser, args=(port,), daemon=True).start()

    print(f"\n  特許クリアランス調査システム v1.01")
    print(f"  http://127.0.0.1:{port}/")
    print(f"  終了するには Ctrl+C を押してください\n")

    app.run(debug=True, port=port, use_reloader=False)
