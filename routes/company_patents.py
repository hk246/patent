"""Tab 1: 自社特許登録"""
import os
import io
import pandas as pd
from flask import (Blueprint, render_template, request, redirect,
                   url_for, flash, g, send_file)
from config import DATA_DIR

bp = Blueprint('company_patents', __name__, url_prefix='/company-patents')


def load_csv_from_stream(stream, fallback_encoding='cp932'):
    """CSV読み込み（UTF-8/CP932自動判定）"""
    try:
        return pd.read_csv(stream, encoding='utf-8')
    except Exception:
        stream.seek(0)
        return pd.read_csv(stream, encoding=fallback_encoding)


def df_to_csv_bytes(df):
    return df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')


@bp.route('/')
def index():
    cp = g.state.get_company_patents()
    company_vectors = g.state.get('company_vectors')

    # サンプルデータの存在確認
    sample_path = os.path.join(DATA_DIR, 'sample_company_patents.csv')
    sample_exists = os.path.exists(sample_path)
    sample_preview = None
    if sample_exists:
        sample_df = pd.read_csv(sample_path, encoding='utf-8')
        sample_preview = sample_df[['patent_id', 'title']].to_dict('records')

    return render_template('company_patents.html',
                           cp=cp,
                           cp_list=cp.to_dict('records') if not cp.empty else [],
                           company_vectors=company_vectors,
                           sample_exists=sample_exists,
                           sample_preview=sample_preview)


@bp.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('csv_file')
    if not file or file.filename == '':
        flash('ファイルを選択してください。', 'error')
        return redirect(url_for('company_patents.index'))

    try:
        df = load_csv_from_stream(file.stream)
        if 'patent_id' not in df.columns:
            flash("'patent_id' 列が必要です。", 'error')
            return redirect(url_for('company_patents.index'))

        for col in ['title', 'abstract', 'claims', 'description']:
            if col not in df.columns:
                df[col] = ''

        g.state.set('company_patents', df.reset_index(drop=True))
        g.state.delete('company_vectors')
        flash(f'{len(df)}件を登録しました', 'success')
    except Exception as e:
        flash(f'読み込みエラー: {e}', 'error')

    return redirect(url_for('company_patents.index'))


@bp.route('/load-sample', methods=['POST'])
def load_sample():
    sample_path = os.path.join(DATA_DIR, 'sample_company_patents.csv')
    if not os.path.exists(sample_path):
        flash('サンプルデータが見つかりません。', 'error')
        return redirect(url_for('company_patents.index'))

    df = pd.read_csv(sample_path, encoding='utf-8')
    g.state.set('company_patents', df)
    g.state.delete('company_vectors')
    flash(f'{len(df)}件のサンプル自社特許を読み込みました', 'success')
    return redirect(url_for('company_patents.index'))


@bp.route('/delete/<int:idx>', methods=['POST'])
def delete(idx):
    cp = g.state.get_company_patents()
    if not cp.empty and 0 <= idx < len(cp):
        cp = cp.drop(index=idx).reset_index(drop=True)
        g.state.set('company_patents', cp)
        g.state.delete('company_vectors')
        flash('特許を削除しました', 'success')
    return redirect(url_for('company_patents.index'))


@bp.route('/download')
def download():
    cp = g.state.get_company_patents()
    if cp.empty:
        flash('ダウンロードするデータがありません。', 'warning')
        return redirect(url_for('company_patents.index'))

    buf = io.BytesIO(df_to_csv_bytes(cp))
    return send_file(buf, mimetype='text/csv', as_attachment=True,
                     download_name='company_patents.csv')


@bp.route('/download-template')
def download_template():
    template = pd.DataFrame({
        'patent_id': ['JP2020-001001', 'JP2020-001002'],
        'title': ['発明のタイトル1', '発明のタイトル2'],
        'abstract': ['要約テキスト', '要約テキスト'],
        'claims': ['【請求項1】...', '【請求項1】...'],
        'description': ['実施例テキスト', '実施例テキスト'],
    })
    buf = io.BytesIO(df_to_csv_bytes(template))
    return send_file(buf, mimetype='text/csv', as_attachment=True,
                     download_name='company_patents_template.csv')
