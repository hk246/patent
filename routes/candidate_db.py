"""Tab 2: 候補特許データベース"""
import os
import io
import pandas as pd
from flask import (Blueprint, render_template, request, redirect,
                   url_for, flash, g, send_file)
from config import DATA_DIR

bp = Blueprint('candidate_db', __name__, url_prefix='/candidate-db')


def load_csv_from_stream(stream, fallback_encoding='cp932'):
    try:
        return pd.read_csv(stream, encoding='utf-8')
    except Exception:
        stream.seek(0)
        return pd.read_csv(stream, encoding=fallback_encoding)


def df_to_csv_bytes(df):
    return df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')


@bp.route('/')
def index():
    db = g.state.get_patent_db()
    db_vectors = g.state.get('db_vectors')

    sample_path = os.path.join(DATA_DIR, 'sample_patents.csv')
    sample_exists = os.path.exists(sample_path)
    sample_preview = None
    label_counts = None

    if sample_exists:
        sample_df = pd.read_csv(sample_path, encoding='utf-8')
        cols = ['patent_id', 'title']
        if 'label' in sample_df.columns:
            cols.append('label')
        sample_preview = sample_df[cols].head(8).to_dict('records')

    db_list = None
    if db is not None:
        disp_cols = ['patent_id', 'title']
        if 'label' in db.columns:
            disp_cols.append('label')
            vc = db['label'].value_counts()
            label_counts = {
                'total': len(db),
                'positive': int(vc.get(1, 0)),
                'negative': int(vc.get(0, 0)),
            }
        db_list = db[disp_cols].to_dict('records')

    return render_template('candidate_db.html',
                           db=db,
                           db_list=db_list,
                           db_vectors=db_vectors,
                           label_counts=label_counts,
                           sample_exists=sample_exists,
                           sample_preview=sample_preview)


@bp.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('csv_file')
    if not file or file.filename == '':
        flash('ファイルを選択してください。', 'error')
        return redirect(url_for('candidate_db.index'))

    try:
        df = load_csv_from_stream(file.stream)
        if 'patent_id' not in df.columns:
            flash("'patent_id' 列が必要です。", 'error')
            return redirect(url_for('candidate_db.index'))

        for col in ['title', 'abstract', 'claims', 'description']:
            if col not in df.columns:
                df[col] = ''

        g.state.set('patent_db', df.reset_index(drop=True))
        g.state.delete('db_vectors')
        g.state.delete('preprocessed_db')
        flash(f'{len(df)}件を登録しました', 'success')
    except Exception as e:
        flash(f'読み込みエラー: {e}', 'error')

    return redirect(url_for('candidate_db.index'))


@bp.route('/load-sample', methods=['POST'])
def load_sample():
    sample_path = os.path.join(DATA_DIR, 'sample_patents.csv')
    if not os.path.exists(sample_path):
        flash('サンプルデータが見つかりません。', 'error')
        return redirect(url_for('candidate_db.index'))

    df = pd.read_csv(sample_path, encoding='utf-8')
    g.state.set('patent_db', df)
    g.state.delete('db_vectors')
    g.state.delete('preprocessed_db')
    flash(f'{len(df)}件のサンプル候補特許を読み込みました', 'success')
    return redirect(url_for('candidate_db.index'))


@bp.route('/download-template')
def download_template():
    template = pd.DataFrame({
        'patent_id': ['JP2020-XXXXX'],
        'title': ['タイトル'],
        'abstract': ['要約'],
        'claims': ['請求項'],
        'description': ['詳細説明'],
        'label': [''],
    })
    buf = io.BytesIO(df_to_csv_bytes(template))
    return send_file(buf, mimetype='text/csv', as_attachment=True,
                     download_name='candidate_patents_template.csv')
