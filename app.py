"""
特許クリアランス調査アプリ  v2.1
論文参考: 文字ベクトル化と機械学習を用いた効率的な特許調査
(tokugikon 2018.11.26 no.291 安藤俊幸)

起動:
  pip install -r requirements.txt
  streamlit run app.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import traceback

from preprocessor import JapanesePreprocessor
from vectorizer import TFIDFDocVectorizer, VectorizerFactory, reduce_dimensions
from similarity import (
    compute_cosine_similarity, rank_patents_by_similarity,
    calculate_precision_recall, build_sentence_index,
    compute_claim_element_similarities,
)
from multi_query import (
    compute_multi_query_similarity,
    rank_candidates_multi_query,
    summarize_multi_query_results,
    STRATEGY_DESCRIPTIONS,
)
from classifier import (
    PatentClassifier, compare_classifiers,
    get_algorithm_list, get_algorithm_description,
)

# ════════════════════════════════════════════
# ページ設定
# ════════════════════════════════════════════
st.set_page_config(
    page_title="特許クリアランス調査システム",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════
# セッション状態の初期化
# ════════════════════════════════════════════
def init_session_state():
    defaults = {
        'company_patents': pd.DataFrame(),
        'company_preprocessed': [],
        'company_vectors': None,
        'patent_db': None,
        'preprocessed_db': [],
        'db_vectors': None,
        'vectorizer': None,
        'preprocessor': None,
        'search_results': None,
        'classifier': None,
        'multi_query_strategy': 'max',
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session_state()

# ════════════════════════════════════════════
# ユーティリティ
# ════════════════════════════════════════════
@st.cache_resource
def get_preprocessor(use_stop_words=True):
    return JapanesePreprocessor(
        pos_filter=['名詞'], min_length=2,
        use_stop_words=use_stop_words, use_patent_stop_words=True,
    )

def load_csv(uploaded, fallback_encoding='cp932') -> pd.DataFrame:
    """アップロードCSVを読み込む（UTF-8/CP932 自動判定）"""
    try:
        return pd.read_csv(uploaded, encoding='utf-8')
    except Exception:
        uploaded.seek(0)
        return pd.read_csv(uploaded, encoding=fallback_encoding)

def load_local_csv(path: str) -> pd.DataFrame | None:
    p = os.path.join(os.path.dirname(__file__), path)
    return pd.read_csv(p, encoding='utf-8') if os.path.exists(p) else None

def df_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')

def preprocess_rows(df: pd.DataFrame, preprocessor, weights=None) -> list:
    out = []
    for _, row in df.iterrows():
        t = preprocessor.preprocess_patent_sections(
            title=str(row.get('title','') or ''),
            abstract=str(row.get('abstract','') or ''),
            claims=str(row.get('claims','') or ''),
            description=str(row.get('description','') or ''),
            weights=weights,
        )
        out.append(t)
    return out

def run_vectorize(max_features, use_lsa, lsa_dims, vectorizer_method,
                  use_stop_words, use_claim_weight):
    """自社特許＋候補DBを同一ベクトル空間で再計算する共通関数"""
    cp = st.session_state['company_patents']
    db = st.session_state['patent_db']

    if cp.empty and db is None:
        st.warning("自社特許・候補DBのいずれかを読み込んでください。")
        return False

    preprocessor = get_preprocessor(use_stop_words)
    st.session_state['preprocessor'] = preprocessor
    weights = {'title':3,'abstract':2,'claims':5,'description':1} if use_claim_weight else None

    company_texts = preprocess_rows(cp, preprocessor, weights) if not cp.empty else []
    db_texts = preprocess_rows(db, preprocessor, weights) if db is not None else []

    st.session_state['company_preprocessed'] = company_texts
    st.session_state['preprocessed_db'] = db_texts

    all_texts = company_texts + db_texts
    if not all_texts:
        st.warning("前処理後のテキストが空です。")
        return False

    kwargs = {'max_features': max_features}
    if use_lsa:
        kwargs['lsa_components'] = lsa_dims
    vec = VectorizerFactory.create(vectorizer_method, **kwargs)
    vec.fit(all_texts)
    st.session_state['vectorizer'] = vec

    if company_texts:
        st.session_state['company_vectors'] = vec.transform(company_texts)
    if db_texts:
        st.session_state['db_vectors'] = vec.transform(db_texts)

    return True

# ════════════════════════════════════════════
# サイドバー
# ════════════════════════════════════════════
with st.sidebar:
    st.header("⚙️ 設定")

    st.subheader("ベクトル化手法")
    vectorizer_method = st.selectbox(
        "手法",
        options=list(VectorizerFactory.METHODS.keys()),
        index=0,
    )
    st.caption(VectorizerFactory.METHODS.get(vectorizer_method,''))

    st.subheader("TF-IDF設定")
    max_features = st.slider("最大特徴数", 1000, 30000, 10000, 1000)
    use_lsa = st.checkbox("LSA（次元削減）", value=False)
    lsa_dims = st.slider("LSA次元数", 50, 500, 100, 50, disabled=not use_lsa)

    st.subheader("前処理")
    use_stop_words = st.checkbox("ストップワード除去", value=True)
    use_claim_weight = st.checkbox("請求項を重視（重み5倍）", value=True)

    st.divider()
    st.subheader("複数クエリ集約")
    strategy = st.selectbox(
        "集約戦略",
        options=list(STRATEGY_DESCRIPTIONS.keys()),
        format_func=lambda x: STRATEGY_DESCRIPTIONS[x],
        index=0,
    )
    st.session_state['multi_query_strategy'] = strategy

    st.divider()
    if st.button("🔄 ベクトル化を実行", type="primary", use_container_width=True):
        with st.spinner("ベクトル化中..."):
            ok = run_vectorize(max_features, use_lsa, lsa_dims,
                               vectorizer_method, use_stop_words, use_claim_weight)
            if ok:
                cp_n = len(st.session_state['company_patents'])
                db_n = len(st.session_state['patent_db']) if st.session_state['patent_db'] is not None else 0
                st.success(f"完了 自社:{cp_n}件 候補:{db_n}件")

    st.caption("論文参考: tokugikon 2018.11.26 no.291\n安藤俊幸（花王株式会社）")

# ════════════════════════════════════════════
# メインUI – 6タブ構成
# ════════════════════════════════════════════
st.title("🔬 特許クリアランス調査システム v2.1")
st.caption("自社特許ポートフォリオ × 文字ベクトル化 × 機械学習")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏢 自社特許登録",
    "📂 候補特許DB",
    "🔍 クリアランス調査",
    "🤖 分類器学習",
    "📊 可視化",
    "📖 専門用語解説",
])


# ════════════════════════════════════════════
# TAB 1: 自社特許登録（CSVのみ）
# ════════════════════════════════════════════
with tab1:
    st.header("🏢 自社特許の登録")
    st.info(
        "クリアランス調査の基準となる **自社特許** を CSV ファイルで登録します。"
        "複数件登録することで、ポートフォリオ全体として調査できます。"
    )

    # ─── CSV フォーマット説明 ───
    with st.expander("📋 CSV フォーマット（クリックして確認）", expanded=False):
        st.markdown("""
| 列名 | 必須 | 説明 |
|------|------|------|
| `patent_id` | ✅ | 特許番号（例: JP2020-001001） |
| `title` | ✅ | 発明の名称 |
| `abstract` | 任意 | 要約 |
| `claims` | 推奨 | 請求項（クリアランス精度に大きく影響） |
| `description` | 任意 | 実施例・詳細説明 |
        """)

    col_sample, col_upload = st.columns(2)

    with col_sample:
        st.subheader("サンプルデータを使用")
        st.caption("ガスバリアフィルム関連特許 5件（デモ用）")
        sample_cp = load_local_csv('data/sample_company_patents.csv')
        if sample_cp is not None:
            if st.button("サンプル自社特許を読み込む", type="primary", use_container_width=True):
                st.session_state['company_patents'] = sample_cp
                st.session_state['company_vectors'] = None
                st.success(f"✅ {len(sample_cp)}件を読み込みました")
            with st.expander("サンプルデータのプレビュー"):
                st.dataframe(sample_cp[['patent_id','title']], hide_index=True)
        else:
            st.warning("data/sample_company_patents.csv が見つかりません")

        # テンプレートDL
        template = pd.DataFrame({
            'patent_id': ['JP2020-001001', 'JP2020-001002'],
            'title': ['発明のタイトル1', '発明のタイトル2'],
            'abstract': ['要約テキスト', '要約テキスト'],
            'claims': ['【請求項1】...', '【請求項1】...'],
            'description': ['実施例テキスト', '実施例テキスト'],
        })
        st.download_button(
            "📥 CSVテンプレートをダウンロード",
            data=df_to_csv(template),
            file_name="company_patents_template.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col_upload:
        st.subheader("CSVファイルをアップロード")
        uploaded = st.file_uploader(
            "自社特許 CSV をドラッグ＆ドロップ",
            type=['csv'],
            label_visibility='collapsed',
        )
        if uploaded:
            try:
                df = load_csv(uploaded)
                if 'patent_id' not in df.columns:
                    st.error("'patent_id' 列が必要です。")
                else:
                    for col in ['title','abstract','claims','description']:
                        if col not in df.columns:
                            df[col] = ''
                    st.success(f"読み込み完了: {len(df)}件")
                    st.dataframe(df[['patent_id','title']].head(5), hide_index=True)
                    if st.button("このデータを登録する", type="primary", use_container_width=True):
                        st.session_state['company_patents'] = df.reset_index(drop=True)
                        st.session_state['company_vectors'] = None
                        st.success(f"✅ {len(df)}件を登録しました")
            except Exception as e:
                st.error(f"読み込みエラー: {e}")

    # ─── 登録済みリスト ───
    st.divider()
    cp = st.session_state['company_patents']
    if cp.empty:
        st.info("自社特許はまだ登録されていません。")
    else:
        st.subheader(f"登録済み自社特許: {len(cp)}件")

        # 自社特許のサンプルDL
        st.download_button(
            "📤 登録済み自社特許をCSVでダウンロード",
            data=df_to_csv(cp),
            file_name="company_patents.csv",
            mime="text/csv",
        )

        for i, row in cp.iterrows():
            c1, c2, c3 = st.columns([2, 5, 1])
            with c1:
                st.write(f"**{row.get('patent_id','')}**")
            with c2:
                st.write(str(row.get('title',''))[:80])
            with c3:
                if st.button("削除", key=f"del_{i}", use_container_width=True):
                    st.session_state['company_patents'] = cp.drop(index=i).reset_index(drop=True)
                    st.session_state['company_vectors'] = None
                    st.rerun()

        st.caption("※ ベクトル化はサイドバーの「🔄 ベクトル化を実行」ボタンで行ってください。")

        # ベクトル化状態
        if st.session_state['company_vectors'] is not None:
            vecs = st.session_state['company_vectors']
            st.success(f"✅ ベクトル化済み（{vecs.shape[0]}件 × {vecs.shape[1]}次元）")
        else:
            st.warning("⚠️ まだベクトル化されていません。サイドバーのボタンを押してください。")


# ════════════════════════════════════════════
# TAB 2: 候補特許DB（CSV＋サンプルのみ）
# ════════════════════════════════════════════
with tab2:
    st.header("📂 候補特許データベース")
    st.info("クリアランス調査の対象となる **候補特許** を登録します。")

    col_s, col_u = st.columns(2)

    with col_s:
        st.subheader("サンプルデータを使用")
        st.caption("ガスバリアフィルム関連特許 15件（正解ラベル付き）")
        sample_db = load_local_csv('data/sample_patents.csv')
        if sample_db is not None:
            if st.button("サンプル候補特許を読み込む", type="primary", use_container_width=True):
                st.session_state['patent_db'] = sample_db
                st.session_state['db_vectors'] = None
                st.session_state['preprocessed_db'] = []
                st.success(f"✅ {len(sample_db)}件を読み込みました")
            with st.expander("サンプルデータのプレビュー"):
                st.dataframe(
                    sample_db[['patent_id','title','label']].head(8),
                    hide_index=True,
                )
        else:
            st.warning("data/sample_patents.csv が見つかりません")

        # テンプレートDL
        template_db = pd.DataFrame({
            'patent_id': ['JP2020-XXXXX'],
            'title': ['タイトル'],
            'abstract': ['要約'],
            'claims': ['請求項'],
            'description': ['詳細説明'],
            'label': [''],
        })
        st.download_button(
            "📥 候補特許CSVテンプレート",
            data=df_to_csv(template_db),
            file_name="candidate_patents_template.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col_u:
        st.subheader("CSVファイルをアップロード")
        st.markdown("""
**必須列:** `patent_id`, `title`
**推奨列:** `abstract`, `claims`, `description`
**任意列:** `label`（1=関連、0=無関係）← 分類器学習に使用
        """)
        uploaded_db = st.file_uploader(
            "候補特許 CSV",
            type=['csv'],
            label_visibility='collapsed',
        )
        if uploaded_db:
            try:
                df = load_csv(uploaded_db)
                if 'patent_id' not in df.columns:
                    st.error("'patent_id' 列が必要です。")
                else:
                    for col in ['title','abstract','claims','description']:
                        if col not in df.columns:
                            df[col] = ''
                    st.success(f"読み込み完了: {len(df)}件")
                    st.dataframe(df[['patent_id','title']].head(5), hide_index=True)
                    if st.button("このデータを候補DBに登録", type="primary", use_container_width=True):
                        st.session_state['patent_db'] = df.reset_index(drop=True)
                        st.session_state['db_vectors'] = None
                        st.session_state['preprocessed_db'] = []
                        st.success(f"✅ {len(df)}件を登録しました")
            except Exception as e:
                st.error(f"読み込みエラー: {e}")

    # ─── DB プレビュー ───
    st.divider()
    db = st.session_state['patent_db']
    if db is None:
        st.info("候補特許DBはまだ登録されていません。")
    else:
        st.subheader(f"候補特許DB: {len(db)}件")
        disp_cols = ['patent_id', 'title']
        if 'label' in db.columns:
            disp_cols.append('label')
            vc = db['label'].value_counts()
            c1, c2, c3 = st.columns(3)
            c1.metric("総件数", len(db))
            c2.metric("正解(1)", int(vc.get(1,0)))
            c3.metric("ノイズ(0)", int(vc.get(0,0)))
        st.dataframe(db[disp_cols], height=300, use_container_width=True, hide_index=True)

        if st.session_state['db_vectors'] is not None:
            vecs = st.session_state['db_vectors']
            st.success(f"✅ ベクトル化済み（{vecs.shape[0]}件 × {vecs.shape[1]}次元）")
        else:
            st.warning("⚠️ サイドバーの「🔄 ベクトル化を実行」を押してください。")


# ════════════════════════════════════════════
# TAB 3: クリアランス調査
# ════════════════════════════════════════════
with tab3:
    st.header("🔍 クリアランス調査")

    cp = st.session_state['company_patents']
    db = st.session_state['patent_db']
    company_vecs = st.session_state['company_vectors']
    db_vecs = st.session_state['db_vectors']

    if cp.empty or company_vecs is None:
        st.info("タブ1で自社特許を登録し、サイドバーでベクトル化してください。")
        st.stop()
    if db is None or db_vecs is None:
        st.info("タブ2で候補特許DBを登録し、サイドバーでベクトル化してください。")
        st.stop()

    # ─── 自社特許選択 ───
    st.subheader("① 調査に使用する自社特許を選択")
    all_ids = cp['patent_id'].tolist()
    selected_ids = st.multiselect(
        "使用する自社特許（複数選択可・全選択がデフォルト）",
        options=all_ids, default=all_ids,
    )
    if not selected_ids:
        st.warning("1件以上選択してください。")
        st.stop()

    selected_mask = cp['patent_id'].isin(selected_ids)
    selected_cp = cp[selected_mask].reset_index(drop=True)
    selected_vecs = company_vecs[selected_mask.values]

    with st.expander(f"選択中: {len(selected_cp)}件の自社特許"):
        st.dataframe(selected_cp[['patent_id','title']], hide_index=True, use_container_width=True)

    # ─── 集約戦略 ───
    current_strategy = st.session_state['multi_query_strategy']
    st.info({
        'max': "**集約: 最大スコア** ― いずれかの自社特許に類似した候補が上位にランク。クリアランスに最も安全な選択。",
        'mean': "**集約: 平均スコア** ― 全自社特許との平均的な近さでランキング。",
        'combined': "**集約: 結合ベクトル** ― 自社特許ベクトルの平均（技術重心）からの距離でランキング。",
    }[current_strategy])

    # ─── 実行 ───
    st.subheader("② 類似度計算・ランキング")
    col_run, col_params = st.columns([3,1])
    with col_params:
        top_n = st.slider("表示件数", 5, 200, 30)
        score_threshold = st.slider("最低スコア閾値", 0.0, 1.0, 0.0, 0.01)
    with col_run:
        if st.button(
            f"🚀 クリアランス調査を実行（自社{len(selected_cp)}件 × 候補{len(db)}件）",
            type="primary", use_container_width=True,
        ):
            with st.spinner("類似度計算中..."):
                try:
                    results = rank_candidates_multi_query(
                        patent_df=db,
                        query_vectors=selected_vecs,
                        candidate_vectors=db_vecs,
                        query_patents_df=selected_cp,
                        strategy=current_strategy,
                        top_n=top_n,
                        score_threshold=score_threshold,
                    )
                    st.session_state['search_results'] = results
                    st.success(f"完了: {len(results)}件を表示")
                except Exception as e:
                    st.error(f"エラー: {e}")
                    st.code(traceback.format_exc())

    # ─── 結果 ───
    results = st.session_state['search_results']
    if results is None or results.empty:
        st.stop()

    st.divider()

    # 自社特許別サマリー
    if len(selected_cp) > 1:
        st.subheader("📋 自社特許別サマリー")
        summary_df = summarize_multi_query_results(results, selected_cp)
        st.dataframe(summary_df, hide_index=True, use_container_width=True)
        st.divider()

    # 精度・再現率
    if 'label' in results.columns:
        st.subheader("📈 精度・再現率（Precision / Recall）")
        pr_df = calculate_precision_recall(results)
        if not pr_df.empty and 'エラー' not in pr_df.columns:
            st.dataframe(pr_df, hide_index=True, use_container_width=True)
        st.divider()

    # スコア分布
    fig_bar = px.bar(
        results.reset_index(),
        x='index', y='similarity_score',
        color='label' if 'label' in results.columns else None,
        color_discrete_map={1:'#2ecc71', 0:'#e74c3c'},
        hover_data=['patent_id', 'title'] +
                   (['most_similar_query'] if 'most_similar_query' in results.columns else []),
        title=f"類似度スコア分布（集約: {STRATEGY_DESCRIPTIONS[current_strategy]}）",
        labels={'index':'順位', 'similarity_score':'コサイン類似度'},
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ヒートマップ（複数クエリ時）
    score_cols = [c for c in results.columns if c.startswith('score_')]
    if len(score_cols) > 1:
        st.subheader("🗺️ 候補特許 × 自社特許 類似度ヒートマップ")
        heat = results[['patent_id'] + score_cols].head(30).set_index('patent_id')
        heat.columns = [c.replace('score_','') for c in heat.columns]
        fig_h = px.imshow(
            heat, color_continuous_scale='RdYlGn', aspect='auto',
            title="候補特許(縦) × 自社特許(横) 類似度スコア",
            labels=dict(x="自社特許", y="候補特許", color="類似度"),
        )
        st.plotly_chart(fig_h, use_container_width=True)

    # テーブル
    st.subheader("検索結果")
    disp_cols = ['patent_id','title','similarity_score']
    if 'most_similar_query' in results.columns: disp_cols.append('most_similar_query')
    if 'label' in results.columns: disp_cols.append('label')
    st.dataframe(results[disp_cols], hide_index=True, use_container_width=True)

    # 詳細カード
    st.subheader("上位10件 詳細")
    for _, row in results.head(10).iterrows():
        with st.expander(
            f"[{row['similarity_score']:.4f}]  {row.get('patent_id','')} — {str(row.get('title',''))[:60]}"
        ):
            c1, c2 = st.columns([1,2])
            with c1:
                st.metric("類似度スコア", f"{row['similarity_score']:.4f}")
                if 'most_similar_query' in row:
                    st.caption(f"最類似自社特許: {row['most_similar_query']}")
                if 'label' in row and pd.notna(row['label']):
                    lv = int(row['label'])
                    st.markdown(f"{'🟢 正解(1)' if lv==1 else '🔴 ノイズ(0)'}")
                # 個別スコアバー
                ind = {c.replace('score_',''):row[c] for c in score_cols if c in row.index}
                if ind:
                    fig_ind = px.bar(x=list(ind.values()), y=list(ind.keys()),
                                     orientation='h', range_x=[0,1],
                                     color=list(ind.values()), color_continuous_scale='RdYlGn')
                    fig_ind.update_layout(height=180, margin=dict(l=0,r=0,t=0,b=0),
                                          showlegend=False, coloraxis_showscale=False)
                    st.plotly_chart(fig_ind, use_container_width=True)
            with c2:
                st.markdown("**請求項:**")
                st.write(str(row.get('claims',''))[:400])

    st.download_button(
        "📥 調査結果をCSVでダウンロード",
        data=df_to_csv(results),
        file_name="clearance_results.csv", mime="text/csv",
    )

    # 構成要素分析
    st.divider()
    st.subheader("🔬 請求項の構成要素単位の類似度分析")
    st.caption("自社特許の請求項を構成要素(a,b,c…)に分解し、候補特許内の根拠文を特定します。")
    query_for_elem = st.selectbox("分析する自社特許", selected_cp['patent_id'].tolist())
    if st.button("構成要素分析を実行"):
        with st.spinner("分析中..."):
            try:
                q_row = selected_cp[selected_cp['patent_id']==query_for_elem].iloc[0]
                preprocessor = st.session_state['preprocessor'] or get_preprocessor()
                vec = st.session_state['vectorizer']
                elements = preprocessor.extract_claim_elements(str(q_row.get('claims','') or ''))
                if not elements:
                    st.warning("請求項から構成要素を抽出できませんでした。")
                else:
                    st.write(f"構成要素: {len(elements)}個")
                    st.dataframe(pd.DataFrame(elements)[['element_id','claim_no','text']],
                                 hide_index=True, use_container_width=True)
                    elem_vecs = vec.transform([e['preprocessed'] for e in elements])
                    sent_infos, sent_texts = build_sentence_index(results.head(10), preprocessor)
                    if sent_texts:
                        sent_vecs = vec.transform(sent_texts)
                        elem_result = compute_claim_element_similarities(
                            elements, elem_vecs, sent_infos, sent_vecs)
                        if not elem_result.empty:
                            st.dataframe(elem_result, hide_index=True, use_container_width=True)
            except Exception as e:
                st.error(f"エラー: {e}")
                st.code(traceback.format_exc())


# ════════════════════════════════════════════
# TAB 4: 分類器学習
# ════════════════════════════════════════════
with tab4:
    st.header("🤖 分類器学習（適合判定）")
    st.caption("ラベル付きデータ（正解=1, ノイズ=0）で機械学習モデルを学習し、未ラベル候補特許の関連度を予測します。")

    db = st.session_state['patent_db']
    db_vecs = st.session_state['db_vectors']

    if db is None or db_vecs is None:
        st.info("タブ2で候補特許DBを登録し、ベクトル化してください。")
        st.stop()
    if 'label' not in db.columns:
        st.warning("候補特許DBに 'label' 列が必要です（1=正解, 0=ノイズ）。")
        st.stop()

    labeled_mask = db['label'].notna() & (db['label'].astype(str).str.strip() != '')
    labeled_df = db[labeled_mask].copy()
    labeled_df['label'] = labeled_df['label'].astype(int)
    unlabeled_df = db[~labeled_mask].copy()

    c1,c2,c3 = st.columns(3)
    c1.metric("ラベルあり", f"{len(labeled_df)}件")
    c2.metric("正解(1)", int((labeled_df['label']==1).sum()))
    c3.metric("ノイズ(0)", int((labeled_df['label']==0).sum()))

    if len(labeled_df) < 4:
        st.error("ラベル付きデータが少なすぎます（最低4件）。")
        st.stop()

    col_a, col_b = st.columns(2)
    with col_a:
        algorithm = st.selectbox("分類アルゴリズム", get_algorithm_list(), index=1)
        st.caption(get_algorithm_description(algorithm))
    with col_b:
        test_ratio = st.slider("テスト割合", 0.1, 0.5, 0.2, 0.05)
        cv_folds = st.slider("交差検証分割数", 2, 10, 5)

    if st.button("分類器を学習する", type="primary"):
        with st.spinner("学習中..."):
            try:
                from sklearn.model_selection import train_test_split
                labeled_pos = [db.index.tolist().index(i) for i in labeled_df.index]
                X_l = db_vecs[labeled_pos]
                y_l = labeled_df['label'].values
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X_l, y_l, test_size=test_ratio,
                    stratify=y_l if len(np.unique(y_l))>1 else None, random_state=42)
                clf = PatentClassifier(algorithm=algorithm)
                clf.fit(X_tr, y_tr)
                st.session_state['classifier'] = clf
                ev = clf.evaluate(X_te, y_te)
                report = ev['classification_report']
                cv = clf.cross_validate(X_l, y_l, cv=cv_folds)
                m1,m2,m3,m4 = st.columns(4)
                m1.metric("正解率", f"{ev['accuracy']:.4f}")
                m2.metric("精度(1)", f"{report.get('1',{}).get('precision',0):.4f}")
                m3.metric("再現率(1)", f"{report.get('1',{}).get('recall',0):.4f}")
                m4.metric(f"CV-F1", f"{cv['mean']:.4f}±{cv['std']:.4f}")
                cm = np.array(ev['confusion_matrix'])
                fig_cm = px.imshow(cm, text_auto=True,
                    x=['ノイズ(0)','正解(1)'], y=['ノイズ(0)','正解(1)'],
                    title="混同行列", color_continuous_scale='Blues')
                st.plotly_chart(fig_cm, use_container_width=True)
                st.success("✅ 学習完了")
            except Exception as e:
                st.error(f"エラー: {e}")
                st.code(traceback.format_exc())

    st.divider()
    st.subheader("アルゴリズム比較（13種類）")
    sel_algos = st.multiselect("比較するアルゴリズム", get_algorithm_list(),
        default=['エイダブースト','ランダムフォレスト','SVM (RBFカーネル)','ロジスティック回帰','ニューラルネット (MLP)'])
    if st.button("アルゴリズム比較を実行"):
        if len(sel_algos) < 2:
            st.warning("2件以上選択してください。")
        else:
            with st.spinner("比較中..."):
                try:
                    from sklearn.model_selection import train_test_split
                    labeled_pos = [db.index.tolist().index(i) for i in labeled_df.index]
                    X_l = db_vecs[labeled_pos]
                    y_l = labeled_df['label'].values
                    X_tr, X_te, y_tr, y_te = train_test_split(
                        X_l, y_l, test_size=test_ratio,
                        stratify=y_l if len(np.unique(y_l))>1 else None, random_state=42)
                    cmp_df = compare_classifiers(X_tr,y_tr,X_te,y_te,algorithms=sel_algos,cv=cv_folds)
                    st.dataframe(cmp_df, hide_index=True, use_container_width=True)
                    fig = px.bar(cmp_df, x='分類器', y='F1スコア(1)',
                                 color='F1スコア(1)', color_continuous_scale='RdYlGn',
                                 title="分類器別 F1スコア比較", text_auto='.4f')
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"エラー: {e}")
                    st.code(traceback.format_exc())

    st.divider()
    st.subheader("未ラベル候補特許の適合判定予測")
    clf = st.session_state['classifier']
    if clf is None:
        st.info("先に分類器を学習してください。")
    elif len(unlabeled_df) == 0:
        st.info("未ラベルの特許がありません（全件にラベルがあります）。")
    else:
        if st.button("未ラベル候補特許を予測", type="primary"):
            with st.spinner("予測中..."):
                try:
                    ul_pos = [db.index.tolist().index(i) for i in unlabeled_df.index]
                    X_ul = db_vecs[ul_pos]
                    preds = clf.predict(X_ul)
                    scores = clf.predict_relevance_score(X_ul)
                    pred_df = unlabeled_df.copy()
                    pred_df['predicted_label'] = preds
                    pred_df['relevance_score'] = scores
                    pred_df = pred_df.sort_values('relevance_score', ascending=False)
                    st.success(f"予測完了: 正解予測{int((preds==1).sum())}件 / ノイズ予測{int((preds==0).sum())}件")
                    st.dataframe(pred_df[['patent_id','title','predicted_label','relevance_score']],
                                 hide_index=True, use_container_width=True)
                    st.download_button("📥 予測結果CSV", data=df_to_csv(pred_df),
                                       file_name="prediction_results.csv", mime="text/csv")
                except Exception as e:
                    st.error(f"エラー: {e}")
                    st.code(traceback.format_exc())


# ════════════════════════════════════════════
# TAB 5: 可視化
# ════════════════════════════════════════════
with tab5:
    st.header("📊 可視化")
    st.caption("SVDによる次元圧縮で特許文書の相互関係を2D散布図に表示します。")

    db = st.session_state['patent_db']
    db_vecs = st.session_state['db_vectors']
    cp = st.session_state['company_patents']
    company_vecs = st.session_state['company_vectors']

    if db is None or db_vecs is None:
        st.info("タブ2でベクトル化を実行してください。")
        st.stop()

    c1, c2 = st.columns([1,3])
    with c1:
        dim_method = st.selectbox("次元削減手法", ['SVD (LSA)','PCA'])
        color_by = st.selectbox("色分け", ['自社/候補の区別','label（正解/ノイズ）','類似度スコア','なし'])
        show_labels = st.checkbox("番号を表示", value=False)

    with c2:
        if st.button("散布図を生成", type="primary", use_container_width=True):
            with st.spinner("次元削減中..."):
                try:
                    method_map = {'SVD (LSA)':'SVD','PCA':'PCA'}
                    all_vecs = db_vecs.copy()
                    all_df = db.copy(); all_df['種別'] = '候補特許'

                    if not cp.empty and company_vecs is not None:
                        all_vecs = np.vstack([db_vecs, company_vecs])
                        cp_copy = cp.copy(); cp_copy['種別'] = '自社特許'
                        all_df = pd.concat([all_df, cp_copy], ignore_index=True)

                    reduced = reduce_dimensions(all_vecs, n_components=2, method=method_map[dim_method])
                    all_df['x'] = reduced[:,0]; all_df['y'] = reduced[:,1]

                    results = st.session_state['search_results']
                    if results is not None and 'similarity_score' in results.columns:
                        all_df = all_df.merge(results[['patent_id','similarity_score']], on='patent_id', how='left')

                    color_col, color_map = None, None
                    if color_by == '自社/候補の区別' and '種別' in all_df.columns:
                        color_col = '種別'
                        color_map = {'自社特許':'#f39c12','候補特許':'#3498db'}
                    elif color_by == 'label（正解/ノイズ）' and 'label' in all_df.columns:
                        all_df['label_str'] = all_df['label'].apply(
                            lambda x: '正解(1)' if x==1 else ('ノイズ(0)' if x==0 else '未ラベル'))
                        color_col = 'label_str'
                        color_map = {'正解(1)':'#2ecc71','ノイズ(0)':'#e74c3c','未ラベル':'#95a5a6'}
                    elif color_by == '類似度スコア' and 'similarity_score' in all_df.columns:
                        color_col = 'similarity_score'

                    hover = ['patent_id','title']
                    for c in ['label','similarity_score','種別']:
                        if c in all_df.columns: hover.append(c)

                    fig = px.scatter(all_df, x='x', y='y',
                        color=color_col,
                        color_discrete_map=color_map if isinstance(color_map,dict) else None,
                        color_continuous_scale='RdYlGn' if color_col=='similarity_score' else None,
                        hover_data=hover,
                        text='patent_id' if show_labels else None,
                        symbol='種別' if '種別' in all_df.columns else None,
                        symbol_map={'自社特許':'star','候補特許':'circle'},
                        title=f"特許文書散布図（{dim_method}）  ★=自社特許  ●=候補特許",
                        labels={'x':f'{dim_method}成分1','y':f'{dim_method}成分2'})
                    if show_labels: fig.update_traces(textposition='top center', textfont_size=8)
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("自社特許(★)周囲の候補特許(●)ほどクリアランスに注意が必要です。")
                except Exception as e:
                    st.error(f"エラー: {e}")
                    st.code(traceback.format_exc())

    st.divider()
    results = st.session_state['search_results']
    if results is not None and 'similarity_score' in results.columns:
        st.subheader("類似度スコア分布")
        fig_h = px.histogram(results, x='similarity_score',
            color='label' if 'label' in results.columns else None,
            color_discrete_map={1:'#2ecc71',0:'#e74c3c'},
            title="類似度スコアのヒストグラム", nbins=20, barmode='overlay', opacity=0.7)
        st.plotly_chart(fig_h, use_container_width=True)

    vec = st.session_state['vectorizer']
    if vec is not None and hasattr(vec,'get_top_features') and not cp.empty:
        st.divider()
        st.subheader("自社特許の重要語（TF-IDF）")
        sel_q = st.selectbox("特許を選択", cp['patent_id'].tolist(), key='feat_q')
        q_row = cp[cp['patent_id']==sel_q].iloc[0]
        prep = st.session_state['preprocessor'] or get_preprocessor()
        q_text = prep.preprocess_patent_sections(
            title=str(q_row.get('title','') or ''),
            abstract=str(q_row.get('abstract','') or ''),
            claims=str(q_row.get('claims','') or ''),
            description=str(q_row.get('description','') or ''),
        )
        try:
            feats = vec.get_top_features(q_text, top_n=30)
            if feats:
                fd = pd.DataFrame(feats, columns=['語','TF-IDFスコア'])
                fig_f = px.bar(fd, x='TF-IDFスコア', y='語', orientation='h',
                               color='TF-IDFスコア', color_continuous_scale='Blues',
                               title=f"{sel_q} の重要語 Top30")
                fig_f.update_layout(yaxis={'categoryorder':'total ascending'}, height=500)
                st.plotly_chart(fig_f, use_container_width=True)
        except Exception:
            st.caption("TF-IDF手法のみ対応しています。")


# ════════════════════════════════════════════
# TAB 6: 専門用語解説
# ════════════════════════════════════════════
with tab6:
    st.header("📖 専門用語解説")
    st.caption("本システムで使用している技術用語をカテゴリ別に解説します。")

    term_tab1, term_tab2, term_tab3, term_tab4 = st.tabs([
        "📐 ベクトル化手法",
        "🤖 機械学習・分類",
        "📊 可視化・類似度",
        "🔬 特許調査",
    ])

    # ────────────────────────────
    # ベクトル化手法
    # ────────────────────────────
    with term_tab1:
        st.subheader("テキストのベクトル化とは")
        st.markdown("""
文章（テキスト）をコンピュータが処理できる **数値のベクトル（数値の配列）** に変換する技術です。
特許文書をベクトル化することで、文書間の類似度を数値として計算できるようになります。
        """)

        with st.expander("📌 BoW（Bag-of-Words）― 最もシンプルな手法", expanded=True):
            col1, col2 = st.columns([3,2])
            with col1:
                st.markdown("""
**概念:** 文書中の単語の **出現回数** を数えてベクトルにする

**例:**
- 文書A:「ガスバリア フィルム 製造 ガスバリア」
- BoWベクトル:「ガスバリア=2, フィルム=1, 製造=1」

**特徴:**
- ✅ シンプルで高速
- ✅ 直感的に理解しやすい
- ❌ 単語の順序を無視する
- ❌ 「製造**しない**」と「製造**する**」を区別できない
- ❌ よく使われる単語（助詞等）が大きな値になりすぎる
                """)
            with col2:
                st.markdown("**イメージ:**")
                bow_data = {'単語':['ガスバリア','フィルム','製造','ポリビニルアルコール'],
                            '文書A':[2,1,1,0],'文書B':[1,2,0,1]}
                st.dataframe(pd.DataFrame(bow_data), hide_index=True)

        with st.expander("📌 TF-IDF ― 本システムのデフォルト手法", expanded=True):
            col1, col2 = st.columns([3,2])
            with col1:
                st.markdown("""
**概念:** 単語の重要度を2つの指標で評価する

| 指標 | 意味 | 計算式 |
|------|------|--------|
| **TF** (Term Frequency) | 文書内での単語の出現頻度 | 単語の出現回数 / 全単語数 |
| **IDF** (Inverse Document Frequency) | 全文書に共通して出現する単語を下げる | log(全文書数 / その単語を含む文書数) |

**TF-IDF = TF × IDF**
→ 「その文書に特徴的な単語」を高く評価

**ハイパーパラメータ:**
""")
                st.table(pd.DataFrame({
                    'パラメータ': ['max_features', 'ngram_range', 'sublinear_tf'],
                    '意味': ['使用する単語の最大数（大きいほど精度↑・速度↓）',
                             '連続する単語のまとまり単位（(1,1)=単語のみ、(1,2)=2語の組も使用）',
                             'TFの値を対数スケールにする（大きな差を緩和）'],
                    '本システムの設定': ['10,000（調整可）', '(1,2)', 'True'],
                }))
            with col2:
                st.markdown("**TF-IDFのイメージ:**")
                st.markdown("""
```
「ガスバリア」
→ TF=高（この文書に多い）
→ IDF=高（特許文書全体では珍しい）
→ TF-IDF=高 ✅ 重要語

「する」
→ TF=高（この文書に多い）
→ IDF=低（全文書に出てくる）
→ TF-IDF=低 ❌ 重要でない
```
                """)

        with st.expander("📌 word2vec ― 単語の意味をベクトルで表現"):
            st.markdown("""
**概念:** 単語を **意味的に近い単語がベクトル空間で近くなる** ように学習する

- ニューラルネットワークで大量のテキストから単語の「文脈」を学習
- 例: 「ポリエチレン」と「ポリプロピレン」は意味が似ているのでベクトルが近くなる

**文書ベクトル化（プーリング）:**
- 文書内の全単語のベクトルの **平均（average pooling）** や **最大値（max pooling）** を取る

**ハイパーパラメータ:**
| パラメータ | 意味 | 推奨値 |
|-----------|------|--------|
| `vector_size` | 単語ベクトルの次元数 | 100〜300 |
| `window` | 文脈として考慮する前後の単語数 | 5〜10 |
| `min_count` | この回数以下の単語は無視 | 2〜5 |
| `epochs` | 学習の繰り返し回数 | 10〜100 |
            """)

        with st.expander("📌 doc2vec ― 文書全体をベクトルで表現（論文の主要手法）"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
**概念:** word2vecを拡張し、文書（パラグラフ）全体をベクトルで表現する
論文で最も効果があった手法として報告されている。

**2つのモード:**

**PV-DBOW（dm=0）**
- 文書ベクトルから文書内の単語を予測するモデル
- シンプルで高速
- 論文ではこちらを主に使用

**PV-DM（dm=1）**
- 文書ベクトル＋前後の単語から次の単語を予測
- より精緻だが計算が重い

**ハイパーパラメータ:**
| パラメータ | 意味 | 推奨値 |
|-----------|------|--------|
| `vector_size` | 文書ベクトルの次元数 | **200**（論文値） |
| `epochs` | 学習回数 | **100**（論文値）|
| `window` | 文脈窓サイズ | 5〜10 |
| `min_count` | 最小出現回数 | 2 |
                """)
            with col2:
                st.markdown("""
**学習の仕組み（PV-DBOW）:**
```
文書タグ（特許番号）
       ↓
  [文書ベクトル]
       ↓
  単語を予測 → 誤差を逆伝播
       ↓
  ベクトルを更新
```
**推論（新文書へのベクトル割り当て）:**
```
新しい文書
       ↓
  学習済み語彙で推論
  (infer_vector)
       ↓
  50〜100回繰り返して収束
       ↓
  文書ベクトル
```
                """)

        with st.expander("📌 LSA（潜在意味解析）― TF-IDFの次元削減版"):
            st.markdown("""
**概念:** TF-IDFで得た高次元ベクトル（数万次元）を SVD（特異値分解）で低次元に圧縮する

**効果:**
- 計算を高速化
- 「酸化ケイ素」と「SiO₂」のような同義語の関係を捉えやすくなる
- 過学習を抑制

**ハイパーパラメータ:**
| パラメータ | 意味 | 推奨値 |
|-----------|------|--------|
| `n_components` | 削減後の次元数 | 100〜300 |

**TF-IDF + LSA = TF-IDF LSA（本システムの「TF-IDF + LSA」オプション）**
            """)

        with st.expander("📌 SCDV（Sparse Composite Document Vectors）"):
            st.markdown("""
**概念:** word2vec × K-means クラスタリング × スパース化 を組み合わせた高精度な文書ベクトル化手法

**手順:**
1. 全単語の word2vec ベクトルを学習
2. K-means でクラスタリング（意味グループ化）
3. 各単語をクラスター別に重み付けして合成
4. スパース化（小さい値を0に）

**特徴:**
- ✅ word2vec の意味情報 + TF-IDF の重要度を両立
- ✅ 論文でも可視化で良好なスコアを確認（score=0.756）
- ❌ 計算コストが比較的高い
            """)

    # ────────────────────────────
    # 機械学習・分類
    # ────────────────────────────
    with term_tab2:
        st.subheader("機械学習による適合判定")
        st.markdown("""
特許文書をベクトル化した後、**教師あり機械学習** でその文書が「関連（正解）」か
「無関係（ノイズ）」かを自動判定します。
        """)

        with st.expander("📌 教師あり学習 / 教師なし学習", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
**教師あり学習（本システムの主な手法）**
- 正解ラベル付きデータで学習
- 例: 「この特許は関連(1)」「この特許は無関係(0)」のラベルを与えてモデルを学習
- → 新しい特許に対して自動で関連度を予測できる
                """)
            with col2:
                st.markdown("""
**教師なし学習**
- ラベルなしで文書をグループ化（クラスタリング）
- 例: 似た特許を自動グルーピング、次元削減による可視化
- → 本システムの「散布図」がこれに相当
                """)

        with st.expander("📌 評価指標: 精度・再現率・F1スコア", expanded=True):
            col1, col2 = st.columns([3,2])
            with col1:
                st.markdown("""
**クリアランス調査における重要性:**
- 調査漏れ（見逃し）は致命的なリスク → **再現率** を重視
- ただし精度も低すぎると確認作業が増える → バランスが必要

**指標の意味:**

| 指標 | 計算式 | クリアランスでの意味 |
|------|--------|---------------------|
| **精度 (Precision)** | TP / (TP + FP) | 関連と予測した中で実際に関連する割合 |
| **再現率 (Recall)** | TP / (TP + FN) | 実際の関連特許のうち正しく検出できた割合 |
| **F1スコア** | 2×P×R / (P+R) | 精度と再現率の調和平均 |

TP=正しく「関連」と予測, FP=誤って「関連」と予測, FN=見逃した「関連」
                """)
            with col2:
                st.markdown("**混同行列（Confusion Matrix）:**")
                cm_data = pd.DataFrame(
                    [['TP（正しく検出）','FN（見逃し！）'],
                     ['FP（誤検出）','TN（正しく除外）']],
                    index=['実際: 関連(1)','実際: 無関係(0)'],
                    columns=['予測: 関連(1)','予測: 無関係(0)'],
                )
                st.dataframe(cm_data)
                st.markdown("""
クリアランス調査では **FN（見逃し）を最小化** することが最重要。
→ 再現率が高いモデルを選ぶ
                """)

        with st.expander("📌 交差検証（Cross-Validation）"):
            st.markdown("""
**概念:** データを複数回に分けて学習・評価を繰り返し、モデルの汎化性能を確認する手法

**K分割交差検証の流れ（K=5の場合）:**
```
データ全体 = [A][B][C][D][E]

試行1: 学習=[B][C][D][E], テスト=[A]
試行2: 学習=[A][C][D][E], テスト=[B]
試行3: 学習=[A][B][D][E], テスト=[C]
試行4: 学習=[A][B][C][E], テスト=[D]
試行5: 学習=[A][B][C][D], テスト=[E]

最終スコア = 5回のスコアの平均±標準偏差
```
**なぜ重要？:** テストデータに対する偶然の過学習を防ぎ、モデルの真の性能を推定できる
            """)

        with st.expander("📌 過学習（Overfitting）"):
            st.markdown("""
**概念:** モデルが学習データに過度に適合し、新しいデータに対して性能が下がる現象

```
学習データの正解率: 100%   ← 丸暗記してしまった状態
テストデータの正解率: 60%  ← 新しいデータには弱い
```

**対策:**
- データ量を増やす（特許DB件数を増やす）
- 交差検証でモデルを評価する
- 正規化（Regularization）パラメータを調整
- ドロップアウト（ニューラルネットの場合）
            """)

        with st.expander("📌 各分類アルゴリズムの特徴（論文: 13種類比較）"):
            alg_data = {
                'アルゴリズム': [
                    'エイダブースト（AdaBoost）',
                    'ランダムフォレスト',
                    'SVM（サポートベクターマシン）',
                    'ロジスティック回帰',
                    'ニューラルネット（MLP）',
                    'バギング',
                    '勾配ブースティング',
                    'k近傍法（k-NN）',
                    'ナイーブベイズ',
                    '決定木',
                ],
                '論文評価': ['◎ 最良','◎ 最良','○','○','○','△','○','△','△','△'],
                '特徴': [
                    '弱い分類器を逐次強化。精度が高い',
                    '多数の決定木のアンサンブル。過学習しにくい',
                    '高次元データに強い。テキスト分類に有効',
                    'シンプルで解釈しやすい線形モデル',
                    '深層学習。大量データで真価を発揮',
                    'ブートストラップサンプリングによる並列アンサンブル',
                    '決定木を逐次改善。XGBoostの原型',
                    '近傍k件の多数決。実装がシンプル',
                    '独立性仮定のベイズ分類。スパムフィルターで実績',
                    '解釈しやすいが過学習しやすい',
                ],
            }
            st.dataframe(pd.DataFrame(alg_data), hide_index=True, use_container_width=True)
            st.caption("論文（tokugikon 2018）では AdaBoost・ランダムフォレストが最良の結果を示した。")

        with st.expander("📌 AUC-ROC スコア"):
            st.markdown("""
**AUC-ROC（Area Under the Curve - Receiver Operating Characteristic）**

- モデルの分類性能を **閾値に依存せず** に評価する指標
- ROC曲線: 閾値を変えたときの「真陽性率（再現率）」と「偽陽性率」の関係を示す曲線
- **AUC = ROC曲線の下の面積**

| AUCの値 | 意味 |
|---------|------|
| 1.0 | 完全な分類（理想） |
| 0.9以上 | 非常に優秀 |
| 0.7〜0.9 | 良好 |
| 0.5 | ランダムと同等（最悪） |

クリアランス調査では **AUC≥0.8** を目標にすると良い。
            """)

    # ────────────────────────────
    # 可視化・類似度
    # ────────────────────────────
    with term_tab3:
        st.subheader("類似度計算と可視化の手法")

        with st.expander("📌 コサイン類似度 ― 本システムの類似度指標", expanded=True):
            col1, col2 = st.columns([3,2])
            with col1:
                st.markdown("""
**概念:** 2つのベクトルが **同じ方向を向いているか** で類似度を測る

$$\\text{cosine similarity} = \\frac{\\vec{A} \\cdot \\vec{B}}{|\\vec{A}| \\times |\\vec{B}|}$$

**特徴:**
- 値の範囲: **-1 〜 1**（テキストの場合は 0〜1）
- 1.0: 完全に同じ方向（非常に類似）
- 0.0: 直交（無関係）
- -1.0: 反対方向

**文書量に影響されない:**
- 長い文書と短い文書でも公平に比較できる
- BoW等では文書の長さで単語頻度が変わるが、コサイン類似度は正規化されている

**本システムでの目安:**
| スコア | 意味 |
|--------|------|
| 0.7以上 | 非常に類似（要注意） |
| 0.5〜0.7 | 類似（確認推奨） |
| 0.3〜0.5 | 部分的に類似 |
| 0.3未満 | ほぼ無関係 |
                """)
            with col2:
                st.markdown("**コサイン類似度のイメージ:**")
                theta = np.linspace(0, np.pi/2, 100)
                fig_cos = go.Figure()
                fig_cos.add_trace(go.Scatter(
                    x=np.cos(theta), y=np.sin(theta),
                    mode='lines', name='単位円', line=dict(color='gray', dash='dot')))
                # ベクトルA
                fig_cos.add_annotation(ax=0,ay=0,x=0.9,y=0.43,
                    xref='x',yref='y',axref='x',ayref='y',
                    showarrow=True,arrowhead=2,arrowcolor='blue',arrowwidth=2)
                fig_cos.add_annotation(x=0.95,y=0.48,text='A（自社特許）',
                    showarrow=False,font=dict(color='blue'))
                # ベクトルB（類似）
                fig_cos.add_annotation(ax=0,ay=0,x=0.85,y=0.53,
                    xref='x',yref='y',axref='x',ayref='y',
                    showarrow=True,arrowhead=2,arrowcolor='green',arrowwidth=2)
                fig_cos.add_annotation(x=0.88,y=0.6,text='B（類似: 0.98）',
                    showarrow=False,font=dict(color='green'))
                # ベクトルC（無関係）
                fig_cos.add_annotation(ax=0,ay=0,x=0.2,y=0.98,
                    xref='x',yref='y',axref='x',ayref='y',
                    showarrow=True,arrowhead=2,arrowcolor='red',arrowwidth=2)
                fig_cos.add_annotation(x=0.25,y=1.05,text='C（無関係: 0.25）',
                    showarrow=False,font=dict(color='red'))
                fig_cos.update_layout(height=280, showlegend=False,
                    xaxis=dict(range=[-0.1,1.2],title=''),
                    yaxis=dict(range=[-0.1,1.2],title=''),
                    margin=dict(l=0,r=0,t=10,b=0))
                st.plotly_chart(fig_cos, use_container_width=True)

        with st.expander("📌 SVD（特異値分解）による次元削減", expanded=True):
            st.markdown("""
**概念:** 高次元の特許ベクトルを2次元または3次元に圧縮して可視化する

$$\\text{高次元行列} \\approx U \\cdot \\Sigma \\cdot V^T$$

**本システムでの用途:**
1. **散布図の生成**: 10,000次元のTF-IDFベクトルを2次元に圧縮
2. **LSA（TF-IDF + LSA）**: 100〜300次元に圧縮して類似度計算の精度を上げる

**次元削減の比喩（論文より）:**
> 地球（3次元の球）を地図（2次元の平面）に投影するとき、必ず歪みが生じる。
> SVD・PCAも同様に高次元の情報を2次元に圧縮する際に情報が失われる。
> 地図の図法（メルカトル、モルワイデ等）のように、目的に応じた手法を選ぶことが重要。

**グラフの読み方:**
- 近くにある点 = 意味的に類似した特許
- 自社特許(★)に近い候補特許(●) = クリアランスに要注意
- 同じ色の点が固まっている = 技術分野がよく分類できている
            """)

        with st.expander("📌 PCA（主成分分析）"):
            st.markdown("""
**概念:** データの分散（ばらつき）が最大になる方向を「主成分」として次元削減する

**SVD との違い:**
| | SVD (LSA) | PCA |
|--|-----------|-----|
| 計算対象 | 文書×単語の行列 | 共分散行列 |
| 負の値 | あり | なし（標準化後はあり） |
| 大規模データ | ✅ 得意 | △ 大規模は遅い |
| 本システムの推奨 | ✅ TF-IDFと相性が良い | 参考用 |
            """)

        with st.expander("📌 複数クエリの集約戦略"):
            st.markdown("""
自社特許が複数件ある場合、各候補特許のスコアをどう集約するかの戦略です。

| 戦略 | 計算式 | 特徴 | 推奨用途 |
|------|--------|------|---------|
| **最大スコア（max）** | `max(sim(自社i, 候補j))` | 1件でも類似していれば高スコア | **クリアランス調査（推奨）** |
| **平均スコア（mean）** | `mean(sim(自社i, 候補j))` | 全自社特許への平均的な近さ | 技術ポートフォリオ分析 |
| **結合ベクトル（combined）** | `sim(mean(自社ベクトル), 候補j)` | 自社技術の「重心」からの距離 | 技術領域の中心からの分析 |

**クリアランス調査では「最大スコア」を推奨:**
侵害リスクは「どれか1件の自社特許に類似していれば発生する」ため、最大値で評価するのが最も安全側の判断となります。
            """)

    # ────────────────────────────
    # 特許調査
    # ────────────────────────────
    with term_tab4:
        st.subheader("特許調査・クリアランス調査の基礎用語")

        with st.expander("📌 特許クリアランス調査（Freedom-to-Operate: FTO調査）とは", expanded=True):
            st.markdown("""
**定義:** 新製品・新技術の **製造・販売・使用が他社の有効特許権を侵害しないか** を事前に調査すること

**なぜ重要か:**
- 知らずに他社特許を侵害すると損害賠償・製品回収のリスク
- 特に市場投入前の調査が重要

**調査の流れ（本システムの位置づけ）:**
```
① 調査対象の製品・技術を特定
② 関連する特許を収集（候補特許DB）
③ 自社製品との類似度を評価  ← 本システムの類似度検索
④ 重要特許を選定・詳細分析  ← 本システムのランキング
⑤ 法律専門家（弁理士）と連携して最終判断
```

**注意:** 本システムの結果は **参考情報** です。最終的な侵害判断は弁理士・弁護士が行う必要があります。
            """)

        with st.expander("📌 SDI（Selective Dissemination of Information: 選択的情報配信）"):
            st.markdown("""
**概念:** 事前に定めた技術領域（スコープ）に関連する新着特許情報を自動的に選択・提供するサービス

**SDIとクリアランス調査の違い:**

| | SDI | クリアランス調査 |
|--|-----|----------------|
| 目的 | 競合技術の **動向監視** | 特定製品の **侵害リスク評価** |
| タイミング | 継続的・定期的 | 製品開発・上市前 |
| スコープ | 既定（変わりにくい） | 製品仕様に依存 |
| 対象 | 新着公開特許 | 有効特許（存続中） |

**本論文でのSDI活用:**
自社の技術スコープを定め、2クラス分類（シグナル/ノイズ）で新着特許から関連文書を自動選別する。
            """)

        with st.expander("📌 特許分類コード（IPC・FI・Fターム）"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
**IPC（International Patent Classification）**
- 国際特許分類。全世界共通
- 例: `C08J 5/18`（プラスチックフィルムの製造）
- セクション(C) → クラス(08) → サブクラス(J) → グループ(5/18)

**FI（File Index）**
- 日本特許庁が IPCを細分化した分類
- 例: `C08J 5/18 100`
- 日本語特許の検索に特に有効

**Fターム（File Forming Term）**
- 日本特許庁独自の多面的分類
- 技術的観点（材料・構造・製法等）を複数の軸で分類
- 例: `4F071AA01` (高分子フィルムの製造)
                """)
            with col2:
                st.markdown("""
**特許調査での活用:**
- 類似度検索と組み合わせることで調査精度が向上
- IPC/FIで「関連する技術領域」を絞り込み → 候補特許DB の範囲を限定
- 論文でも「IPC,FI,Ftermをキーワードと組み合わせて使うことが重要」と言及

**本システムへの組み込み方:**
```
候補特許DBを作成する際に、
IPC・FIで絞り込んだ検索結果を
J-PlatPat等でCSVエクスポートし
本システムに読み込む
```
                """)

        with st.expander("📌 精度（Precision）と再現率（Recall）のトレードオフ"):
            st.markdown("""
**クリアランス調査における考え方:**

```
再現率（Recall）= 実際の関連特許のうち何件発見できたか

   再現率100%を目指す場合:
   → 全件確認（コスト大、現実的でない）

   再現率 60% では:
   → 重要な関連特許の40%を見逃す可能性
   → クリアランスリスクが残る

→ クリアランス調査では再現率を最優先に設定
```

**確認数（K）を増やすと:**
- 再現率は上がる（見逃しが減る）
- 精度は下がる（確認が増える）

**本システムの Precision/Recall テーブルの読み方:**

| 確認数(K) | 精度 | 再現率 |
|-----------|------|--------|
| 10 | 0.80 | 0.40 | ← 10件見ても関連特許の40%しかカバーできない |
| 30 | 0.60 | 0.90 | ← 30件確認すれば関連特許の90%をカバー ✅ |
| 50 | 0.40 | 1.00 | ← 50件見れば全件カバー（コストが高い） |

調査工数と見逃しリスクのバランスを取り、**確認数を決定する。**
            """)

        with st.expander("📌 フリーラムレー・ノーフリー定理（NFL定理）"):
            st.markdown("""
**論文で言及されているNFL定理（No Free Lunch Theorem）:**

> 「あらゆる問題に対して最適なアルゴリズムは存在しない」

これは機械学習の重要な定理で、

- 「特許調査に常に最良のアルゴリズム」はない
- データの特性（技術分野、語彙、文書量）によって最良のアルゴリズムは変わる
- → だから本システムでは **13種類のアルゴリズムを比較** して最適なものを選ぶ設計になっている

**実践への示唆:**
- 最初から特定アルゴリズムに決めずに複数を試す
- 自社のデータ（ラベル付き特許）で実際に評価する
- 定期的にモデルを更新・評価し直す
            """)

        with st.expander("📌 スーパーセットの子の定理（クレームの読み方）"):
            st.markdown("""
**論文で言及されている特許クレーム解釈の基本原則:**

クリアランス調査では「文言侵害」の判断が基本となりますが、
均等論・上位概念での侵害にも注意が必要です。

**クレーム要素のスーパーセット（上位概念）問題:**
- 自社製品の要素が、他社特許クレームの上位概念に含まれる場合は侵害になりうる
- 例: 他社特許「ポリビニルアルコール系樹脂を含むコーティング層」
  → 自社製品に「ポリビニルアルコールのコーティング」があれば → 侵害の可能性

**本システムの「構成要素単位の類似度分析」:**
- 各クレームの構成要素(a,b,c...)に分解して類似度を計算
- どの要素が候補特許に含まれているかを根拠文レベルで特定
- より精緻な侵害リスクの事前評価が可能
            """)

        with st.expander("📌 正解公報 / ノイズ公報 の考え方"):
            st.markdown("""
**論文（特許文書の2クラス分類）の用語:**

| 用語 | 意味 | 本システムの `label` 値 |
|------|------|----------------------|
| **正解公報（シグナル）** | クリアランス調査で関連すると判断された特許 | `1` |
| **ノイズ公報** | 関連しないと判断された特許 | `0` |

**教師データの作成方法:**
1. 過去のクリアランス調査結果を利用する
2. 弁理士・研究者が手動でラベルを付ける（本格運用に必須）
3. IPC/FI分類を教師データとして利用する

**教師データ数の目安:**
- 最低: 各クラス10件以上（合計20件）
- 推奨: 各クラス50件以上（合計100件）
- ディープラーニング活用: 500件以上が望ましい

**論文の実験規模:**
- 正解公報: 49件 / ノイズ公報: 705件 / 合計: 754件
            """)

    st.divider()
    st.markdown("""
### 参考文献
- 安藤俊幸.「機械学習を用いた効率的な特許調査 ― アジア特許情報研究会における研究活動紹介」
  *tokugikon* 2018.11.26. no.291, pp.50-64.
- Yoon Kim. "Convolutional Neural Networks for Sentence Classification". *arXiv:1408.5882* (2014).
- Quoc Le, Tomas Mikolov. "Distributed Representations of Sentences and Documents". *ICML 2014*.
- Moen & Salimäki. "Sparse Composite Document Vectors". *arXiv:1612.06778* (2016).
    """)
