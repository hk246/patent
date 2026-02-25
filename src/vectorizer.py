"""
テキストベクトル化モジュール
論文参考: TF-IDF, doc2vec, word2vec, SCDV等のベクトル化手法
"""

import numpy as np
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA


# gensim の遅延インポート
def _try_import_gensim():
    try:
        import gensim
        return gensim
    except ImportError:
        return None


class TFIDFDocVectorizer:
    """
    TF-IDFベースの文書ベクトル化
    論文中のBoW / TF-IDFモデルに対応
    """

    def __init__(
        self,
        max_features: int = 10000,
        ngram_range: Tuple[int, int] = (1, 2),
        lsa_components: Optional[int] = None,
    ):
        """
        Args:
            max_features: TF-IDFの最大特徴数
            ngram_range: n-gramの範囲
            lsa_components: LSA (SVD)の次元数 (None=SVDなし)
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.lsa_components = lsa_components
        self.tfidf = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,  # TF値の対数スケーリング
        )
        self.svd = TruncatedSVD(n_components=lsa_components, random_state=42) if lsa_components else None
        self.is_fitted = False

    def fit(self, texts: List[str]) -> np.ndarray:
        """テキストリストでフィット"""
        X = self.tfidf.fit_transform(texts)
        if self.svd:
            X = self.svd.fit_transform(X)
            self.is_fitted = True
            return X
        self.is_fitted = True
        return X.toarray()

    def transform(self, texts: List[str]) -> np.ndarray:
        """テキストをベクトルに変換"""
        if not self.is_fitted:
            raise ValueError("先にfit()を実行してください")
        X = self.tfidf.transform(texts)
        if self.svd:
            return self.svd.transform(X)
        return X.toarray()

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """フィットしてベクトルを返す"""
        X = self.tfidf.fit_transform(texts)
        if self.svd:
            X = self.svd.fit_transform(X)
            self.is_fitted = True
            return X
        self.is_fitted = True
        return X.toarray()

    def get_feature_names(self) -> List[str]:
        return self.tfidf.get_feature_names_out().tolist()

    def get_top_features(self, text: str, top_n: int = 20) -> List[Tuple[str, float]]:
        """テキストの重要語上位N件を返す"""
        vec = self.tfidf.transform([text]).toarray()[0]
        feature_names = self.get_feature_names()
        indices = np.argsort(vec)[::-1][:top_n]
        return [(feature_names[i], float(vec[i])) for i in indices if vec[i] > 0]

    def save(self, path: str):
        """モデルを保存"""
        with open(path, 'wb') as f:
            pickle.dump({
                'tfidf': self.tfidf,
                'svd': self.svd,
                'is_fitted': self.is_fitted,
                'max_features': self.max_features,
                'lsa_components': self.lsa_components,
            }, f)

    def load(self, path: str):
        """モデルを読み込み"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.tfidf = data['tfidf']
        self.svd = data['svd']
        self.is_fitted = data['is_fitted']
        self.max_features = data['max_features']
        self.lsa_components = data['lsa_components']


class Doc2VecVectorizer:
    """
    doc2vecベースの文書ベクトル化
    論文参考: gensimのPV-DBOW / PV-DM モデルを使用
    PV-DBOW: dm=0 (単語の順序を考慮しない、シンプルで高速)
    PV-DM: dm=1 (単語の出現頻度と出現順序を考慮、より精緻)
    """

    def __init__(
        self,
        vector_size: int = 200,
        window: int = 5,
        min_count: int = 2,
        epochs: int = 100,
        dm: int = 0,  # 0=PV-DBOW, 1=PV-DM
        workers: int = 4,
    ):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.dm = dm
        self.workers = workers
        self.model = None
        self.is_fitted = False

    def _check_gensim(self):
        gensim = _try_import_gensim()
        if gensim is None:
            raise ImportError("gensimがインストールされていません: pip install gensim")
        return gensim

    def fit(self, texts: List[str], doc_ids: Optional[List[str]] = None) -> np.ndarray:
        """
        doc2vecモデルを学習する
        Args:
            texts: 前処理済みテキストリスト（スペース区切りトークン）
            doc_ids: 文書IDリスト（Noneの場合は連番）
        """
        gensim = self._check_gensim()
        from gensim.models.doc2vec import Doc2Vec, TaggedDocument

        if doc_ids is None:
            doc_ids = [str(i) for i in range(len(texts))]

        tagged_docs = [
            TaggedDocument(words=text.split(), tags=[doc_id])
            for text, doc_id in zip(texts, doc_ids)
        ]

        self.model = Doc2Vec(
            documents=tagged_docs,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            epochs=self.epochs,
            dm=self.dm,
            workers=self.workers,
            seed=42,
        )
        self.doc_ids = doc_ids
        self.is_fitted = True

        # 全文書のベクトルを返す
        return np.array([self.model.dv[doc_id] for doc_id in doc_ids])

    def transform(self, texts: List[str]) -> np.ndarray:
        """新しいテキストをベクトルに変換（推論）"""
        if not self.is_fitted or self.model is None:
            raise ValueError("先にfit()を実行してください")
        vectors = []
        for text in texts:
            tokens = text.split()
            vec = self.model.infer_vector(tokens, epochs=50)
            vectors.append(vec)
        return np.array(vectors)

    def fit_transform(self, texts: List[str], doc_ids: Optional[List[str]] = None) -> np.ndarray:
        return self.fit(texts, doc_ids)

    def save(self, path: str):
        if self.model:
            self.model.save(path)

    def load(self, path: str):
        gensim = self._check_gensim()
        from gensim.models.doc2vec import Doc2Vec
        self.model = Doc2Vec.load(path)
        self.is_fitted = True


class Word2VecPoolingVectorizer:
    """
    word2vec + プーリング（平均/最大）による文書ベクトル化
    論文参考: 単語ベクトルの平均・最大値プーリング
    """

    def __init__(
        self,
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 2,
        epochs: int = 10,
        pooling: str = 'mean',  # 'mean' or 'max'
    ):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.pooling = pooling
        self.model = None
        self.is_fitted = False

    def _check_gensim(self):
        gensim = _try_import_gensim()
        if gensim is None:
            raise ImportError("gensimがインストールされていません: pip install gensim")
        return gensim

    def fit(self, texts: List[str]) -> np.ndarray:
        gensim = self._check_gensim()
        from gensim.models import Word2Vec

        sentences = [text.split() for text in texts]
        self.model = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            epochs=self.epochs,
            workers=4,
            seed=42,
        )
        self.is_fitted = True
        return self._texts_to_vectors(texts)

    def _text_to_vector(self, text: str) -> np.ndarray:
        """1テキストをベクトル化"""
        tokens = [t for t in text.split() if t in self.model.wv]
        if not tokens:
            return np.zeros(self.vector_size)
        vecs = np.array([self.model.wv[t] for t in tokens])
        if self.pooling == 'mean':
            return np.mean(vecs, axis=0)
        elif self.pooling == 'max':
            return np.max(vecs, axis=0)
        return np.mean(vecs, axis=0)

    def _texts_to_vectors(self, texts: List[str]) -> np.ndarray:
        return np.array([self._text_to_vector(text) for text in texts])

    def transform(self, texts: List[str]) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("先にfit()を実行してください")
        return self._texts_to_vectors(texts)

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        return self.fit(texts)


class VectorizerFactory:
    """ベクトル化手法のファクトリークラス"""

    METHODS = {
        'TF-IDF': 'TF-IDF（高速・安定、デフォルト推奨）',
        'TF-IDF + LSA': 'TF-IDF + 潜在意味解析（次元削減あり）',
        'doc2vec (PV-DBOW)': 'doc2vec PV-DBOWモデル（高精度・学習あり）',
        'doc2vec (PV-DM)': 'doc2vec PV-DMモデル（精緻・計算重め）',
        'Word2Vec + 平均プーリング': 'Word2Vec 単語ベクトル平均（意味ベース）',
    }

    # 各ベクトル化クラスが受け付けるパラメータ
    _TFIDF_PARAMS = {'max_features', 'ngram_range', 'lsa_components'}
    _DOC2VEC_PARAMS = {'vector_size', 'window', 'min_count', 'epochs', 'dm', 'workers'}
    _WORD2VEC_PARAMS = {'vector_size', 'window', 'min_count', 'epochs', 'pooling'}

    @staticmethod
    def _filter_kwargs(kwargs, allowed):
        return {k: v for k, v in kwargs.items() if k in allowed}

    @classmethod
    def create(cls, method: str, **kwargs):
        """指定した手法のベクトル化インスタンスを生成"""
        if method == 'TF-IDF':
            return TFIDFDocVectorizer(**cls._filter_kwargs(kwargs, cls._TFIDF_PARAMS))
        elif method == 'TF-IDF + LSA':
            params = {'lsa_components': 100}
            params.update(cls._filter_kwargs(kwargs, cls._TFIDF_PARAMS))
            return TFIDFDocVectorizer(**params)
        elif method == 'doc2vec (PV-DBOW)':
            params = {'dm': 0}
            params.update(cls._filter_kwargs(kwargs, cls._DOC2VEC_PARAMS))
            return Doc2VecVectorizer(**params)
        elif method == 'doc2vec (PV-DM)':
            params = {'dm': 1}
            params.update(cls._filter_kwargs(kwargs, cls._DOC2VEC_PARAMS))
            return Doc2VecVectorizer(**params)
        elif method == 'Word2Vec + 平均プーリング':
            params = {'pooling': 'mean'}
            params.update(cls._filter_kwargs(kwargs, cls._WORD2VEC_PARAMS))
            return Word2VecPoolingVectorizer(**params)
        else:
            raise ValueError(f"未対応のベクトル化手法: {method}")

    @staticmethod
    def is_gensim_available() -> bool:
        return _try_import_gensim() is not None


def reduce_dimensions(vectors: np.ndarray, n_components: int = 2, method: str = 'SVD') -> np.ndarray:
    """
    ベクトルの次元削減（可視化用）
    論文参考: LSA (SVD)による次元圧縮・散布図作成

    Args:
        vectors: 入力ベクトル行列 (n_samples, n_features)
        n_components: 削減後の次元数
        method: 'SVD' or 'PCA'
    """
    if vectors.shape[1] <= n_components:
        return vectors

    if method == 'SVD':
        reducer = TruncatedSVD(n_components=n_components, random_state=42)
    else:
        reducer = PCA(n_components=n_components, random_state=42)

    return reducer.fit_transform(vectors)
