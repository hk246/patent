"""
特許適合判定分類器モジュール
論文参考: 13種類の分類アルゴリズム比較
  - エイダブースト、ランダムフォレストが最良
  - ディープラーニング (MLP)
  - SVM, k近傍法, ナイーブベイズ, 決定木等
"""

import numpy as np
import pandas as pd
import pickle
from typing import List, Dict, Tuple, Optional

from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier, BaggingClassifier,
    GradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score,
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelBinarizer


# 論文参照の分類器ラインナップ（13種類 + α）
CLASSIFIER_CONFIGS = {
    'エイダブースト': {
        'class': AdaBoostClassifier,
        'params': {'n_estimators': 100, 'random_state': 42},
        'description': '論文で最良評価。弱分類器を逐次的に組み合わせる。',
    },
    'ランダムフォレスト': {
        'class': RandomForestClassifier,
        'params': {'n_estimators': 100, 'random_state': 42},
        'description': '論文で最良評価。多数の決定木のアンサンブル。',
    },
    'SVM (RBFカーネル)': {
        'class': SVC,
        'params': {'kernel': 'rbf', 'probability': True, 'random_state': 42},
        'description': 'ガウシアンカーネルSVM。高次元データに有効。',
    },
    'SVM (線形カーネル)': {
        'class': SVC,
        'params': {'kernel': 'linear', 'probability': True, 'random_state': 42},
        'description': '線形SVM。テキスト分類に向いている。',
    },
    'ロジスティック回帰': {
        'class': LogisticRegression,
        'params': {'max_iter': 1000, 'random_state': 42},
        'description': 'シンプルで解釈しやすい線形分類器。',
    },
    'ニューラルネット (MLP)': {
        'class': MLPClassifier,
        'params': {'hidden_layer_sizes': (300, 100), 'max_iter': 500, 'random_state': 42},
        'description': '多層ニューラルネットワーク。論文のDeep Learning相当。',
    },
    'バギング': {
        'class': BaggingClassifier,
        'params': {'n_estimators': 10, 'random_state': 42},
        'description': 'ブートストラップサンプリングによるアンサンブル。',
    },
    '勾配ブースティング': {
        'class': GradientBoostingClassifier,
        'params': {'n_estimators': 100, 'random_state': 42},
        'description': '勾配ブースティング法。汎化性能が高い。',
    },
    'k近傍法 (k=7)': {
        'class': KNeighborsClassifier,
        'params': {'n_neighbors': 7},
        'description': '論文に記載。k=7の近傍分類器。',
    },
    'ナイーブベイズ': {
        'class': GaussianNB,
        'params': {},
        'description': '論文に記載。スパムフィルターでも使われるベイズ分類。',
    },
    '決定木': {
        'class': DecisionTreeClassifier,
        'params': {'random_state': 42},
        'description': 'C5.0相当の決定木分類器。解釈しやすい。',
    },
}


class PatentClassifier:
    """
    特許適合判定分類器
    論文参考: 正解(signal=1) / ノイズ(noise=0) の2クラス分類
    """

    def __init__(self, algorithm: str = 'ランダムフォレスト'):
        if algorithm not in CLASSIFIER_CONFIGS:
            raise ValueError(f"未対応のアルゴリズム: {algorithm}\n利用可能: {list(CLASSIFIER_CONFIGS.keys())}")

        self.algorithm = algorithm
        config = CLASSIFIER_CONFIGS[algorithm]
        self.model = config['class'](**config['params'])
        self.is_fitted = False
        self.classes_ = None

    def _prepare_X(self, X) -> np.ndarray:
        """入力をndarrayに変換"""
        if hasattr(X, 'toarray'):
            return X.toarray()
        return np.array(X)

    def fit(self, X, y) -> 'PatentClassifier':
        """学習"""
        X = self._prepare_X(X)
        self.model.fit(X, y)
        self.is_fitted = True
        self.classes_ = self.model.classes_
        return self

    def predict(self, X) -> np.ndarray:
        """予測ラベルを返す"""
        if not self.is_fitted:
            raise ValueError("先にfit()を実行してください")
        X = self._prepare_X(X)
        return self.model.predict(X)

    def predict_proba(self, X) -> Optional[np.ndarray]:
        """予測確率を返す（利用可能な場合）"""
        if not self.is_fitted:
            raise ValueError("先にfit()を実行してください")
        X = self._prepare_X(X)
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        return None

    def predict_relevance_score(self, X) -> np.ndarray:
        """
        関連度スコアを返す（0〜1）
        predict_probaが使えない場合はdecision_functionを代用
        """
        proba = self.predict_proba(X)
        if proba is not None:
            # クラス1（正解）の確率を返す
            if 1 in self.model.classes_:
                idx = list(self.model.classes_).index(1)
                return proba[:, idx]
            return proba[:, -1]

        if hasattr(self.model, 'decision_function'):
            X = self._prepare_X(X)
            scores = self.model.decision_function(X)
            # スコアを0〜1に正規化
            min_s, max_s = scores.min(), scores.max()
            if max_s > min_s:
                return (scores - min_s) / (max_s - min_s)
            return np.ones(len(scores)) * 0.5

        return self.predict(X).astype(float)

    def cross_validate(
        self,
        X,
        y,
        cv: int = 5,
        scoring: str = 'f1',
    ) -> Dict:
        """交差検証"""
        X = self._prepare_X(X)
        cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_val_score(self.model, X, y, cv=cv_strategy, scoring=scoring)
        return {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'scores': scores.tolist(),
        }

    def evaluate(self, X_test, y_test) -> Dict:
        """テストデータで評価"""
        y_pred = self.predict(X_test)
        proba = self.predict_proba(X_test)

        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        result = {
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'accuracy': float(report.get('accuracy', 0)),
        }

        # AUCスコア（ラベルが2種類の場合）
        if proba is not None and len(np.unique(y_test)) == 2:
            try:
                if 1 in self.model.classes_:
                    idx = list(self.model.classes_).index(1)
                    auc = roc_auc_score(y_test, proba[:, idx])
                    result['auc_roc'] = float(auc)
            except Exception:
                pass

        return result

    def save(self, path: str):
        """モデルを保存"""
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'algorithm': self.algorithm,
                'is_fitted': self.is_fitted,
                'classes_': self.classes_,
            }, f)

    def load(self, path: str):
        """モデルを読み込み"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        self.algorithm = data['algorithm']
        self.is_fitted = data['is_fitted']
        self.classes_ = data['classes_']


def compare_classifiers(
    X_train,
    y_train,
    X_test,
    y_test,
    algorithms: Optional[List[str]] = None,
    cv: int = 5,
) -> pd.DataFrame:
    """
    複数の分類器を比較評価する
    論文参考: 13種類のアルゴリズムの箱ひげ図比較

    Args:
        X_train, y_train: 学習データ
        X_test, y_test: テストデータ
        algorithms: 比較する分類器リスト (Noneの場合は全て)
        cv: 交差検証の分割数

    Returns:
        分類器毎の評価結果テーブル
    """
    if algorithms is None:
        algorithms = list(CLASSIFIER_CONFIGS.keys())

    results = []
    for algo in algorithms:
        try:
            clf = PatentClassifier(algorithm=algo)
            clf.fit(X_train, y_train)

            # テスト評価
            eval_result = clf.evaluate(X_test, y_test)
            report = eval_result['classification_report']

            # 交差検証
            cv_result = clf.cross_validate(X_train, y_train, cv=cv)

            results.append({
                '分類器': algo,
                '正解率': round(eval_result['accuracy'], 4),
                '精度(1)': round(report.get('1', {}).get('precision', 0), 4),
                '再現率(1)': round(report.get('1', {}).get('recall', 0), 4),
                'F1スコア(1)': round(report.get('1', {}).get('f1-score', 0), 4),
                'CV平均F1': round(cv_result['mean'], 4),
                'CV標準偏差': round(cv_result['std'], 4),
                'AUC-ROC': round(eval_result.get('auc_roc', 0), 4),
            })

        except Exception as e:
            results.append({
                '分類器': algo,
                '正解率': 0,
                '精度(1)': 0,
                '再現率(1)': 0,
                'F1スコア(1)': 0,
                'CV平均F1': 0,
                'CV標準偏差': 0,
                'AUC-ROC': 0,
                'エラー': str(e),
            })

    df = pd.DataFrame(results)
    if 'エラー' not in df.columns:
        df = df.sort_values('F1スコア(1)', ascending=False)
    return df


def get_algorithm_list() -> List[str]:
    """利用可能な分類アルゴリズム名リストを返す"""
    return list(CLASSIFIER_CONFIGS.keys())


def get_algorithm_description(algorithm: str) -> str:
    """アルゴリズムの説明を返す"""
    config = CLASSIFIER_CONFIGS.get(algorithm, {})
    return config.get('description', '')
