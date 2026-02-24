"""
セッション状態管理 - st.session_state の代替
DataFrames, numpy arrays, sklearn/gensim モデルをファイルベースで保存
"""
import os
import pickle
import shutil
import time
import pandas as pd

from config import SESSION_DATA_DIR


class SessionStateManager:
    """セッションUUID毎にpickleファイルで状態を管理"""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.base_path = os.path.join(SESSION_DATA_DIR, session_id)
        os.makedirs(self.base_path, exist_ok=True)
        self._touch()

    def _touch(self):
        """セッションディレクトリのアクセス時刻を更新"""
        ts_path = os.path.join(self.base_path, '.last_access')
        with open(ts_path, 'w') as f:
            f.write(str(time.time()))

    def set(self, key: str, value) -> None:
        """任意のPythonオブジェクトをpickleで保存"""
        path = os.path.join(self.base_path, f"{key}.pkl")
        with open(path, 'wb') as f:
            pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)

    def get(self, key: str, default=None):
        """pickleファイルからオブジェクトを読み込み"""
        path = os.path.join(self.base_path, f"{key}.pkl")
        if os.path.exists(path):
            with open(path, 'rb') as f:
                return pickle.load(f)
        return default

    def delete(self, key: str) -> None:
        """指定キーの状態を削除"""
        path = os.path.join(self.base_path, f"{key}.pkl")
        if os.path.exists(path):
            os.remove(path)

    def has(self, key: str) -> bool:
        path = os.path.join(self.base_path, f"{key}.pkl")
        return os.path.exists(path)

    def get_company_patents(self) -> pd.DataFrame:
        return self.get('company_patents', pd.DataFrame())

    def get_patent_db(self):
        return self.get('patent_db', None)

    @staticmethod
    def cleanup_old_sessions(max_age_hours=24):
        """古いセッションデータを削除"""
        if not os.path.exists(SESSION_DATA_DIR):
            return
        cutoff = time.time() - (max_age_hours * 3600)
        for name in os.listdir(SESSION_DATA_DIR):
            session_path = os.path.join(SESSION_DATA_DIR, name)
            if not os.path.isdir(session_path):
                continue
            ts_path = os.path.join(session_path, '.last_access')
            try:
                if os.path.exists(ts_path):
                    with open(ts_path) as f:
                        last_access = float(f.read().strip())
                    if last_access < cutoff:
                        shutil.rmtree(session_path, ignore_errors=True)
                else:
                    mtime = os.path.getmtime(session_path)
                    if mtime < cutoff:
                        shutil.rmtree(session_path, ignore_errors=True)
            except (ValueError, OSError):
                pass
