"""
特許書誌情報取得モジュール
Azure API Management / J-PlatPat API 対応

【対応API】
1. Azure API Management（カスタムエンドポイント）
   - 認証: Ocp-Apim-Subscription-Key ヘッダー
2. J-PlatPat INPIT API
   - 認証: Bearer Token
3. モック（テスト用）

【JP番号フォーマット】
  入力: JP2020-001001, JP2020001001, 2020001001 など → 自動正規化
"""

import re
import time
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict

import requests
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# データクラス
# ─────────────────────────────────────────────
@dataclass
class PatentRecord:
    """特許書誌情報"""
    patent_id: str = ''
    title: str = ''
    abstract: str = ''
    claims: str = ''
    description: str = ''
    applicant: str = ''
    inventor: str = ''
    filing_date: str = ''
    publication_date: str = ''
    ipc: str = ''
    fi: str = ''
    fetch_status: str = 'pending'   # pending / success / error
    error_message: str = ''

    def to_dict(self) -> dict:
        return asdict(self)

    def is_empty(self) -> bool:
        return not any([self.title, self.abstract, self.claims, self.description])


# ─────────────────────────────────────────────
# 特許番号ユーティリティ
# ─────────────────────────────────────────────
def normalize_jp_number(patent_id: str) -> str:
    """
    JP特許番号を正規化する
    例: 'JP2020-001001' → '2020001001'
        'JP2020001001' → '2020001001'
        '特許第6123456号' → '6123456'
    """
    s = str(patent_id).strip()
    # 前置詞 JP / 特許第 を除去
    s = re.sub(r'^JP[-\s]?', '', s, flags=re.IGNORECASE)
    s = re.sub(r'^特許第?', '', s)
    s = re.sub(r'号$', '', s)
    # ハイフン・スペースを除去
    s = re.sub(r'[-\s]', '', s)
    return s


def format_jp_number_display(patent_id: str) -> str:
    """表示用JP番号フォーマット（例: 2020001001 → JP2020-001001）"""
    norm = normalize_jp_number(patent_id)
    if len(norm) >= 10:
        year = norm[:4]
        num = norm[4:]
        return f"JP{year}-{num}"
    return f"JP{norm}"


# ─────────────────────────────────────────────
# Azure API クライアント
# ─────────────────────────────────────────────
class AzurePatentAPIClient:
    """
    Azure API Management 経由の特許情報取得クライアント

    エンドポイント例:
      https://your-api.azure-api.net/patents/v1/JP2020001001

    認証:
      Ocp-Apim-Subscription-Key: {subscription_key}

    レスポンス例 (JSON):
      {
        "patent_id": "JP2020001001",
        "title": "発明のタイトル",
        "abstract": "要約テキスト",
        "claims": "請求項テキスト",
        "description": "詳細説明",
        "applicant": "出願人",
        "inventor": "発明者",
        "filing_date": "2020-01-01",
        "ipc": "C08J 5/18"
      }

    ※ レスポンスのJSONキー名が異なる場合は field_map で対応付けを設定する
    """

    DEFAULT_FIELD_MAP = {
        # レスポンスのJSONキー → PatentRecord フィールド
        'patent_id':        ['patent_id', 'patentId', 'publication_number', 'docNumber'],
        'title':            ['title', 'inventionTitle', 'patent_title', 'タイトル'],
        'abstract':         ['abstract', 'abstractText', '要約', 'abst'],
        'claims':           ['claims', 'claimsText', '請求項', 'claim'],
        'description':      ['description', 'descriptionText', '詳細説明', 'spec'],
        'applicant':        ['applicant', 'assignee', '出願人', 'applicantName'],
        'inventor':         ['inventor', 'inventors', '発明者', 'inventorName'],
        'filing_date':      ['filing_date', 'filingDate', '出願日', 'applicationDate'],
        'publication_date': ['publication_date', 'publicationDate', '公開日', 'pubDate'],
        'ipc':              ['ipc', 'ipcCode', 'IPC', 'classificationIPC'],
        'fi':               ['fi', 'fiCode', 'FI', 'classificationFI'],
    }

    def __init__(
        self,
        base_url: str,
        subscription_key: str = '',
        bearer_token: str = '',
        timeout: int = 30,
        retry_count: int = 3,
        retry_delay: float = 1.0,
        rate_limit_delay: float = 0.5,
        field_map: Optional[Dict[str, List[str]]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        patent_id_in_url: bool = True,
        patent_id_param: str = 'patent_id',
        url_template: str = '',
    ):
        """
        Args:
            base_url:          API エンドポイントのベースURL
                               例: 'https://your-api.azure-api.net/patents/v1'
            subscription_key:  Azure Subscription Key (Ocp-Apim-Subscription-Key)
            bearer_token:      Bearer Token 認証の場合
            timeout:           リクエストタイムアウト秒数
            retry_count:       リトライ回数
            retry_delay:       リトライ間隔(秒)
            rate_limit_delay:  リクエスト間隔(秒) レート制限対策
            field_map:         レスポンスJSONのキー対応マップ
            extra_headers:     追加リクエストヘッダー
            patent_id_in_url:  True → GET {base_url}/{patent_id}
                               False → クエリパラメータで渡す
            patent_id_param:   クエリパラメータ名 (patent_id_in_url=False の場合)
            url_template:      URLテンプレート (空の場合は base_url/{patent_id})
                               例: '{base_url}/document?id={patent_id}'
        """
        self.base_url = base_url.rstrip('/')
        self.subscription_key = subscription_key
        self.bearer_token = bearer_token
        self.timeout = timeout
        self.retry_count = retry_count
        self.retry_delay = retry_delay
        self.rate_limit_delay = rate_limit_delay
        self.field_map = field_map or self.DEFAULT_FIELD_MAP
        self.extra_headers = extra_headers or {}
        self.patent_id_in_url = patent_id_in_url
        self.patent_id_param = patent_id_param
        self.url_template = url_template

    def _build_headers(self) -> Dict[str, str]:
        headers = {'Accept': 'application/json', 'Content-Type': 'application/json'}
        if self.subscription_key:
            headers['Ocp-Apim-Subscription-Key'] = self.subscription_key
        if self.bearer_token:
            headers['Authorization'] = f'Bearer {self.bearer_token}'
        headers.update(self.extra_headers)
        return headers

    def _build_url(self, patent_id: str) -> Tuple[str, dict]:
        """URL とクエリパラメータを構築する"""
        if self.url_template:
            url = self.url_template.format(base_url=self.base_url, patent_id=patent_id)
            return url, {}
        if self.patent_id_in_url:
            return f"{self.base_url}/{patent_id}", {}
        else:
            return self.base_url, {self.patent_id_param: patent_id}

    def _extract_field(self, data: dict, target_field: str) -> str:
        """レスポンスJSONから対応フィールドを取得"""
        candidate_keys = self.field_map.get(target_field, [target_field])
        for key in candidate_keys:
            # ネストされたキー対応 (例: 'body.text')
            if '.' in key:
                parts = key.split('.')
                val = data
                for part in parts:
                    if isinstance(val, dict):
                        val = val.get(part, '')
                    else:
                        val = ''
                        break
                if val:
                    return str(val)
            elif key in data and data[key]:
                return str(data[key])
        return ''

    def _parse_response(self, data: dict, patent_id: str) -> PatentRecord:
        """レスポンスJSONをPatentRecordに変換"""
        record = PatentRecord(patent_id=patent_id)
        for field_name in ['title', 'abstract', 'claims', 'description',
                           'applicant', 'inventor', 'filing_date',
                           'publication_date', 'ipc', 'fi']:
            setattr(record, field_name, self._extract_field(data, field_name))
        record.fetch_status = 'success'
        return record

    def fetch_one(self, patent_id: str, normalize: bool = True) -> PatentRecord:
        """
        1件の特許情報を取得する

        Args:
            patent_id: JP特許番号
            normalize: Trueの場合、番号を正規化してからAPIを呼び出す
        """
        api_id = normalize_jp_number(patent_id) if normalize else patent_id
        url, params = self._build_url(api_id)
        headers = self._build_headers()

        last_error = ''
        for attempt in range(self.retry_count):
            try:
                resp = requests.get(url, headers=headers, params=params, timeout=self.timeout)

                if resp.status_code == 200:
                    try:
                        data = resp.json()
                        # レスポンスがリストの場合は最初の要素を使用
                        if isinstance(data, list):
                            data = data[0] if data else {}
                        return self._parse_response(data, patent_id)
                    except Exception as e:
                        last_error = f"JSONパースエラー: {e}"

                elif resp.status_code == 404:
                    return PatentRecord(
                        patent_id=patent_id,
                        fetch_status='error',
                        error_message=f"特許が見つかりません (404): {patent_id}",
                    )
                elif resp.status_code == 429:
                    # レート制限
                    wait = self.retry_delay * (attempt + 1) * 2
                    logger.warning(f"レート制限 (429): {wait}秒待機")
                    time.sleep(wait)
                    continue
                elif resp.status_code == 401:
                    return PatentRecord(
                        patent_id=patent_id,
                        fetch_status='error',
                        error_message='認証エラー (401): APIキーを確認してください',
                    )
                else:
                    last_error = f"HTTPエラー {resp.status_code}: {resp.text[:200]}"

            except requests.exceptions.Timeout:
                last_error = f"タイムアウト (試行 {attempt+1}/{self.retry_count})"
            except requests.exceptions.ConnectionError:
                last_error = "接続エラー: APIサーバーに接続できません"
            except Exception as e:
                last_error = f"予期しないエラー: {e}"

            if attempt < self.retry_count - 1:
                time.sleep(self.retry_delay * (attempt + 1))

        return PatentRecord(
            patent_id=patent_id,
            fetch_status='error',
            error_message=last_error,
        )

    def fetch_batch(
        self,
        patent_ids: List[str],
        progress_callback=None,
        normalize: bool = True,
    ) -> List[PatentRecord]:
        """
        複数特許情報を一括取得する

        Args:
            patent_ids:         JP特許番号リスト
            progress_callback:  進捗コールバック (idx, total, patent_id) -> None
            normalize:          番号を正規化するか
        """
        results = []
        total = len(patent_ids)

        for idx, pid in enumerate(patent_ids):
            if progress_callback:
                progress_callback(idx, total, pid)

            record = self.fetch_one(pid, normalize=normalize)
            results.append(record)

            # レート制限対策
            if idx < total - 1:
                time.sleep(self.rate_limit_delay)

        return results

    def test_connection(self) -> Tuple[bool, str]:
        """API接続テスト"""
        try:
            url, _ = self._build_url('JP2020000001')
            headers = self._build_headers()
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code in (200, 404):
                return True, f"接続OK (HTTP {resp.status_code})"
            elif resp.status_code == 401:
                return False, "認証エラー: APIキーを確認してください"
            else:
                return False, f"HTTP {resp.status_code}: {resp.text[:100]}"
        except requests.exceptions.ConnectionError:
            return False, "接続失敗: URLを確認してください"
        except Exception as e:
            return False, f"エラー: {e}"


# ─────────────────────────────────────────────
# J-PlatPat API クライアント（INPIT形式）
# ─────────────────────────────────────────────
class JplatpatAPIClient(AzurePatentAPIClient):
    """
    INPIT J-PlatPat API クライアント
    (J-PlatPat API 登録済みユーザー向け)

    参考: https://jpp.inpit.go.jp/
    """

    JPLATPAT_FIELD_MAP = {
        'patent_id':        ['publicationNumber', 'applicationNumber', 'docId'],
        'title':            ['inventionTitle', 'title', 'titleJa'],
        'abstract':         ['abstractText', 'abstract', 'abstractJa'],
        'claims':           ['claimsText', 'claims', 'claimsJa'],
        'description':      ['descriptionText', 'description', 'descriptionJa'],
        'applicant':        ['applicantName', 'applicant', 'assignee'],
        'inventor':         ['inventorName', 'inventor'],
        'filing_date':      ['applicationDate', 'filingDate'],
        'publication_date': ['publicationDate', 'pubDate'],
        'ipc':              ['ipcCode', 'ipc'],
        'fi':               ['fiCode', 'fi'],
    }

    def __init__(self, api_key: str, base_url: str = 'https://api.j-platpat.inpit.go.jp/rest'):
        super().__init__(
            base_url=base_url,
            bearer_token=api_key,
            field_map=self.JPLATPAT_FIELD_MAP,
            url_template='{base_url}/patent/{patent_id}',
        )


# ─────────────────────────────────────────────
# モック（テスト用）
# ─────────────────────────────────────────────
class MockPatentFetcher:
    """
    APIキーなしでの動作確認用モック
    サンプルデータをランダムに返す
    """

    SAMPLE_TEXTS = [
        {
            'title': 'ガスバリア性フィルムおよびその製造方法',
            'abstract': '熱可塑性樹脂フィルム基材上に無機酸化物蒸着層とポリビニルアルコール系コーティング層を設けたガスバリア性フィルム。',
            'claims': '熱可塑性樹脂フィルム基材層、無機酸化物蒸着層、ポリビニルアルコール系コーティング層を積層してなるガスバリア性フィルム。',
            'description': '本発明はガスバリア性フィルムに関する。ポリエチレンテレフタレートフィルム上に酸化ケイ素を蒸着し、ポリビニルアルコールとモンモリロナイトの混合液を塗工した。',
        },
        {
            'title': '積層体の製造方法',
            'abstract': '基材フィルムと機能性コーティング層を積層した積層体の製造方法。コーティング組成物と積層条件を最適化した。',
            'claims': '基材フィルム上に機能性コーティング組成物を塗工し乾燥する工程を含む積層体の製造方法。',
            'description': '本発明は積層体の製造方法に関する。グラビアコーター法により均一なコーティング層を形成する。',
        },
        {
            'title': 'コーティング組成物および被膜',
            'abstract': 'ポリマーマトリックスと無機フィラーを含むコーティング組成物およびそれから形成される被膜。',
            'claims': 'ポリビニルアルコール系樹脂と無機フィラーを含有するコーティング組成物。',
            'description': 'ポリビニルアルコール水溶液に無機フィラーを分散させ、コーティング液を調製した。塗工乾燥後に優れたバリア性を示した。',
        },
    ]

    def fetch_one(self, patent_id: str, normalize: bool = True) -> PatentRecord:
        import random
        import hashlib
        # 番号に基づいてサンプルを選択（再現性のため）
        h = int(hashlib.md5(patent_id.encode()).hexdigest(), 16)
        sample = self.SAMPLE_TEXTS[h % len(self.SAMPLE_TEXTS)]
        time.sleep(0.1)  # 実際のAPI呼び出しを模倣
        return PatentRecord(
            patent_id=patent_id,
            fetch_status='success',
            **sample,
        )

    def fetch_batch(self, patent_ids, progress_callback=None, normalize=True):
        results = []
        for idx, pid in enumerate(patent_ids):
            if progress_callback:
                progress_callback(idx, len(patent_ids), pid)
            results.append(self.fetch_one(pid, normalize))
        return results

    def test_connection(self):
        return True, "モード: モック（テスト用）"


# ─────────────────────────────────────────────
# ファクトリー関数
# ─────────────────────────────────────────────
def create_fetcher(
    api_type: str = 'azure',
    base_url: str = '',
    subscription_key: str = '',
    bearer_token: str = '',
    field_map: Optional[Dict] = None,
    **kwargs,
):
    """
    APIタイプに応じたフェッチャーを生成する

    Args:
        api_type: 'azure' | 'jplatpat' | 'mock'
        base_url: APIエンドポイントURL
        subscription_key: Azure Subscription Key
        bearer_token: Bearer Token
        field_map: レスポンスフィールドマッピング
    """
    if api_type == 'mock':
        return MockPatentFetcher()

    if api_type == 'jplatpat':
        return JplatpatAPIClient(
            api_key=bearer_token or subscription_key,
            base_url=base_url or 'https://api.j-platpat.inpit.go.jp/rest',
        )

    # azure (default)
    return AzurePatentAPIClient(
        base_url=base_url,
        subscription_key=subscription_key,
        bearer_token=bearer_token,
        field_map=field_map,
        **kwargs,
    )


def records_to_dataframe(records: List[PatentRecord]) -> pd.DataFrame:
    """PatentRecordリストをDataFrameに変換"""
    return pd.DataFrame([r.to_dict() for r in records])


def extract_patent_ids_from_csv(df: pd.DataFrame, id_column: str = 'patent_id') -> List[str]:
    """CSVのDataFrameから特許番号リストを取得"""
    if id_column not in df.columns:
        # JP始まりの列を自動検出
        for col in df.columns:
            sample_vals = df[col].dropna().astype(str).head(5).tolist()
            if any(re.match(r'^JP', v, re.IGNORECASE) or re.match(r'^\d{10}', v) for v in sample_vals):
                id_column = col
                break
    return df[id_column].dropna().astype(str).tolist()
