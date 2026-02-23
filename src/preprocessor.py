"""
日本語テキスト前処理モジュール
論文参考: MeCab/Janomeによる形態素解析、名詞抽出
"""

import re
from typing import List, Optional


# Janome の遅延インポート（インストール確認用）
_janome_tokenizer = None


def _get_tokenizer():
    global _janome_tokenizer
    if _janome_tokenizer is None:
        try:
            from janome.tokenizer import Tokenizer
            _janome_tokenizer = Tokenizer()
        except ImportError:
            return None
    return _janome_tokenizer


# 日本語ストップワード
STOP_WORDS = set([
    'の', 'に', 'は', 'を', 'た', 'が', 'で', 'て', 'と', 'し',
    'れ', 'さ', 'ある', 'いる', 'も', 'する', 'から', 'な', 'こと',
    'として', 'い', 'や', 'れる', 'など', 'なっ', 'ない', 'この',
    'ため', 'その', 'あっ', 'よっ', 'より', 'もの', 'という',
    'あり', 'まし', 'さらに', 'また', 'および', '等', '場合',
    'ところ', 'しかし', 'ただし', 'すなわち', 'または', 'あるいは',
    'つまり', 'すなわち', 'なお', 'ここで', 'これ', 'それ', 'あれ',
    'どれ', 'どの', 'この', 'その', 'あの', 'こんな', 'そんな',
    'どんな', 'もの', 'こと', 'ところ', 'とき', 'まず', '次に',
    'さらに', '一方', 'ただ', 'なお', 'ゆえ', '以下', '以上',
    '以内', 'からなる', 'なる', 'よる', '基づく',
])

# 特許文書で意味のない一般語
PATENT_STOP_WORDS = set([
    '発明', '特許', '請求', '実施', '例', '形態', '態様', '方法',
    '手段', '工程', '処理', '製造', '装置', '構成', '効果', '目的',
    '問題', '解決', '改善', '向上', '提供', '得る', '行う', '有する',
    '含む', '設ける', '形成', '製品', '物質', 'もの', 'いずれ',
    '上記', '前記', '当該', '所定', '適宜', '各種', '種々',
])


class JapanesePreprocessor:
    """
    日本語特許テキスト前処理クラス
    論文に基づき形態素解析で名詞を抽出しベクトル化の入力を生成する
    """

    def __init__(
        self,
        pos_filter: Optional[List[str]] = None,
        min_length: int = 2,
        use_stop_words: bool = True,
        use_patent_stop_words: bool = True,
    ):
        """
        Args:
            pos_filter: 抽出する品詞リスト (デフォルト: 名詞)
            min_length: 最小文字数 (デフォルト: 2)
            use_stop_words: 一般ストップワードを除去するか
            use_patent_stop_words: 特許ストップワードを除去するか
        """
        self.pos_filter = pos_filter or ['名詞']
        self.min_length = min_length
        self.stop_words = set()
        if use_stop_words:
            self.stop_words |= STOP_WORDS
        if use_patent_stop_words:
            self.stop_words |= PATENT_STOP_WORDS

    def _clean_text(self, text: str) -> str:
        """テキストのクリーニング"""
        if not text:
            return ''
        # 括弧類を除去
        text = re.sub(r'[「」『』【】（）()｛｝{}\[\]〔〕《》]', ' ', text)
        # 特殊文字・記号を除去（化学式等は保持）
        text = re.sub(r'[！？。、，．・：；]', ' ', text)
        # 連続スペースを統一
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def tokenize(self, text: str) -> List[str]:
        """
        テキストをトークン化して名詞リストを返す
        Janomeが利用可能な場合は形態素解析、不可な場合は簡易分割
        """
        if not text or not isinstance(text, str):
            return []

        text = self._clean_text(text)
        tokenizer = _get_tokenizer()

        if tokenizer is not None:
            return self._tokenize_janome(text, tokenizer)
        else:
            return self._tokenize_simple(text)

    def _tokenize_janome(self, text: str, tokenizer) -> List[str]:
        """Janomeによる形態素解析トークン化"""
        tokens = []
        try:
            for token in tokenizer.tokenize(text):
                pos = token.part_of_speech.split(',')[0]
                pos_detail = token.part_of_speech.split(',')[1] if ',' in token.part_of_speech else ''
                surface = token.surface
                base_form = token.base_form if token.base_form and token.base_form != '*' else surface

                # 品詞フィルタ（名詞の場合、非自立・代名詞・数を除外）
                if pos not in self.pos_filter:
                    continue
                if pos == '名詞' and pos_detail in ('非自立', '代名詞', '数', '接尾', '特殊'):
                    continue

                # 長さフィルタ
                if len(surface) < self.min_length:
                    continue

                # ストップワードフィルタ
                if surface in self.stop_words or base_form in self.stop_words:
                    continue

                # 数字のみは除外
                if re.match(r'^[0-9０-９]+$', surface):
                    continue

                tokens.append(surface)
        except Exception:
            pass
        return tokens

    def _tokenize_simple(self, text: str) -> List[str]:
        """
        Janome不使用時の簡易トークン化
        (スペース区切り + 最低限のフィルタリング)
        """
        tokens = text.split()
        result = []
        for token in tokens:
            token = token.strip()
            if len(token) >= self.min_length and token not in self.stop_words:
                result.append(token)
        return result

    def preprocess(self, text: str) -> str:
        """テキストを前処理してスペース区切りトークン文字列を返す"""
        tokens = self.tokenize(text)
        return ' '.join(tokens)

    def preprocess_patent_sections(
        self,
        title: str = '',
        abstract: str = '',
        claims: str = '',
        description: str = '',
        weights: Optional[dict] = None,
    ) -> str:
        """
        特許の各セクションを重み付きで前処理する
        論文参考: タイトル・要約・請求項・実施例の重み付き統合

        Args:
            weights: 各セクションの重み {'title': 3, 'abstract': 2, 'claims': 5, 'description': 1}
        """
        if weights is None:
            weights = {'title': 3, 'abstract': 2, 'claims': 5, 'description': 1}

        all_tokens = []
        sections = [
            ('title', title),
            ('abstract', abstract),
            ('claims', claims),
            ('description', description),
        ]

        for section_name, text in sections:
            if not text:
                continue
            tokens = self.tokenize(text)
            weight = weights.get(section_name, 1)
            all_tokens.extend(tokens * weight)  # 重みに応じてトークンを繰り返す

        return ' '.join(all_tokens)

    def split_into_sentences(self, text: str) -> List[str]:
        """テキストを文に分割する（段落単位の類似度計算用）"""
        if not text:
            return []
        # 句点・改行で文を分割
        sentences = re.split(r'[。\n]+', text)
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) >= 5]

    def extract_claim_elements(self, claims_text: str) -> List[dict]:
        """
        請求項を構成要素単位に分割する
        論文参考: 「構成要素a, b, c...」のような分割
        """
        if not claims_text:
            return []

        elements = []
        # 請求項番号で分割
        claim_pattern = re.split(r'(請求項\s*\d+|【請求項\s*\d+】|\d+\s*[．.]\s*)', claims_text)

        current_claim_num = 1
        for i, part in enumerate(claim_pattern):
            if re.match(r'(請求項\s*\d+|【請求項\s*\d+】|\d+\s*[．.]\s*)', part):
                # 番号から数字を抽出
                nums = re.findall(r'\d+', part)
                if nums:
                    current_claim_num = int(nums[0])
            elif part.strip():
                # 構成要素を「、」や改行で分割
                sub_elements = re.split(r'[、\n]+', part)
                for j, elem in enumerate(sub_elements):
                    elem = elem.strip()
                    if len(elem) >= 5:
                        elements.append({
                            'claim_no': current_claim_num,
                            'element_id': chr(ord('a') + j),  # a, b, c, ...
                            'text': elem,
                            'preprocessed': self.preprocess(elem),
                        })

        return elements
