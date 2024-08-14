from collections import defaultdict
from functools import cache
import re
import unicodedata
import components.homoglyphs as hg


def get_normalizer(strategy_name: str) -> object:
    if strategy_name == "unicode":
        return UnicodeNormalizer()
    elif strategy_name == "homoglyph":
        return HomoglyphNormalizer()
    elif strategy_name == "truecase":
        return TrueCaseNormalizer()


class HomoglyphNormalizer:
    """Detects homoglyph attacks and normalizes text to a consistent canonical form."""

    def __init__(self):
        self.homoglyphs = None

    def __call__(self, text: str) -> str:
        target_category, all_categories = self._identify_categories(text)
        homoglyph_map = self._load_homoglyph_map(target_category, all_categories)
        return self._replace_homoglyphs(target_category, homoglyph_map, text)

    def _identify_categories(self, text: str) -> tuple:
        category_count = defaultdict(int)
        for char in text:
            category_count[hg.UnicodeCategories.identify_category(char)] += 1
        target_category = max(category_count, key=category_count.get)
        all_categories = tuple(category_count)
        return target_category, all_categories

    @cache
    def _load_homoglyph_map(self, target_category: str, all_categories: tuple) -> dict:
        homoglyphs = hg.HomoglyphManager(categories=(target_category, "COMMON"))
        source_alphabet = hg.UnicodeCategories.get_category_alphabet(all_categories)
        return homoglyphs._generate_restricted_table(source_alphabet, homoglyphs.alphabet)

    def _replace_homoglyphs(self, target_category: str, homoglyph_map: dict, text: str) -> str:
        result = ""
        for char in text:
            cat = hg.UnicodeCategories.identify_category(char)
            if target_category in cat or "COMMON" in cat or len(cat) == 0:
                result += char
            else:
                result += list(homoglyph_map[char])[0]
        return result


class UnicodeNormalizer:
    """Normalizes Unicode text according to specified rulesets."""

    def __init__(self, ruleset="whitespace"):
        if ruleset == "whitespace":
            self.pattern = re.compile(
                r"[\u00A0\u1680\u180E\u2000-\u200B\u200C\u200D\u200E\u200F\u2060\u2063\u202F\u205F\u3000\uFEFF\uFFA0\uFFF9\uFFFA\uFFFB"
                r"\uFE00\uFE01\uFE02\uFE03\uFE04\uFE05\uFE06\uFE07\uFE08\uFE09\uFE0A\uFE0B\uFE0C\uFE0D\uFE0E\uFE0F\u3164\u202A\u202B\u202C\u202D"
                r"\u202E\u202F]"
            )
        elif ruleset == "IDN":
            self.pattern = re.compile(
                r"[\u00A0\u1680\u180E\u2000-\u200B\u202F\u205F\u2060\u2063\uFEFF\uFFF9-\uFFFB\uD800-\uDB7F\uDB80-\uDBFF]"
                r"[\uDC00-\uDFFF]?|[\uDB40\uDC20-\uDB40\uDC7F][\uDC00-\uDFFF]"
            )
        else:
            self.pattern = re.compile(r"[^\x00-\x7F]+")

    def __call__(self, text: str) -> str:
        text = unicodedata.normalize("NFC", text)
        text = self.pattern.sub(" ", text)
        text = re.sub(" +", " ", text)
        text = "".join(c for c in text if unicodedata.category(c) != "Cc")
        return text


class TrueCaseNormalizer:
    """Normalizes text to its true capitalization using POS tagging."""

    upper_pos_tags = ["PROPN"]

    def __init__(self, backend="spacy"):
        if backend == "spacy":
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
            self.normalize = self._spacy_normalize
        else:
            from nltk import pos_tag, word_tokenize
            import nltk
            nltk.download("punkt")
            nltk.download("averaged_perceptron_tagger")
            nltk.download("universal_tagset")
            self.normalize = self._nltk_normalize

    def __call__(self, text: str) -> str:
        return self.normalize(text)

    def _spacy_normalize(self, text: str) -> str:
        doc = self.nlp(text.lower())
        return "".join(
            w.text_with_ws.capitalize() if w.pos_ in self.upper_pos_tags or w.is_sent_start else w.text_with_ws
            for w in doc
        )

    def _nltk_normalize(self, text: str) -> str:
        from nltk import pos_tag, word_tokenize
        POS_TAGS = ["NNP", "NNPS"]
        tagged_text = pos_tag(word_tokenize(text.lower()))
        return " ".join(w.capitalize() if p in POS_TAGS else w for w, p in tagged_text)
