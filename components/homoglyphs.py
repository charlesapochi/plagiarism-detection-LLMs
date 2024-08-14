"""
Updated version of core.py from
https://github.com/yamatt/homoglyphs/tree/main/homoglyphs_fork
for modern python3
"""

from collections import defaultdict
import json
from itertools import product
import os
import unicodedata

# Actions if char not in alphabet
ACTION_LOAD = 1  # load category for this char
ACTION_IGNORE = 2  # add char to result
ACTION_REMOVE = 3  # remove char from result

ASCII_CHAR_RANGE = range(128)

CURRENT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
DATA_DIRECTORY = os.path.join(CURRENT_DIRECTORY, "data")


class UnicodeCategories:
    """
    Work with aliases from ISO 15924.
    https://en.wikipedia.org/wiki/ISO_15924#List_of_codes
    """

    file_path = os.path.join(DATA_DIRECTORY, "categories.json")

    @classmethod
    def _get_unicode_ranges(cls, categories):
        """
        :return: iter: (start code, end code)
        :rtype: list
        """
        with open(cls.file_path, encoding="utf-8") as file:
            data = json.load(file)

        for category in categories:
            if category not in data["aliases"]:
                raise ValueError(f"Invalid category: {category}")

        for point in data["points"]:
            if point[2] in categories:
                yield point[:2]

    @classmethod
    def get_category_alphabet(cls, categories):
        """
        :return: set of chars in alphabet by categories list
        :rtype: set
        """
        alphabet = set()
        for start, end in cls._get_unicode_ranges(categories):
            chars = (chr(code) for code in range(start, end + 1))
            alphabet.update(chars)
        return alphabet

    @classmethod
    def identify_category(cls, char):
        """
        :return: category
        :rtype: str
        """
        with open(cls.file_path, encoding="utf-8") as file:
            data = json.load(file)

        # try detect category by unicodedata
        try:
            category = unicodedata.name(char).split()[0]
        except (TypeError, ValueError):
            pass
        else:
            if category in data["aliases"]:
                return category

        # try detect category by ranges from JSON file.
        code = ord(char)
        for point in data["points"]:
            if point[0] <= code <= point[1]:
                return point[2]

    @classmethod
    def get_all_categories(cls):
        with open(cls.file_path, encoding="utf-8") as file:
            data = json.load(file)
        return set(data["aliases"])


class LanguageIdentifiers:
    file_path = os.path.join(DATA_DIRECTORY, "languages.json")

    @classmethod
    def get_language_alphabet(cls, languages):
        """
        :return: set of chars in alphabet by languages list
        :rtype: set
        """
        with open(cls.file_path, encoding="utf-8") as file:
            data = json.load(file)
        alphabet = set()
        for lang in languages:
            if lang not in data:
                raise ValueError(f"Invalid language code: {lang}")
            alphabet.update(data[lang])
        return alphabet

    @classmethod
    def identify_languages(cls, char):
        """
        :return: set of languages which alphabet contains passed char.
        :rtype: set
        """
        with open(cls.file_path, encoding="utf-8") as file:
            data = json.load(file)
        languages = set()
        for lang, alphabet in data.items():
            if char in alphabet:
                languages.add(lang)
        return languages

    @classmethod
    def get_all_languages(cls):
        with open(cls.file_path, encoding="utf-8") as file:
            data = json.load(file)
        return set(data.keys())


class HomoglyphManager:
    def __init__(
        self,
        categories=None,
        languages=None,
        alphabet=None,
        strategy=ACTION_IGNORE,
        ascii_strategy=ACTION_IGNORE,
        ascii_range=ASCII_CHAR_RANGE,
    ):
        # strategies
        if strategy not in (ACTION_LOAD, ACTION_IGNORE, ACTION_REMOVE):
            raise ValueError("Invalid strategy")
        self.strategy = strategy
        self.ascii_strategy = ascii_strategy
        self.ascii_range = ascii_range

        # HomoglyphManager must be initialized by any alphabet for correct work
        if not categories and not languages and not alphabet:
            categories = ("LATIN", "COMMON")

        # cats and langs
        self.categories = set(categories or [])
        self.languages = set(languages or [])

        # alphabet
        self.alphabet = set(alphabet or [])
        if self.categories:
            alphabet = UnicodeCategories.get_category_alphabet(self.categories)
            self.alphabet.update(alphabet)
        if self.languages:
            alphabet = LanguageIdentifiers.get_language_alphabet(self.languages)
            self.alphabet.update(alphabet)
        self.table = self._generate_homoglyph_table(self.alphabet)

    @staticmethod
    def _generate_homoglyph_table(alphabet):
        table = defaultdict(set)
        with open(os.path.join(DATA_DIRECTORY, "confusables.json")) as file:
            data = json.load(file)
        for char in alphabet:
            if char in data:
                for homoglyph in data[char]:
                    if homoglyph in alphabet:
                        table[char].add(homoglyph)
        return table

    @staticmethod
    def _generate_restricted_table(source_alphabet, target_alphabet):
        table = defaultdict(set)
        with open(os.path.join(DATA_DIRECTORY, "confusables.json")) as file:
            data = json.load(file)
        for char in source_alphabet:
            if char in data:
                for homoglyph in data[char]:
                    if homoglyph in target_alphabet:
                        table[char].add(homoglyph)
        return table

    @staticmethod
    def _uniq_and_sort(data):
        result = list(set(data))
        result.sort(key=lambda x: (-len(x), x))
        return result

    def _update_alphabet_with_char(self, char):
        # try detect languages
        langs = LanguageIdentifiers.identify_languages(char)
        if langs:
            self.languages.update(langs)
            alphabet = LanguageIdentifiers.get_language_alphabet(langs)
            self.alphabet.update(alphabet)
        else:
            # try detect categories
            category = UnicodeCategories.identify_category(char)
            if category is None:
                return False
            self.categories.add(category)
            alphabet = UnicodeCategories.get_category_alphabet([category])
            self.alphabet.update(alphabet)
        # update table for new alphabet
        self.table = self._generate_homoglyph_table(self.alphabet)
        return True

    def _get_char_variants(self, char):
        if char not in self.alphabet:
            if self.strategy == ACTION_LOAD:
                if not self._update_alphabet_with_char(char):
                    return []
            elif self.strategy == ACTION_IGNORE:
                return [char]
            elif self.strategy == ACTION_REMOVE:
                return []

        # find alternative chars for current char
        alt_chars = self.table.get(char, set())
        if alt_chars:
            # find alternative chars for alternative chars for current char
            alt_chars2 = [self.table.get(alt_char, set()) for alt_char in alt_chars]
            # combine all alternatives
            alt_chars.update(*alt_chars2)
        # add current char to alternatives
        alt_chars.add(char)

        # uniq, sort and return
        return self._uniq_and_sort(alt_chars)

    def _get_combinations(self, text, ascii=False):
        variations = []
        for char in text:
            alt_chars = self._get_char_variants(char)

            if ascii:
                alt_chars = [char for char in alt_chars if ord(char) in self.ascii_range]
                if not alt_chars and self.ascii_strategy == ACTION_IGNORE:
                    return

            if alt_chars:
                variations.append(alt_chars)
        if variations:
            for variant in product(*variations):
                yield "".join(variant)

    def get_all_combinations(self, text):
        return list(self._get_combinations(text))

    def _convert_to_ascii(self, text):
        for variant in self._get_combinations(text, ascii=True):
            if max(map(ord, variant)) in self.ascii_range:
                yield variant

    def convert_to_ascii(self, text):
        return self._uniq_and_sort(self._convert_to_ascii(text))
