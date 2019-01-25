from collections import defaultdict, namedtuple as nt
import unicodedata
from typing import Union, List
from collections import Counter
import json
import cProfile


class LanguagePrimitives(nt('Language',
                            'name spaces digits punctuations accents special_characters alphabets')):
    pass


class ReplaceAndLeaveCharOptions(nt('ReplaceCharOptions', 'replace leave')):
    pass


class LanguageFeatures(object):
    NAME = None
    SPACES = list()
    DIGITS = list()
    PUNCTUATIONS = list()
    ACCENTS = list()
    SPECIAL_CHARACTERS = list()
    ALPHABETS = list()

    def __init__(self, language_attributes):
        assert isinstance(language_attributes, LanguagePrimitives)
        self._name = language_attributes.name
        self._spaces = language_attributes.spaces
        self._digits = language_attributes.digits
        self._punctuations = language_attributes.punctuations
        self._accents = language_attributes.accents
        self._special_characters = language_attributes.special_characters
        self._alphabets = language_attributes.alphabets

    @property
    def name(self):
        if self.__class__.NAME is None:
            self.__class__.NAME = self._name
        return self._name

    @property
    def spaces(self):
        if not any(self.__class__.SPACES):
            self.__class__.SPACES.extend(self._spaces)
        return self._spaces

    @property
    def digits(self):
        if not any(self.__class__.DIGITS):
            self.__class__.DIGITS.extend(self._digits)
        return self._digits

    @property
    def punctuations(self):
        if not any(self.__class__.PUNCTUATIONS):
            self.__class__.PUNCTUATIONS.extend(self._punctuations)
        return self._punctuations

    @property
    def accents(self):
        if not any(self.__class__.ACCENTS):
            self.__class__.ACCENTS.extend(self._accents)
        return self._accents

    @property
    def special_characters(self):
        if not any(self.__class__.SPECIAL_CHARACTERS):
            self.__class__.SPECIAL_CHARACTERS.extend(self._special_characters)
        return self._special_characters

    @property
    def alphabets(self):
        if not any(self.__class__.ALPHABETS):
            self.__class__.ALPHABETS.extend(self._alphabets)
        return self._alphabets

    def __call__(self):
        _ = self.name
        _ = self.spaces
        _ = self.digits
        _ = self.punctuations
        _ = self.accents
        _ = self.special_characters
        _ = self.alphabets
        return self

    def __str__(self):
        return str(self.NAME)


class LanguagePrimitivesToFeatures(object):

    def __init__(self,
                 name,
                 spaces,
                 digits,
                 punctuations,
                 accents,
                 special_characters,
                 alphabets):

        # data for language primitives
        self._name = name
        self._spaces = spaces
        self._digits = digits
        self._punctuations = punctuations
        self._accents = accents
        self._special_characters = special_characters
        self._alphabets = alphabets

        self._primitives = None
        self._lang_features = None
        self()

    def _set_lang_primitives(self):
        if self._primitives is None:
            self._primitives = LanguagePrimitives(self._name,
                                                  self._spaces,
                                                  self._digits,
                                                  self._punctuations,
                                                  self._accents,
                                                  self._special_characters,
                                                  self._alphabets)

    def _set_lang_features(self):
        if self._lang_features is None:
            self._lang_features = LanguageFeatures(self._primitives)

    def __call__(self):
        self._set_lang_primitives()
        self._set_lang_features()

    @property
    def features(self):
        return self._lang_features()


def language_primitive_feature(name,
                               spaces,
                               digits,
                               punctuations,
                               accents,
                               special_characters,
                               alphabets):
    lang_feat = LanguagePrimitivesToFeatures(name=name,
                                             spaces=spaces,
                                             digits=digits,
                                             punctuations=punctuations,
                                             accents=accents,
                                             special_characters=special_characters,
                                             alphabets=alphabets)

    return lang_feat.features


class BaseParser(object):
    # https://github.com/igormq/asr-study/blob/master/preprocessing/text.py
    def __init__(self, language_data):
        self._lang = language_data

    def maps(self, _inputs, replacer):
        pass

    def imaps(self, _inputs):
        pass

    def maybe_replace(self, word, check_list, replace_char=None):
        word = list(word)
        for chid, char in enumerate(word):
            if char in check_list:
                if replace_char is not None:
                    if isinstance(replace_char, str):
                        word[chid] = replace_char
                    else:
                        assert isinstance(replace_char, ReplaceAndLeaveCharOptions)
                        if char == replace_char.leave or char in replace_char.leave:
                            continue
                        else:
                            word[chid] = replace_char.replace
                else:
                    continue
        # word = ''.join(word) please check
        return ''.join(map(str, word))

    def is_valid(self, _inputs):
        pass

    def __call__(self, _inputs, replacer):
        self.maps(_inputs, replacer)


class LanguageParser(BaseParser):

    def __init__(self, language_data, modes=None):

        super(LanguageParser, self).__init__(language_data)
        self._all_modes = {'case_sensitive': 'cs',
                           'punctuations': 'p',
                           'accents': 'a',
                           'digits': 'd',
                           'spaces': 's',
                           'special_characters': 'sc'}
        self._mode_keys = list(self._all_modes.keys())
        self._mode_values = list(self._all_modes.values())
        self._mode_value_to_key = dict(zip(self._mode_values, self._mode_keys))
        if (modes == 'all') or (modes is None):
            self.mode = self._all_modes.keys()
        else:
            self._mode = []
            if ' ' in modes:
                modes = modes.split(' ')
            elif '|' in modes:
                modes = modes.split('|')
            elif ',' in modes:
                modes = modes.split(',')
            else:
                modes = modes

            for mode in modes:

                if ((mode in self._mode_keys) or (mode in self._mode_values)) and any(mode):
                    if mode in self._mode_keys:
                        self._mode.append(mode)
                    else:
                        self._mode.append(self._mode_value_to_key[mode])
                else:
                    raise ValueError(
                        '{mode} mode not supported for {lang}!'.format(mode=mode, lang=self._lang))

            self._replacer = None

    def maps(self, inputs, replacer_dict):

        for mode in self._mode:
            for idx, word in enumerate(inputs):
                if mode == 'punctuations':
                    inputs[idx] = self.punctuations(word, replacer_dict[mode].value)
                if mode == 'digits':
                    inputs[idx] = self.digits(word, replacer_dict[mode].value)
                if mode == 'saces':
                    inputs[idx] = self.spaces(word, replacer_dict[mode].value)
                if mode == 'special_characters':
                    inputs[idx] = self.special_characters(word, replacer_dict[mode].value)
                if mode == 'accents':
                    inputs[idx] = self.accents(word, replacer_dict[mode].value)
                if mode == 'case_sensitive':
                    inputs[idx] = self.case_sensitive(word, case=replacer_dict[mode].value)
        return inputs

    def punctuations(self, word, replace_char=None):
        return self.maybe_replace(word, self._lang.PUNCTUATIONS, replace_char)

    def digits(self, word, replace_char=None):
        return self.maybe_replace(word, self._lang.DIGITS, replace_char)

    def spaces(self, word, replace_char=None):
        return self.maybe_replace(word, self._lang.SPACES, replace_char)

    def special_characters(self, word, replace_char=None):
        return self.maybe_replace(word, self._lang.SPECIAL_CHARACTERS, replace_char)

    def accents(self, word, replace_char=None):
        if replace_char is None:
            return self._strip_accents(word)
        else:
            return self.maybe_replace(word, self._lang.ACCENTS, replace_char)

    @staticmethod
    def case_sensitive(word, case=None):
        if case:
            return word.lower()
        else:
            return word

    @staticmethod
    def _strip_accents(s):
        return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

    @property
    def mode_list(self):
        return self._mode

    def replacer(self, punctuations_replacer: Union[str, ReplaceAndLeaveCharOptions] = None,
                 spaces_replacer: Union[str, ReplaceAndLeaveCharOptions] = None,
                 digits_replacer: Union[str, ReplaceAndLeaveCharOptions] = None,
                 special_characters_replacer: Union[str, ReplaceAndLeaveCharOptions] = None,
                 accents_replacer: Union[str, ReplaceAndLeaveCharOptions] = None,
                 case_sensitive_replace: Union[bool, ReplaceAndLeaveCharOptions] = None):
        modes = dict(zip(self.mode_list, self.mode_list))
        for mode in self.mode_list:
            if mode == 'punctuations':
                modes[mode] = nt('{name}'.format(name=mode), 'value')(punctuations_replacer)
            if mode == 'spaces':
                modes[mode] = nt('{name}'.format(name=mode), 'value')(spaces_replacer)
            if mode == 'digits':
                modes[mode] = nt('{name}'.format(name=mode), 'value')(digits_replacer)
            if mode == 'special_characters':
                modes[mode] = nt('{name}'.format(name=mode), 'value')(special_characters_replacer)
            if mode == 'accents':
                modes[mode] = nt('{name}'.format(name=mode), 'value')(accents_replacer)
            if mode == 'case_sensitive':
                modes[mode] = nt('{name}'.format(name=mode), 'value')(case_sensitive_replace)
        if self._replacer is None:
            self._replacer = modes
        return self._replacer


sos_token = 'SOS'
eos_token = 'EOS'


class SentenceIndexer(object):

    def __init__(self, language_name):
        # data language
        self._lang = language_name

        # indexing
        self.n_word = 0
        self._word2count = defaultdict(int)
        self._index2word = {0: sos_token,
                            1: eos_token}

        self._word2index = defaultdict(list)
        self._sentence = None

    def index_words(self, sentence):
        _sentence = list()
        _sentence.append(self._index2word[0])
        if not isinstance(sentence, list):
            sentence = sentence.split(' ')
        _sentence.extend(sentence)
        _sentence.append(self._index2word[1])
        for idx, words in enumerate(_sentence):
            print(idx, words)
            self.index_word(idx, words)

    def index_word(self, idx, word):
        word = word.lower()
        if word not in self.word2index:
            self.n_word += 1
            self._word2index[word].append(idx)
            self._index2word[idx] = word
            self._word2count[word] = 1
        else:
            self.n_word += 1
            self._word2count[word] += 1
            self._word2index[word].append(idx)
            self._index2word[idx] = word
        if self.sentence is None:
            self._sentence = list()
        self._sentence.append(word)

    @property
    def index2word(self):
        return self._index2word

    @property
    def word2index(self):
        return dict(self._word2index)

    @property
    def word2count(self):
        return dict(self._word2count)

    @property
    def sentence(self):
        return self._sentence

    @property
    def lang(self):
        return self._lang


NodeCount = 0


# set of words.
class TrieNode:

    def __init__(self):

        self.word = None
        self.children = {}

        global NodeCount
        NodeCount += 1

    def insert(self, word):
        node = self
        for letter in word:
            if letter not in node.children:
                print(letter)
                node.children[letter] = TrieNode()
            node = node.children[letter]

        node.word = word

    def get_tree_dict(self, node=None):
        node = node or self
        letters = dict()
        if any(node.children):
            for letter in node.children:
                letters[letter] = {'word': node.children[letter].word,
                                   'children': self.get_tree_dict(node.children[letter])}
        if any(letters):
            return letters
        else:
            return 'end'

    def get_words(self, tree_dict: dict=None, remove_if_all=None, remove_if_any=None):
        tree_dict = tree_dict or self.get_tree_dict()
        words_list = list()
        for k in tree_dict:
            if hasattr(tree_dict[k]['children'], 'keys'):
                words_list.extend(self.get_words(tree_dict[k]['children']))
            else:
                if tree_dict[k]['word'] is not None:
                    words_list.append(tree_dict[k]['word'])

        if remove_if_all:
            for words in words_list:
                if all(word in words for word in remove_if_all):
                    for word in remove_if_all:
                        words.pop(words.index(word))

        if remove_if_any:
            for words in words_list:
                if any(word in words for word in remove_if_any):
                    for word in remove_if_any:
                        if word in words:
                            for maybe_remove_word in words:
                                if maybe_remove_word == word:
                                    words.pop(words.index(word))
        if any(words_list):
            return words_list
        else:
            pass

    def delete(self, word, trie):
        node = trie
        for letter in word:
            if any(node.children[letter]):
                print(node.children[letter])
                node = node.children[letter]
        if node.word == word:
            del node


class Language(object):

    def __init__(self,
                 name,
                 spaces,
                 digits,
                 punctuations,
                 accents,
                 special_characters,
                 alphabets):
        self.lang_data = None
        if self.lang_data is None:
            self.lang_data = language_primitive_feature(name=name, spaces=spaces, digits=digits,
                                                        punctuations=punctuations, accents=accents,
                                                        special_characters=special_characters,
                                                        alphabets=alphabets)
        self.sentence_indexer = None

    def parser(self, modes: str = None):
        return LanguageParser(self.lang_data, modes=modes)

    def maps(self, parser: LanguageParser, words: list, replacer_dict: dict):
        parsed = list()
        for word in words:
            parsed.extend(parser.maps([word], replacer_dict=replacer_dict))
        return parsed

    def index_words(self, words: list):
        self.sentence_indexer = SentenceIndexer(language_name=self.lang_data.NAME)
        self.sentence_indexer.index_words(sentence=words)
        return self.sentence_indexer

    def trie(self, words: list, keep_word_as_keys: bool = False):
        trie = TrieNode()
        if keep_word_as_keys:
            trie.insert(words)
        else:
            for word in words:
                trie.insert(word)
        return trie


class Parser(object):

    def __init__(self, language: Language,
                 modes: str = None,
                 case_sensitive_replace: Union[bool, ReplaceAndLeaveCharOptions] = None,
                 spaces_replacer: Union[str, ReplaceAndLeaveCharOptions] = None,
                 punctuations_replacer: Union[str, ReplaceAndLeaveCharOptions] = None,
                 digits_replacer: Union[str, ReplaceAndLeaveCharOptions] = None,
                 accents_replacer: Union[str, ReplaceAndLeaveCharOptions] = None,
                 special_characters_replacer: Union[str, ReplaceAndLeaveCharOptions] = None):
        self._lang = language
        self._modes = modes
        self._parser = self._lang.parser(self._modes)
        self._replacer = self._parser.replacer(case_sensitive_replace=case_sensitive_replace,
                                               spaces_replacer=spaces_replacer,
                                               punctuations_replacer=punctuations_replacer,
                                               digits_replacer=digits_replacer,
                                               accents_replacer=accents_replacer,
                                               special_characters_replacer=special_characters_replacer)

    def parse_sentence(self, sentence_list: list):
        return self._lang.maps(self._parser, sentence_list, self._replacer)

    def index_words(self, sentence_list_or_str):
        return self._lang.index_words(sentence_list_or_str)

    def trie(self, words: list, keep_word_as_keys: bool = False):
        return self._lang.trie(words, keep_word_as_keys=keep_word_as_keys)


def en_lang():
    en_acc_a = 'àáâãäåæÀÁÂÃÄÅÆ'
    en_acc_e = 'ēĕėęěĒÈÉÊË'
    en_acc_i = ''
    en_acc_o = ''
    en_acc_u = ''

    EN_NAME = 'English'
    EN_SPACES = ' '
    EN_DIGITS = '0123456789'
    EN_ALPHABETS = 'abcdefghijklemnopqrstuvwxyz'
    EN_PUNCTUATIONS = '.,?&\'!\";'
    EN_ACCENTS = ''.join([en_acc_a, en_acc_e, en_acc_i, en_acc_o, en_acc_u])
    EN_SPECIAL_CHARACTERS = '@#\$%*^~:`<>(_)[-]{|}+=\t\n\r'
    language = Language(name=EN_NAME,
                        spaces=EN_SPACES,
                        digits=EN_DIGITS,
                        alphabets=EN_ALPHABETS,
                        punctuations=EN_PUNCTUATIONS,
                        special_characters=EN_SPECIAL_CHARACTERS,
                        accents=EN_ACCENTS)

    return language


en_parser = Parser(language=en_lang(),
                   modes='s,p,cs,sc,a',  # No digits are removed as digits mode 'd' is absent
                   case_sensitive_replace=True,  # Lower Case data
                   digits_replacer='',  # Even though the digits remover is present the digits aren't removed
                   punctuations_replacer=ReplaceAndLeaveCharOptions('', ['.', '&']),
                   spaces_replacer='',
                   special_characters_replacer=ReplaceAndLeaveCharOptions('', ['#', '-', '+']))


data_json = r'C:/Users/shivam.agarwal/naukri_skills.json'
remove_if_all_words = ['it', 'skills', 'details']
remove_if_any_words = ['.', '-', '/', 'and', '&', '&amp', 'are', 'of', 'in', 'the', 'getter',
                       'then', 'there', 'thereby', 'thereafter', 'their', 'do', 'can', 'which',
                       'a', 'amp', 'all', 'by', 'well', 'have', 'good', 'through', 'they', 'know',
                       'more', 'other', 'both', 'looking', 'able', 'now', 'be', 'highly', 'why',
                       'one', 'within', 'since', 'take', 'such', 'entire', 'taking',
                       'into', 'how', 'an', 'i', 'my', 'am', 'is', 'on', 'with', 'to', 'for',
                       'as', 'about', 'or', 'using', 'like', 'having', 'from', 'at', 'etc.',
                       'etc', 'also', '+', '', '--', '..', '...', '.....', '�']

start_tags = ['..', '...', '....', '/', '-', '&', '#', '+', 'a¢']
middle_tags = ['/', '..', '...', '&', '¿', '½', 'a€¢', '�', '€']
end_tags = ['.', '-', '/', '&', '..', '...', '�']


class WordList(object):

    def __init__(self, parser,
                 remove_if_all=None,
                 remove_if_any=None):
        self._data = None
        self._parser = parser
        self._remove_if_all = remove_if_all
        self._remove_if_any = remove_if_any

    def _open_json(self, json_path):
        if self._data is None:
            with open(json_path, 'rb') as f:
                self._data = json.load(f)
        return self._data

    def _clean_dict(self, data_dict, parser):
        clean_dict = dict()
        for key in data_dict:
            for doc in data_dict[key]:
                if data_dict[key][doc] is not None:
                    clean_dict[doc] = self._parse_words_list(
                        parser.parse_sentence(data_dict[key][doc]), parser)
        return clean_dict

    @staticmethod
    def _parse_words_list(word_list, parser):
        new_word_list = list()
        for word in word_list:
            word = parser.parse_sentence(word.split())
            if any(word):
                new_word_list.append(' '.join(word))
        return new_word_list

    @staticmethod
    def _clean_if_startswith(word: str, start_tags: List[str]):
        assert isinstance(word, str), 'Only Strings allowed'
        for start_tag in start_tags:
            if word.startswith(start_tag):
                word = word[len(start_tag):]
        return word

    @staticmethod
    def _clean_if_endswith(word: str,
                           end_tags: List[str]):
        assert isinstance(word, str), 'Only Strings allowed'
        for end_tag in end_tags:
            if word.endswith(end_tag):
                word = word[:-len(end_tag)]
        return word

    @staticmethod
    def _clean_if_in_between(word: str, middle_tags: List[str]):
        assert isinstance(word, str), 'Only Strings allowed'
        for middle_tag in middle_tags:
            if not (word.startswith(middle_tag)
                    or word.endswith(middle_tag)) and (middle_tag in word):
                word = word.split(middle_tag)
                word2 = list()
                for wd in word:
                    if any(wd):
                        word2.append(wd)
                word = ''.join(word2)
        return word

    def _clean_if_whitespace(self,
                             word: str,
                             start_tags: List[str],
                             middle_tags: List[str],
                             end_tags: List[str]):
        assert isinstance(word, str), 'Only Strings allowed'
        if ' ' in word and len(word) > 1:
            word = word.split(' ')
            word2 = list()
            for wd in word:

                wd = self._clean_if_startswith(wd, start_tags)
                wd = self._clean_if_endswith(wd, end_tags)
                wd = self._clean_if_in_between(wd, middle_tags)
                if any(wd):
                    word2.extend(wd.split())
            word = word2
        else:
            word = self._clean_if_startswith(word, start_tags)
            word = self._clean_if_endswith(word, end_tags)
            word = self._clean_if_in_between(word, middle_tags)
            if ' ' in word:
                word = word.split()
            else:
                word = [word]
        return word

    def _clean_word_in_list(self,
                            word,
                            remove_start_tags=start_tags,
                            remove_middle_tags=middle_tags,
                            remove_end_tags=end_tags):
        word = self._clean_if_whitespace(word,
                                         start_tags=remove_start_tags,
                                         middle_tags=remove_middle_tags,
                                         end_tags=remove_end_tags)
        return word

    def _trie(self, data_list):
        trie = TrieNode()
        for data in data_list:
            for word in data:
                word_list = self._clean_word_in_list(word)
                trie.insert(word_list)
        return trie

    def _get_words(self,
                   tree_dict_or_word_lists,
                   parser,
                   remove_if_all=None,
                   remove_if_any=None,
                   make_trie=None):
        if make_trie is None:
            make_trie = True

        if isinstance(tree_dict_or_word_lists, list) and make_trie:
            words_list = self._trie(tree_dict_or_word_lists).get_words(remove_if_all=remove_if_all,
                                                                       remove_if_any=remove_if_any)
        else:
            assert isinstance(tree_dict_or_word_lists, (dict, list))
            if isinstance(tree_dict_or_word_lists, dict):
                words_list = TrieNode().get_words(tree_dict_or_word_lists,
                                                  remove_if_all=remove_if_all,
                                                  remove_if_any=remove_if_any)
            else:
                assert isinstance(tree_dict_or_word_lists, list)
                words_list = self._remove_if(tree_dict_or_word_lists,
                                             remove_if_all=remove_if_all,
                                             remove_if_any=remove_if_any)
                clean_list = list()
                for words in words_list:
                    temp_words_list = list()
                    for word in words:
                        temp_words_list.extend(self._clean_word_in_list(word))
                    temp_words_list = self._parse_words_list(temp_words_list, parser)
                    self._remove_if([temp_words_list],
                                    remove_if_all=remove_if_all,
                                    remove_if_any=remove_if_any)
                    clean_list.append(temp_words_list)
                words_list = clean_list
        return words_list

    @staticmethod
    def _remove_if(words_list, remove_if_all=None, remove_if_any=None):
        if remove_if_all:
            for words in words_list:
                if all(word in words for word in remove_if_all):
                    for word in remove_if_all:
                        words.pop(words.index(word))

        if remove_if_any:
            for words in words_list:
                if any(word in words for word in remove_if_any):
                    for word in remove_if_any:
                        if word in words:
                            for maybe_remove_word in words:
                                if maybe_remove_word == word:
                                    words.pop(words.index(word))
        return words_list

    def __call__(self, json_path_or_dict_or_words_list):
        if isinstance(json_path_or_dict_or_words_list, (str, dict)):
            make_trie = True
            json_dict = self._open_json(json_path_or_dict_or_words_list) \
                if isinstance(json_path_or_dict_or_words_list, str) else json_path_or_dict_or_words_list
            clean_dict = self._clean_dict(json_dict, self._parser)
            print('cleaning dict....')
            trie = self._trie(clean_dict.values())
            trie_dict_or_words_list = trie.get_tree_dict()
        else:
            assert isinstance(json_path_or_dict_or_words_list, list)
            make_trie = False
            trie_dict_or_words_list = json_path_or_dict_or_words_list
        words_list = self._get_words(trie_dict_or_words_list,
                                     self._parser,
                                     remove_if_all=self._remove_if_all,
                                     remove_if_any=self._remove_if_any,
                                     make_trie=make_trie)

        return words_list


def most_common(words_list, word_count, percentage):
    counts = dict()
    values = dict()
    for l in range(40):
        counts[(l + 1)] = sum([1 for i in words_list for _ in i if l < len(i) < (l + 2)])
        values[(l + 1)] = [i for i in words_list if l < len(i) < (l + 2)]
    check_words = list()

    count_dict = counts
    value_dict = values

    for wl in value_dict[word_count]:
        check_words.extend(wl)
    c = Counter(check_words)
    _mc_list = list()
    _count = 0
    for w, wc in c.most_common():
        if _count <= count_dict[word_count] * percentage:
            _mc_list.append((w, wc))
            _count += wc
    return _mc_list, _count, count_dict[word_count], value_dict[word_count]


def mc_words(word_lists):
    mc_words = list()
    for i in range(1, 40):
        args = most_common(word_lists, i, .8)
        args = args[0]
        for w, c in args:
            mc_words.append(w)

    return Counter(mc_words).most_common()


def clean_word_list(json_path_or_dict_or_words_list,
                    parser,
                    remove_if_all,
                    remove_if_any):
    wlf = WordList(parser=parser,
                   remove_if_all=remove_if_all,
                   remove_if_any=remove_if_any)

    if isinstance(json_path_or_dict_or_words_list, (str, dict)):
        words_list_in_list = wlf(json_path_or_dict_or_words_list)
        words_list_in_list = wlf(words_list_in_list)
    else:
        assert isinstance(json_path_or_dict_or_words_list, list)
        words_list_in_list = wlf(json_path_or_dict_or_words_list)
    return words_list_in_list


clean_up = WordList(parser=en_parser,
                    remove_if_all=remove_if_all_words,
                    remove_if_any=remove_if_any_words)


class WordContexts(object):

    def __init__(self):
        self._list_word_list = list()
        self._word_dict = dict()
        self._word_contexts = dict()

    def add_to_dictionary(self, list_of_word_list):
        if isinstance(list_of_word_list, str):
            list_of_word_list = [list_of_word_list.split()]
        self._list_word_list.extend(list_of_word_list)

        for word_list in list_of_word_list:
            for word in word_list:
                if word not in self._word_dict:
                    self._word_dict[word] = 1
                else:
                    self._word_dict[word] += 1

    def word_list(self):
        return self._list_word_list

    def word_context(self, context_word, context_window=None):
        context_window = context_window or 5
        if context_word not in self._word_contexts:
            self._word_contexts[context_word] = dict()
        if context_window not in self._word_contexts[context_word]:
            self._word_contexts[context_word][context_window] = dict()

            for word_list in self._list_word_list:
                if context_word in word_list:
                    for window in range(context_window, context_window + 1):
                        word_indexes = [idx for idx, word in enumerate(word_list) if word == context_word]
                        contexts = list()
                        for index in word_indexes:
                            contexts.append(self._context_words(index, word_list, window))
                        if any(contexts):
                            for context in contexts:
                                for key in context:
                                    if key not in self._word_contexts[context_word][context_window]:
                                        self._word_contexts[context_word][context_window][key] = list()
                                    self._word_contexts[context_word][context_window][key].extend(context[key])

        return list(self._word_contexts[context_word][context_window].values())

    @staticmethod
    def _left_context_words(word_index, word_list, context_window):
        word_index = word_index
        context_words = None
        if word_index == 0:
            return [None]
        else:
            if len(word_list[:word_index]) < context_window:
                context_words = word_list[:word_index]
            else:
                context_words = word_list[(word_index - context_window):word_index]
        return context_words

    @staticmethod
    def _right_context_words(word_index, word_list, context_window):
        word_index = word_index
        context_words = None
        if word_index == len(word_list) - 1:
            return [None]
        else:
            if len(word_list[word_index + 1:]) < context_window:
                context_words = word_list[word_index + 1:]
            else:
                context_words = word_list[word_index + 1:(word_index + 1 + context_window)]
        return context_words

    def _context_words(self, word_index, word_list, context_window):
        context = dict()
        context['left'] = list()
        context['right'] = list()

        left = self._left_context_words(word_index, word_list, context_window)
        right = self._right_context_words(word_index, word_list, context_window)

        if any(left): context['left'].append(left)
        if any(right): context['right'].append(right)
        return context

    @staticmethod
    def _keep_index_list(list_len, skip):
        return [i + skip for i in range(list_len)][::skip + 1]

    def get_skipped_words(self, contexts, skip=None):
        skip = 0 or skip
        if any(contexts):
            left, right = contexts
        else:
            left, right = [], []
        _keep_index = self._keep_index_list
        if skip:
            if any(left):
                left = [[word for idx, word in enumerate(lft_cntxt[::-1]) if
                         idx in _keep_index(len(lft_cntxt), skip) and any(word)] for lft_cntxt in left if
                        any(lft_cntxt)]
            if any(right):
                right = [[word for idx, word in enumerate(rght_cntxt) if idx in _keep_index(len(rght_cntxt), skip)] for
                         rght_cntxt in right if any(rght_cntxt)]
        return left, right

    @property
    def word_dict(self):
        return dict(sorted(self._word_dict.items(), key=lambda x: x[1], reverse=True))

    def __call__(self, context_word, context_window=None, skip=None, remove_context_word=True):
        contexts = self.word_context(context_word, context_window)

        left, right = self.get_skipped_words(contexts, skip)
        if remove_context_word:
            if any(left):
                left = [[word for word in word_list if word != context_word] for word_list in left if any(word_list)]
            if any(right):
                right = [[word for word in word_list if word != context_word] for word_list in right if any(word_list)]
        return left, right


if __name__ == '__main__':
    cProfile.run('print(clean_up([\'Hello and How!! ---   are You Man ??\'.split()*10000]))')
