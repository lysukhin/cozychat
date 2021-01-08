from string import ascii_lowercase, digits

from scipy.stats import mode as scipy_mode_fn

# https://pypi.org/project/stop-words/
from stop_words import get_stop_words

# https://pypi.org/project/python-obscene-words-filter/
from obscene_words_filter import conf
from obscene_words_filter.words_filter import ObsceneWordsFilter

# https://github.com/bureaucratic-labs/dostoevsky
from dostoevsky.tokenization import RegexTokenizer
from dostoevsky.models import FastTextSocialNetworkModel

NEW_MESSAGE_PATTERN = r"\n[А-ЯA-Z][а-яa-z]+ [А-ЯA-Z][а-яa-z]+ \([\d]{1,2} [а-я]+. [\d]{4} г. [\d]{2}:[\d]{2}:[\d]{2}\):\n"
MONTHS = ("янв", "февр", "мар", "апр", "мая", "июн", "июл", "авг", "сент", "окт", "нояб", "дек")

STOP_WORDS = get_stop_words("ru")
MESSAGE_STOP_WORDS = {"прикрепления"}  # attachment 
MIN_WORD_LEN = 3

FIGURE_SIZE = (20, 8)

ascii_lowercase = set(ascii_lowercase)
digits = set(digits)

OBSCENE_WORDS_FILTER = ObsceneWordsFilter(conf.bad_words_re, conf.good_words_re)
OBSCENE_FILTER_FALSES = ["команд", "дубляж"]

LETSGO_WORDS = {
    "пойдем",
    "пошли",
    "пойдемте",
    "го",
    "хочет"
}

SENTIMENT_TOKENIZER = RegexTokenizer()
SENTIMENT_MODEL = FastTextSocialNetworkModel(tokenizer=SENTIMENT_TOKENIZER)

def mode(a):
    mode_value, mode_count = scipy_mode_fn(a)
    return mode_value


def strip_message(msg):
    msg = msg.strip()
    msg = msg.replace('\n', ' ')
    msg = msg.lower()
    return msg