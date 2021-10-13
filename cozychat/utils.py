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

# Kate Mobile specific message parsing patterns, date format is different for different in-app's locales
VK_NAME_PATTERN = r"[А-ЯЁA-Z][а-яёa-z]+ [А-ЯЁA-Z][а-яёa-z]+"
VK_TIMESTAMP_PATTERN_RU = r"[\d]{1,2} [а-я]+. [\d]{4} г. [\d]{2}:[\d]{2}:[\d]{2}"
VK_TIMESTAMP_PATTERN_EN = r"[A-Za-z]+ [\d]{1,2}, [\d]{4} [\d]{2}:[\d]{2}:[\d]{2}"
VK_NEW_MESSAGE_PATTERN_RU = r"\n[А-ЯЁA-Z][а-яёa-z]+ [А-ЯЁA-Z][а-яёa-z]+ \([\d]{1,2} [а-я]+. [\d]{4} г. [\d]{2}:[\d]{2}:[\d]{2}\):\n"
VK_NEW_MESSAGE_PATTERN_EN = r"\n[А-ЯЁA-Z][а-яёa-z]+ [А-ЯЁA-Z][а-яёa-z]+ \([A-Za-z]+ [\d]{1,2}, [\d]{4} [\d]{2}:[\d]{2}:[\d]{2}\):\n"
VK_MONTHS_RU = ("янв", "февр", "мар", "апр", "мая", "июн", "июл", "авг", "сент", "окт", "нояб", "дек")
VK_MONTHS_EN = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")
VK_MESSAGE_STOP_WORDS = {"прикрепления", "attachments"}

# Personal manually compiled list to ignore bots
TELEGRAM_IGNORE_IDS = [1145805651, 446851068, 1311511290, 166483052, 1617463834]

STOP_WORDS = get_stop_words("ru")
MIN_WORD_LEN = 3

FIGURE_SIZE = (20, 8)
# FONT_SIZE = 20
FONT_SIZE = 40

ascii_lowercase = set(ascii_lowercase)
digits = set(digits)

OBSCENE_WORDS_FILTER = ObsceneWordsFilter(conf.bad_words_re, conf.good_words_re)
OBSCENE_FILTER_FALSES = ["команд", "дубляж"]

LETSGO_WORDS = {
    "пойдем",
    "пошли",
    "пойдемте",
    "погнали",
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
