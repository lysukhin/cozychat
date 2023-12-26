"""Simple tool for analyzing VK or Telegram chat contents.
VK chat's txt can be saved via Kate Mobile (https://vk.com/kate_mobile) app.
Telegram group chat's json can be exported via Telegram Desktop (https://desktop.telegram.org/) app."""

import re
import json
import datetime

from collections import defaultdict, Counter

import tqdm

from pandas import DataFrame
import seaborn

seaborn.set_palette("pastel")

import matplotlib.pyplot as plt

# https://matplotlib.org/stable/users/explain/text/text_props.html
# https://matplotlib.org/3.1.1/gallery/text_labels_and_annotations/font_family_rc_sgskip.html#configuring-the-font-family
plt.rcParams["font.family"] = "sans-serif"
# https://fonts.google.com/noto/specimen/Noto+Sans
# rm -rf ~/.cache/matplotlib
plt.rcParams["font.sans-serif"] = ["Noto Sans", "DejaVu Sans"]

# https://natasha.github.io/razdel/
from razdel import tokenize

# https://github.com/amueller/word_cloud
from wordcloud import WordCloud

from .utils import VK_NEW_MESSAGE_PATTERN_RU, VK_NEW_MESSAGE_PATTERN_EN, VK_NAME_PATTERN, VK_TIMESTAMP_PATTERN_RU, \
    VK_TIMESTAMP_PATTERN_EN, VK_MONTHS_RU, VK_MONTHS_EN, STOP_WORDS, VK_MESSAGE_STOP_WORDS, TELEGRAM_IGNORE_IDS, \
    MIN_WORD_LEN, FIGURE_SIZE, FONT_SIZE, ascii_lowercase, digits, OBSCENE_WORDS_FILTER, OBSCENE_FILTER_FALSES, \
    LETSGO_WORDS, SENTIMENT_MODEL
from .utils import mode, strip_message


class CozyChat(object):
    """TODO"""

    def __init__(self, chat_txt_path, chat_type="vk", bots_names=[]):
        self.df = self._create_dataframe(chat_txt_path, chat_type)
        self.df = self._filter_df(ignore_users=bots_names)

    def _filter_df(self, year=None, message_type="text", forwarded=False, ignore_users=[]):
        result = self.df
        if year:
            result = result[result.year == year]
        if message_type == "text":
            # we want all message types containing text
            result = result[result.text != ""]
        elif message_type:
            result = result[result.message_type == message_type]
        if forwarded is not None:
            result = result[result.forwarded == forwarded]
        if len(ignore_users) > 0:
            ignore_users = set(ignore_users)
            result = result[~result.name.isin(ignore_users)]
        return result

    def show_total_messages_per_user(self, year=None, message_type="text", forwarded=False):
        """Всего сообщений (одного типа) по людям."""
        df_filtered = self._filter_df(year, message_type, forwarded)
        forwarded_text = "forwarded" if forwarded else "original"
        print(f"Всего {message_type} {forwarded_text} сообщений за {year} год: {len(df_filtered)}")

        total_messages_per_user = df_filtered.value_counts("name")
        plt.figure(figsize=FIGURE_SIZE)
        plt.suptitle(f"Year = {year}")

        plt.subplot(1, 2, 1)
        total_messages_per_user.plot(kind="barh", ylabel="")
        plt.grid()
        plt.xlabel("Число сообщений за год")

        plt.subplot(1, 2, 2)
        total_messages_per_user.plot(kind="pie", ylabel="")

        plt.tight_layout()
        plt.show()

        return total_messages_per_user

    def show_total_messages_length_per_user(self, year=None, message_type="text", forwarded=False):
        """Длина всех сообщений (одного типа) по людям."""
        df_filtered = self._filter_df(year, message_type, forwarded)
        df_filtered["text_len"] = list(map(len, df_filtered.text))
        total_text_len_per_user = df_filtered.groupby("name").agg({"text_len": "sum"}) \
            .sort_values(by="text_len", ascending=False)

        forwarded_text = "forwarded" if forwarded else "original"
        print(
            f"Длина всех {message_type} {forwarded_text} сообщений за {year} год: {sum(total_text_len_per_user.text_len)}")

        plt.figure(figsize=FIGURE_SIZE)
        plt.suptitle(f"Year = {year}")

        plt.subplot(1, 2, 1)
        # total_text_len_per_user.plot.barh(y="text_len", ax=axes[0], ylabel="")
        # FIXME: above is not working wtf
        plt.barh(y=total_text_len_per_user.index, width=total_text_len_per_user.text_len)
        plt.grid()
        plt.xlabel("Длина сообщений за год")

        plt.subplot(1, 2, 2)
        plt.pie(x=total_text_len_per_user.text_len, labels=total_text_len_per_user.index)

        plt.tight_layout()
        plt.show()

        return total_text_len_per_user

    def show_average_messages_length_per_user(self, year=None, message_type="text", forwarded=False):
        """Средняя длина сообщения (одного типа) по людям."""
        df_filtered = self._filter_df(year, message_type, forwarded)
        df_filtered["text_len"] = list(map(len, df_filtered.text))
        mean_text_len_per_user = df_filtered.groupby("name").agg({"text_len": "mean"}) \
            .sort_values(by="text_len", ascending=False)

        forwarded_text = "forwarded" if forwarded else "original"
        print(
            f"Средняя длина сообщения {message_type} {forwarded_text} сообщений за {year} год: {df_filtered.text_len.mean()}")

        plt.figure(figsize=FIGURE_SIZE)
        plt.suptitle(f"Year = {year}")

        plt.subplot(1, 2, 1)
        plt.barh(y=mean_text_len_per_user.index, width=mean_text_len_per_user.text_len)
        plt.grid()
        plt.xlabel("Средняя длина сообщения за год")

        plt.subplot(1, 2, 2)
        plt.pie(x=mean_text_len_per_user.text_len, labels=mean_text_len_per_user.index)

        plt.tight_layout()
        plt.show()

        return mean_text_len_per_user

    def show_total_messages_per_month(self, year=None):
        """Всего сообщений по месяцам."""
        df_year = self._filter_df(year)

        total_messages_per_month = df_year.value_counts("month")

        plt.figure(figsize=FIGURE_SIZE)
        title = f"Всего сообщений по месяцам, {year}"
        plt.suptitle(title)

        xs = total_messages_per_month.index.to_list()
        ys = total_messages_per_month.to_list()
        plt.bar(xs, ys)
        plt.grid()
        plt.xticks(ticks=xs)
        plt.xlabel("Месяц")
        plt.ylabel("Число сообщений")
        plt.tight_layout()
        plt.show()

        return total_messages_per_month

    def show_total_messages_per_hour(self, year=None):
        """Всего сообщений по времени суток."""
        df_year = self._filter_df(year)

        total_messages_per_hour = df_year.value_counts("hour")

        plt.figure(figsize=FIGURE_SIZE)
        title = f"Всего сообщений по времени суток, {year}"
        plt.suptitle(title)

        xs = total_messages_per_hour.index.to_list()
        ys = total_messages_per_hour.to_list()
        plt.bar(xs, ys)
        plt.grid()
        plt.xticks(ticks=xs)
        plt.xlabel("Время суток")
        plt.ylabel("Число сообщений")
        plt.tight_layout()
        plt.show()

        return total_messages_per_hour

    def show_total_messages_per_hour_per_user(self, year=None):
        """Всего сообщений по времени суток по людям."""
        df_year = self._filter_df(year)

        groupby_user = df_year.groupby("name")
        groupby_user_hours = groupby_user["hour"]
        # Doesn't work in pandas 1.4 and higher:
        # https://pandas.pydata.org/docs/whatsnew/v1.4.0.html#backwards-incompatible-api-changes
        mode_hour_per_user = groupby_user_hours.agg(mode)

        plt.figure(figsize=FIGURE_SIZE)
        title = f"Пик активности по времени суток, {year}"
        plt.suptitle(title)

        mode_hour_per_user.plot(kind="barh", ylabel="")
        plt.grid()
        plt.xticks(ticks=range(24))
        plt.xlabel("Время суток")
        plt.tight_layout()
        plt.show()

        users = df_year["name"].unique()
        plt.figure(figsize=(FIGURE_SIZE[0], FIGURE_SIZE[1] // 2 * len(users)))
        title = f"Всего сообщений по времени суток, {year}"
        plt.suptitle(title, y=1)
        for i, user in enumerate(users, start=1):
            plt.subplot(len(users), 1, i)
            plt.title(user)

            df_year_user = df_year[df_year["name"] == user]
            total_messages_per_hour_per_user = df_year_user.value_counts("hour")

            xs = total_messages_per_hour_per_user.index.to_list()
            ys = total_messages_per_hour_per_user.to_list()
            plt.bar(xs, ys)
            plt.grid()
            plt.xticks(ticks=range(24))
            plt.xlabel("Время суток")
            plt.ylabel("Число сообщений")
        plt.tight_layout()
        plt.show()

        return mode_hour_per_user

    def _count_words(self, messages_list, stop_words=STOP_WORDS, message_stop_words=VK_MESSAGE_STOP_WORDS,
                     min_word_len=MIN_WORD_LEN, ru_only=True):
        words_counter = Counter()
        for msg in messages_list:
            words = {w.text for w in tokenize(msg)}  # 'set' for faster in/out check
            if len(message_stop_words.intersection(words)) > 0:
                continue
            for word in words:
                if word in stop_words or len(word) < min_word_len:
                    continue
                word_set = set(word)
                if ru_only and len(ascii_lowercase.intersection(word_set)) > 0:
                    continue
                if len(digits.intersection(word_set)) == len(word_set):
                    continue
                words_counter[word] += 1
        return words_counter

    def _get_wordcloud(self, words_count):
        wordcloud_gen = WordCloud(width=1600, height=900, background_color="white", colormap="tab10",
                                  relative_scaling=1.0)
        wordcloud = wordcloud_gen.generate_from_frequencies(words_count)
        return wordcloud

    def show_words_cloud(self, year=None, per_user=False):
        """Нарисовать облака популярных слов."""
        df_year = self._filter_df(year)

        if not per_user:
            messages = self._get_messages_list(df_year)
            words_count = self._count_words(messages)
            wordcloud = self._get_wordcloud(words_count)

            plt.figure(figsize=(FIGURE_SIZE[0], int(FIGURE_SIZE[1] * 1.5)))
            plt.title(f"Облако самых популярных слов, {year}", fontsize=FONT_SIZE)
            plt.imshow(wordcloud)
            plt.axis("off")
            plt.show()

            return

        users = df_year["name"].unique()
        for i, user in enumerate(users, start=1):
            plt.figure(figsize=(FIGURE_SIZE[0], int(FIGURE_SIZE[1] * 1.5)))
            title = f"Облако самых популярных слов, {year}, {user}"
            plt.title(title, fontsize=FONT_SIZE)

            messages = self._get_messages_list(df_year, user=user)
            words_count = self._count_words(messages)
            wordcloud = self._get_wordcloud(words_count)

            plt.imshow(wordcloud)
            plt.axis("off")
            plt.show()

    def _count_obscene_words_usage(self, words_count, obscene_words_filter=OBSCENE_WORDS_FILTER):
        counter = Counter()
        for word, count in words_count.items():
            for bad_word_false in OBSCENE_FILTER_FALSES:
                if bad_word_false in word:
                    break
            else:
                if obscene_words_filter.is_word_bad(word):
                    counter[word] += count
        return counter

    def show_obscene_usage(self, year=None):
        """Топ ругающихся пользователей."""
        df_year = self._filter_df(year)

        users = df_year["name"].unique()
        obscene_users_rating = {}
        for i, user in enumerate(users, start=1):

            messages = self._get_messages_list(df_year, user=user)
            words_count = self._count_words(messages)
            obscene_count = self._count_obscene_words_usage(words_count)
            obscene_rating = sum(v for k, v in obscene_count.items())
            obscene_users_rating[user] = obscene_rating

            if not obscene_count:
                print(f"Ни одного ругательного сообщения от {user} в {year} году!")
                continue

            plt.figure(figsize=(FIGURE_SIZE[0], int(FIGURE_SIZE[1] * 1.5)))
            title = f"{user} ругался(-ась) {obscene_rating} раз(а) в {year} году"
            plt.title(title, fontsize=FONT_SIZE)

            wordcloud = self._get_wordcloud(obscene_count)

            plt.imshow(wordcloud)
            plt.axis("off")
            plt.show()

        xs, ys = [], []
        for user, rating in sorted(obscene_users_rating.items(), key=lambda x: x[1])[::-1]:
            if rating == 0:
                continue
            xs.append(f"{user} ({rating})")
            ys.append(rating)

        plt.figure(figsize=FIGURE_SIZE)
        title = f"Самые матерящиеся люди, {year}"
        plt.suptitle(title, fontsize=FONT_SIZE)

        plt.subplot(1, 2, 1)
        plt.bar(xs, ys)
        plt.xticks(rotation=90)
        plt.grid()
        plt.ylabel("Число ругательств")

        plt.subplot(1, 2, 2)
        plt.pie(ys, labels=xs)

        plt.show()

    def _count_letsgo_words_usage(self, words_count, letsgo_words=LETSGO_WORDS):
        counter = Counter()
        for word, count in words_count.items():
            if word in letsgo_words:
                counter[word] += count
        return counter

    def show_letsgo_usage(self, year=None):
        """Топ проактивных пользователей."""
        df_year = self._filter_df(year)

        users = df_year["name"].unique()
        letsgo_users_rating = {}
        for i, user in enumerate(users, start=1):

            messages = self._get_messages_list(df_year, user=user)
            words_count = self._count_words(messages)
            letsgo_count = self._count_letsgo_words_usage(words_count)
            letsgo_rating = sum(v for k, v in letsgo_count.items())
            letsgo_users_rating[user] = letsgo_rating

            if not letsgo_count:
                print(f"Ни одного предложения куда-нибудь пойти от {user} в {year} году!")
                continue

            plt.figure(figsize=(FIGURE_SIZE[0], int(FIGURE_SIZE[1] * 1.5)))
            title = f"{user} звал куда-то {letsgo_rating} раз(а) в {year} году"
            plt.title(title, fontsize=FONT_SIZE)

            wordcloud = self._get_wordcloud(letsgo_count)

            plt.imshow(wordcloud)
            plt.axis("off")
            plt.show()

        xs, ys = [], []
        for user, rating in sorted(letsgo_users_rating.items(), key=lambda x: x[1])[::-1]:
            if rating == 0:
                continue
            xs.append(f"{user} ({rating})")
            ys.append(rating)

        plt.figure(figsize=FIGURE_SIZE)
        title = f"Самые про-активные люди, {year}"
        plt.suptitle(title, fontsize=FONT_SIZE)

        plt.subplot(1, 2, 1)
        plt.bar(xs, ys)
        plt.xticks(rotation=90)
        plt.grid()
        plt.ylabel("Число призывов")

        plt.subplot(1, 2, 2)
        plt.pie(ys, labels=xs)

        plt.show()

    def _compute_positive_rating(self, messages):
        rating = 0
        for msg in messages:
            num_opening_pars = msg.count('(')
            num_closing_pars = msg.count(')')
            rating += num_closing_pars - num_opening_pars
        return 1. * rating / len(messages)

    def show_positive_usage(self, year=None):
        """Топ позитивных пользователей."""
        df_year = self._filter_df(year)

        users = df_year["name"].unique()
        positive_users_rating = {}
        for i, user in enumerate(users, start=1):
            messages = self._get_messages_list(df_year, user=user)
            positive_rating = self._compute_positive_rating(messages)
            positive_users_rating[user] = positive_rating

        xs, ys = [], []
        for user, rating in sorted(positive_users_rating.items(), key=lambda x: x[1])[::-1]:
            if rating == 0:
                continue
            xs.append(f"{user} ({rating:.2f})")
            ys.append(rating)

        plt.figure(figsize=FIGURE_SIZE)
        title = f"Самые позитивные)))) люди, {year}"
        plt.suptitle(title, fontsize=FONT_SIZE)

        plt.subplot(1, 2, 1)
        plt.bar(xs, ys)
        plt.xticks(rotation=90)
        plt.grid()
        plt.ylabel("Индекс позитивности")

        # TODO pie chart cannot show negative((( people
        # plt.subplot(1, 2, 2)
        # plt.pie(ys, labels=xs)

        plt.show()

    def _compute_sentiment_ratings(self, messages, sentiment_model=SENTIMENT_MODEL):
        positive_rating, negative_rating, neutral_rating = 0, 0, 0
        results = sentiment_model.predict(messages)
        for sentiment_dict in results:
            positive_score = sentiment_dict["positive"]
            negative_score = sentiment_dict["negative"]
            neutral_score = sentiment_dict["neutral"]
            skip_score = sentiment_dict["skip"]
            if skip_score > 0.5:
                continue
            if positive_score > 0.5:
                positive_rating += 1
            if negative_score > 0.5:
                negative_rating += 1
            if neutral_score > 0.5:
                neutral_rating += 1
        positive_rating /= 1. * len(messages)
        negative_rating /= 1. * len(messages)

        return dict(positive=positive_rating, negative=negative_rating, neutral=neutral_rating)

    def show_sentiment(self, year=None):
        """Топ добрых пользователей."""
        df_year = self._filter_df(year)

        users = df_year["name"].unique()
        positive_sentiment_users_rating, negative_sentiment_users_rating = {}, {}
        for i, user in enumerate(users, start=1):
            messages = self._get_messages_list(df_year, user=user)
            sentiment_ratings = self._compute_sentiment_ratings(messages)
            positive_sentiment_users_rating[user] = sentiment_ratings["positive"]
            negative_sentiment_users_rating[user] = sentiment_ratings["negative"]

        plt.figure(figsize=FIGURE_SIZE)
        title = f"Эмоциональная окраска сообщений, {year}"
        plt.suptitle(title, fontsize=FONT_SIZE)

        plt.subplot(1, 2, 1)
        plt.title("Для положительных сообщений")
        xs, ys = [], []
        for user, rating in sorted(positive_sentiment_users_rating.items(), key=lambda x: x[1])[::-1]:
            if rating == 0:
                continue
            xs.append(f"{user} ({rating:.2f})")
            ys.append(rating)

        plt.bar(xs, ys)
        plt.xticks(rotation=90)
        plt.grid()

        plt.subplot(1, 2, 2)
        plt.title("Для отрицательных сообщений")
        xs, ys = [], []
        for user, rating in sorted(negative_sentiment_users_rating.items(), key=lambda x: x[1])[::-1]:
            if rating == 0:
                continue
            xs.append(f"{user} ({rating:.2f})")
            ys.append(rating)

        plt.bar(xs, ys)
        plt.xticks(rotation=90)
        plt.grid()

        plt.show()

    def _create_dataframe(self, chat_txt_path, chat_type="vk"):
        with open(chat_txt_path, "rt", encoding="utf-8") as fp:
            messages_str = fp.read()
        messages_list = []
        if chat_type == "vk":
            messages_list = self._vk_collect_messages(messages_str)
        elif chat_type == "telegram":
            messages_list = self._telegram_collect_messages(messages_str)
        # print(messages_list)
        df = self._build_df(messages_list)
        return df

    def _vk_parse_message_header(self, header, name_pattern=VK_NAME_PATTERN, timestamp_pattern=VK_TIMESTAMP_PATTERN_EN,
                                 months=VK_MONTHS_EN):
        header = header.strip()
        name = re.findall(name_pattern, header)[0]
        datetime_str = re.findall(timestamp_pattern, header)[0]
        if timestamp_pattern == VK_TIMESTAMP_PATTERN_RU:
            day, month, year, _, time = datetime_str.split()
        elif timestamp_pattern == VK_TIMESTAMP_PATTERN_EN:
            month, day, year, time = datetime_str.split()
        day = day.strip(',')
        month = month.strip('.').lower()
        month = months.index(month) + 1
        hours, minutes, seconds = time.split(':')
        timestamp = datetime.datetime(int(year), int(month), int(day), int(hours), int(minutes), int(seconds))

        return name, timestamp

    def _vk_collect_messages(self, messages_str, vk_new_msg_pattern=VK_NEW_MESSAGE_PATTERN_EN):
        msgs_list = []
        matches = list(re.finditer(vk_new_msg_pattern, messages_str))
        # print(len(matches))
        for i in tqdm.trange(len(matches)):
            match = matches[i]
            header_start_pos, header_end_pos = match.span()
            header = messages_str[header_start_pos:header_end_pos]
            name, timestamp = self._vk_parse_message_header(header)

            text_start_pos = header_end_pos
            if i == len(matches) - 1:
                text_end_pos = len(messages_str)
            else:
                next_match = matches[i + 1]
                text_end_pos, _ = next_match.span()

            text = messages_str[text_start_pos:text_end_pos]
            text = text.strip()
            if not text:
                continue
            message_type = "text"
            forwarded = False

            msgs_list.append((name, timestamp, text, message_type, forwarded))
        return msgs_list

    def _telegram_collect_messages(self, messages_str):
        msgs_list = []
        messages = json.loads(messages_str)['messages']
        needed_keys = {'from', 'date', 'text'}
        for i in tqdm.trange(len(messages)):
            message = messages[i]
            if needed_keys.intersection(message.keys()) != needed_keys:
                continue
            id = int(''.join(filter(str.isdigit, message['from_id'])))
            if id in TELEGRAM_IGNORE_IDS:
                continue
            name = message['from']
            if not name:
                continue
            timestamp = datetime.datetime.fromisoformat(message['date'])
            forwarded = False
            if 'forwarded_from' in message:
                forwarded = True

            text = ""
            message_type = ""

            text_src = message['text']
            if isinstance(text_src, list):
                for entity in text_src:
                    if isinstance(entity, str):
                        text += " " + entity
                    elif isinstance(entity, dict):
                        if entity["type"] == "link":
                            if "open.spotify.com" in entity["text"]:
                                message_type = "link_spotify"
                            elif "youtu" in entity["text"]:
                                message_type = "link_youtube"
            else:
                text = text_src
            text = text.strip()

            if "media_type" in message:
                if message["media_type"] == "voice_message":
                    message_type = "voice_or_round"
                elif message["media_type"] == "video_message":
                    message_type = "voice_or_round"
                elif message["media_type"] == "video_file":
                    message_type = "photo_or_video"
            elif "photo" in message:
                message_type = "photo_or_video"
            if not message_type:
                message_type = "misc"
            # print(text, message_type)
            msgs_list.append((name, timestamp, text, message_type, forwarded))
        return msgs_list

    def _build_df(self, msgs_list):
        df = DataFrame(msgs_list, columns=("name", "timestamp", "text", "message_type", "forwarded"))
        for t in ("year", "day", "month", "hour", "minute", "second"):
            df[t] = df.timestamp.map(lambda x: getattr(x, t))
        return df

    def _get_messages_list(self, df, user=None):
        if not user:
            df_user = df
        else:
            df_user = df[df["name"] == user]
        messages = [strip_message(msg) for msg in df_user["text"]]
        return messages
