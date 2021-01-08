"""Simple tool for analyzing VK chat contents. Chat's txt can be downloaded via Kate Mobile (https://vk.com/kate_mobile) app."""

import re
import tqdm
import datetime

from collections import defaultdict, Counter

from pandas import DataFrame
import seaborn
seaborn.set_palette("pastel")
import matplotlib.pyplot as plt

# https://natasha.github.io/razdel/
from razdel import tokenize

# https://github.com/amueller/word_cloud
from wordcloud import WordCloud

from .utils import NEW_MESSAGE_PATTERN, MONTHS, STOP_WORDS, MESSAGE_STOP_WORDS, MIN_WORD_LEN, FIGURE_SIZE, ascii_lowercase, digits, OBSCENE_WORDS_FILTER, OBSCENE_FILTER_FALSES, LETSGO_WORDS, SENTIMENT_MODEL
from .utils import mode, strip_message


class CozyChat(object):
    """TODO"""
    def __init__(self, chat_txt_path):
        self.df = self._create_dataframe(chat_txt_path)
        
    def _filter_df_by_year(self, year):
        if year is not None:
            return self.df[self.df.year==year]
        else:
            return df
    
    def show_total_messages_per_user(self, year=None):
        """Всего сообщений по людям."""
        df_year = self._filter_df_by_year(year)
        print(f"Всего сообщений за {year} год: {len(df_year)}")

        total_messages_per_user = df_year.value_counts("name")

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
    
    def show_total_messages_per_month(self, year=None):
        """Всего сообщений по месяцам."""
        df_year = self._filter_df_by_year(year)  

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
        df_year = self._filter_df_by_year(year)  

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
        df_year = self._filter_df_by_year(year)  
 
        groupby_user = df_year.groupby("name")
        groupby_user_hours = groupby_user["hour"]
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

            df_year_user = df_year[df_year["name"]==user]
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
    
    def _count_words(self, messages_list, stop_words=STOP_WORDS, message_stop_words=MESSAGE_STOP_WORDS, min_word_len=MIN_WORD_LEN, ru_only=True):
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
        wordcloud_gen = WordCloud(width=1600, height=900, background_color="white", colormap="tab10", relative_scaling=1.0)
        wordcloud = wordcloud_gen.generate_from_frequencies(words_count)
        return wordcloud

    def show_words_cloud(self, year=None, per_user=False):
        """Нарисовать облака популярных слов."""
        df_year = self._filter_df_by_year(year)  

        if not per_user:
            messages = self._get_messages_list(df_year)
            words_count = self._count_words(messages)
            wordcloud = self._get_wordcloud(words_count)

            plt.figure(figsize=(FIGURE_SIZE[0], int(FIGURE_SIZE[1] * 1.5)))
            plt.title(f"Облако самых популярных слов, {year}", fontsize=20)
            plt.imshow(wordcloud)
            plt.axis("off")
            plt.show()

            return

        users = df_year["name"].unique()
        for i, user in enumerate(users, start=1):
            plt.figure(figsize=(FIGURE_SIZE[0], int(FIGURE_SIZE[1] * 1.5)))
            title = f"Облако самых популярных слов, {year}, {user}"
            plt.title(title, fontsize=20)

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
        df_year = self._filter_df_by_year(year)  

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
            plt.title(title, fontsize=20)

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
        plt.suptitle(title, fontsize=20)

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
        df_year = self._filter_df_by_year(year)  

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
            plt.title(title, fontsize=20)

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
        plt.suptitle(title, fontsize=20)

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
        df_year = self._filter_df_by_year(year)  

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
        plt.suptitle(title, fontsize=20)

        plt.subplot(1, 2, 1)
        plt.bar(xs, ys)
        plt.xticks(rotation=90)
        plt.grid()
        plt.ylabel("Индекс позитивности")

        plt.subplot(1, 2, 2)
        plt.pie(ys, labels=xs)

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
        df_year = self._filter_df_by_year(year)  

        users = df_year["name"].unique()
        positive_sentiment_users_rating, negative_sentiment_users_rating = {}, {}
        for i, user in enumerate(users, start=1):
            messages = self._get_messages_list(df_year, user=user)
            sentiment_ratings = self._compute_sentiment_ratings(messages)
            positive_sentiment_users_rating[user] = sentiment_ratings["positive"]
            negative_sentiment_users_rating[user] = sentiment_ratings["negative"]


        plt.figure(figsize=FIGURE_SIZE)
        title = f"Эмоцинальная окраска сообщений, {year}"
        plt.suptitle(title, fontsize=20)

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
    
    def _create_dataframe(self, chat_txt_path):
        with open(chat_txt_path, "rt", encoding="utf-8") as fp:
            messages_str = fp.read()
        messages_list = self._collect_messages(messages_str)
        df = self._build_df(messages_list)
        return df
    
    def _parse_message_header(self, header):
        header = header.strip()
        name_pattern = r"[А-ЯA-Z][а-яa-z]+ [А-ЯA-Z][а-яa-z]+"
        name = re.findall(name_pattern, header)[0]

        timestamp_pattern = r"[\d]{1,2} [а-я]+. [\d]{4} г. [\d]{2}:[\d]{2}:[\d]{2}"
        datetime_str = re.findall(timestamp_pattern, header)[0]
        day, month, year, _, time = datetime_str.split()
        month = month.strip('.')
        month = MONTHS.index(month) + 1
        hours, minutes, seconds = time.split(':')
        timestamp = datetime.datetime(int(year), int(month), int(day), int(hours), int(minutes), int(seconds))

        return name, timestamp

    def _collect_messages(self, messages_str, new_msg_pattern=NEW_MESSAGE_PATTERN):
        msgs_list = []
        i = 0
        matches = list(re.finditer(new_msg_pattern, messages_str))
        for i in tqdm.trange(len(matches)):
            match = matches[i]
            header_start_pos, header_end_pos = match.span()
            header = messages_str[header_start_pos: header_end_pos]
            name, timestamp = self._parse_message_header(header)

            text_start_pos = header_end_pos
            if i == len(matches) - 1:
                text_end_pos = len(messages_str)
            else:
                next_match = matches[i + 1]
                text_end_pos, _ = next_match.span() 

            text = messages_str[text_start_pos: text_end_pos]
            text = text.strip()
            if not text:
                continue

            msgs_list.append((name, timestamp, text))

        return msgs_list
    
    def _build_df(self, msgs_list):
        df = DataFrame(msgs_list, columns=("name", "timestamp", "text"))
        for t in ("year", "day", "month", "hour", "minute", "second"):
            df[t] = df.timestamp.map(lambda x:getattr(x, t))
        return df
    
    
    def _get_messages_list(self, df, user=None):
        if user is None:
            df_user = df
        else:
            df_user = df[df["name"]==user]
        messages = [strip_message(msg) for msg in df_user["text"]]
        return messages