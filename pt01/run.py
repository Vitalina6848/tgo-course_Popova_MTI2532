
"""Лабораторная работа №1. Заочное отделение.
Языковая модель из счётчиков и свой токенизатор.
"""

import argparse
import json
import math
import random
import re
import time

from pathlib import Path
from urllib.request import urlretrieve
from collections import Counter, defaultdict

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
CORPUS = HERE / "corpus.txt"

CORPUS_URL = (
    "https://raw.githubusercontent.com/nevmenandr/"
    "word2vec-russian-novels/master/books_before/WarAndPeace.txt"
)

END_WORD = "</w>"
SEED = 42


class BigramModel:
    # Инициализация модели
    def __init__(self, alpha=0.01):
        self.alpha = alpha
    # Обучение модели
    def fit(self, words):
        self.word_counts = Counter(words)
        self.bigram_counts = Counter(zip(words, words[1:]))
        self.context_counts = Counter(words[:-1])

        self.vocab = sorted(self.word_counts)
        self.vocab_size = len(self.vocab)

        # Возможные продолжения каждого слова
        self.followers = defaultdict(Counter)

        for (v, w), count in self.bigram_counts.items():
            self.followers[v][w] = count

        return self

    # Вероятность со сглаживанием Лапласа
    def prob(self, v, w):
        pair_count = self.bigram_counts.get((v, w), 0)
        context_count = self.context_counts.get(v, 0)

        # Дополнительная категория для неизвестных слов
        V = self.vocab_size + 1

        return (pair_count + self.alpha) / (
            context_count + self.alpha * V
        )

    # Генерация текста с температурой
    def generate(self, start, length=30,
                 temperature=1.0, seed=42):

        if temperature <= 0:
            raise ValueError("Температура должна быть больше нуля")

        rng = random.Random(seed)
        result = [start]

        while len(result) < length:
            current = result[-1]
            candidates = self.followers.get(current)

            if not candidates:
                candidates = self.word_counts

            next_words = list(candidates)

            weights = [
                count ** (1 / temperature)
                for count in candidates.values()
            ]
            next_word = rng.choices(
                next_words, weights=weights, k=1
            )[0]

            result.append(next_word)

        return " ".join(result)

    # Логарифм правдоподобия текста
    def logprob(self, words):
        return sum(
            math.log(self.prob(v, w))
            for v, w in zip(words, words[1:])
        )

def calculate_perplexity(model, words):
    log_probability = model.logprob(words)

    # Первое слово не оценивается
    N = len(words) - 1

    return math.exp(-log_probability / N)

def merge_pair(symbols, pair):
    result = []
    i = 0

    while i < len(symbols):

        if (
            i < len(symbols) - 1
            and (symbols[i], symbols[i + 1]) == pair
        ):
            result.append(symbols[i] + symbols[i + 1])
            i += 2

        else:
            result.append(symbols[i])
            i += 1

    return result

def train_bpe(words, num_merges=1500, min_freq=5):

    # Частоты слов всего корпуса
    frequencies = Counter(words)

    # Разбиваем слова на символы
    # Редкие слова исключаем только из обучения BPE
    vocabulary = {
        word: list(word) + [END_WORD]
        for word, count in frequencies.items()
        if count >= min_freq
    }

    # Начальный словарь символов
    alphabet = set("".join(frequencies))
    alphabet.add(END_WORD)

    merges = []

    print("Уникальных слов в корпусе:",
          len(frequencies))

    print("Слов в обучающем словаре BPE:",
          len(vocabulary))

    print("Начальное количество символов:",
          len(alphabet))

    # Последовательное объединение пар
    for step in range(num_merges):

        pair_counts = Counter()

        # Подсчёт частот соседних пар
        for word, symbols in vocabulary.items():

            for pair in zip(symbols, symbols[1:]):
                pair_counts[pair] += frequencies[word]

        if not pair_counts:
            break

        # Выбираем наиболее частую пару
        best_pair = pair_counts.most_common(1)[0][0]

        # Сохраняем порядок слияний
        merges.append(best_pair)

        # Обновляем разбиение слов
        for word, symbols in vocabulary.items():
            vocabulary[word] = merge_pair(
                symbols, best_pair
            )

        # Добавляем новый токен в словарь
        alphabet.add(best_pair[0] + best_pair[1])

        # Отображение прогресса обучения
        if (step + 1) % 100 == 0:
            print(f"Выполнено слияний: "
                  f"{step + 1}/{num_merges}")

    return merges, alphabet

def tokenize_bpe(word, ranks):

    symbols = list(word) + [END_WORD]

    while len(symbols) > 1:

        # Находим пары, присутствующие в списке слияний
        ranked_pairs = [
            (ranks[pair], i)
            for i, pair in enumerate(
                zip(symbols, symbols[1:])
            )
            if pair in ranks
        ]

        # Если больше нет доступных слияний
        if not ranked_pairs:
            break
        # Выбираем пару с наименьшим рангом
        _, i = min(ranked_pairs)
        # Объединяем выбранную пару
        symbols = (
            symbols[:i]
            + [symbols[i] + symbols[i + 1]]
            + symbols[i + 2:]
        )

    return symbols



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fast",
        action="store_true",
        help="Быстрый режим для CI"
    )

    args = parser.parse_args()
    ARTIFACTS.mkdir(exist_ok=True)

    # Задание 1. Корпус и слова
    if not CORPUS.exists():
        urlretrieve(CORPUS_URL, CORPUS)

    text = CORPUS.read_text(encoding="utf-8")
    text = text.lower().replace("ё", "е")

    words = re.findall(r"[а-я]+(?:-[а-я]+)*", text)

    split_index = int(len(words) * 0.9)

    train_words = words[:split_index]
    val_words = words[split_index:]

    frequencies = Counter(words)
    top20 = frequencies.most_common(20)

    # Задание 2. Обучение биграммной модели
    model = BigramModel(alpha=0.01)
    model.fit(train_words)

    # Задание 3. Генерация текста
    generation = []

    for temperature in [0.5, 1.0, 2.0]:
        for start in ["князь", "она"]:

            generated_text = model.generate(
                start=start,
                length=30,
                temperature=temperature,
                seed=SEED
            )

            generation.append({
                "temperature": temperature,
                "start": start,
                "text": generated_text
            })

    # Задание 4. Правдоподобие текста
    sentence_words = val_words[:12]
    shuffled_words = sentence_words.copy()

    rng = random.Random(SEED)
    rng.shuffle(shuffled_words)

    likelihood = {
        "sentence": " ".join(sentence_words),
        "shuffled": " ".join(shuffled_words),
        "alpha": model.alpha,
        "logprob_sentence": model.logprob(sentence_words),
        "logprob_shuffled": model.logprob(shuffled_words),
        "unseen_pairs_in_sentence": sum(
            1
            for v, w in zip(
                sentence_words, sentence_words[1:]
            )
            if model.bigram_counts.get((v, w), 0) == 0
        )
    }

    # Задание 5. Расчёт перплексии
    perplexity = {}

    for alpha in [1, 0.1, 0.01]:
        model.alpha = alpha

        perplexity[str(alpha)] = calculate_perplexity(
            model, val_words
        )

    model.alpha = 0.01

    bigram_results = {
        "corpus": CORPUS_URL,
        "words": len(words),
        "train_words": len(train_words),
        "val_words": len(val_words),
        "vocab_size": model.vocab_size,
        "top20": top20,
        "perplexity": perplexity,
        "likelihood": likelihood,
        "generation": generation
    }

    with (ARTIFACTS / "bigram.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(
            bigram_results, f,
            ensure_ascii=False, indent=2
        )

    # Задание 6. Обучение BPE
    num_merges = 200 if args.fast else 1500

    start_time = time.perf_counter()

    merges, bpe_vocab = train_bpe(
        words,
        num_merges=num_merges,
        min_freq=5
    )

    training_time = time.perf_counter() - start_time

    merge_ranks = {
        pair: i for i, pair in enumerate(merges)
    }

    tokenized_words = {
        word: tokenize_bpe(word, merge_ranks)
        for word in frequencies
    }

    chars = sum(
        len(word) * count
        for word, count in frequencies.items()
    )

    tokens_bpe = sum(
        len(tokenized_words[word]) * count
        for word, count in frequencies.items()
    )

    bpe_stats = {
        "chars": chars,
        "words": len(words),
        "tokens_bpe": tokens_bpe,
        "tokens_per_word": tokens_bpe / len(words),
        "compression_vs_chars": chars / tokens_bpe
    }

    example_words = [
        "наполеон",
        "сказала",
        "неудовольствие",
        "андрей",
        "лошадь"
    ]

    bpe_examples = {
        word: tokenize_bpe(word, merge_ranks)
        for word in example_words
    }

    bpe_results = {
        "num_merges": len(merges),
        "vocab_size": len(bpe_vocab),
        "merges": merges,
        "stats": bpe_stats,
        "examples": bpe_examples
    }

    with (ARTIFACTS / "bpe.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(
            bpe_results, f,
            ensure_ascii=False, indent=2
        )

    print("Лабораторная работа выполнена")
    print("Всего слов:", len(words))
    print("Перплексия:", perplexity)
    print("Количество слияний:", len(merges))
    print("Размер словаря BPE:", len(bpe_vocab))
    print("Время обучения BPE:", round(training_time, 2))
    print("Статистика BPE:", bpe_stats)
    print("Результаты сохранены в:", ARTIFACTS)


if __name__ == "__main__":
    main()
