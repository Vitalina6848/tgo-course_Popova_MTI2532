
"""Лабораторная работа №1.
Дополнительные задания.

Бонус 1. Триграммная языковая модель.
Бонус 2. Исследование размера словаря BPE.
Бонус 3. Наивный байесовский классификатор.
"""

import argparse
import shutil
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")


if __name__ == "__main__":

    SCRIPT_DIR = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Бонусные задания лабораторной №1"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "bonus"
    )

    args = parser.parse_args()

    OUTPUT_DIR = args.output_dir.resolve()

    with tempfile.TemporaryDirectory(
        prefix="pt01_bonus_"
    ) as temp_dir:

        LAB_WORK_DIR = Path(temp_dir)

        # Используем уже скачанный корпус,
        # если он доступен
        existing_corpus = SCRIPT_DIR / "corpus.txt"

        if existing_corpus.exists():

            shutil.copy2(
                existing_corpus,
                LAB_WORK_DIR / "corpus.txt"
            )

        # Исходная ячейка №5

        from pathlib import Path
        from urllib.request import urlretrieve
        # Ссылка на корпус из методички
        CORPUS_URL = (
            "https://raw.githubusercontent.com/nevmenandr/"
            "word2vec-russian-novels/master/books_before/WarAndPeace.txt"
        )
        # Пути к файлам первой лабораторной
        LAB_DIR = LAB_WORK_DIR
        CORPUS = LAB_DIR / "corpus.txt"
        ARTIFACTS = LAB_DIR / "artifacts"
        # Создание папки для результатов
        ARTIFACTS.mkdir(exist_ok=True)
        # Скачивание корпуса
        if not CORPUS.exists():
            urlretrieve(CORPUS_URL, CORPUS)
        print("Корпус успешно скачан")
        print("Путь к файлу:", CORPUS)
        print("Размер файла в байтах:", CORPUS.stat().st_size)

        # Исходная ячейка №6

        import re
        from collections import Counter
        # Чтение текста романа
        text = CORPUS.read_text(encoding="utf-8")
        # Приведение текста к нижнему регистру
        text = text.lower().replace("ё", "е")
        # Выделение слов по регулярному выражению
        words = re.findall(r"[а-я]+(?:-[а-я]+)*", text)
        # Разделение корпуса на 90% и 10%
        split_index = int(len(words) * 0.9)
        train_words = words[:split_index]
        val_words = words[split_index:]
        # Подсчёт частоты слов всего корпуса
        word_counts = Counter(words)
        # Размер словаря
        vocab_size = len(word_counts)
        # Двадцать самых частых слов
        top20 = word_counts.most_common(20)
        # Вывод результатов
        print("Всего слов:", len(words))
        print("Обучающая выборка:", len(train_words))
        print("Проверочная выборка:", len(val_words))
        print("Размер словаря:", vocab_size)

        print("\n20 самых частых слов:")
        for i, (word, count) in enumerate(top20, 1):
            print(f"{i:2}. {word:15} {count}")

        # Исходная ячейка №7

        # Подсчёт частоты слов обучающей выборки
        train_counts = Counter(train_words)
        # Подсчёт частоты пар соседних слов
        bigram_counts = Counter(
            zip(train_words, train_words[1:])
        )
        # Размер словаря обучающей выборки
        train_vocab_size = len(train_counts)

        print("Размер словаря всего корпуса:", vocab_size)
        print("Размер обучающего словаря:", train_vocab_size)
        print("Всего биграмм:", sum(bigram_counts.values()))
        print("Уникальных биграмм:", len(bigram_counts))

        print("\n10 самых частых биграмм:")

        for i, ((w1, w2), count) in enumerate(
            bigram_counts.most_common(10), 1
        ):
            print(f"{i:2}. {w1} + {w2}: {count}")

        # Исходная ячейка №8

        import math
        import random
        from collections import Counter, defaultdict

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

        # Создание и обучение модели
        model = BigramModel(alpha=0.01)
        model.fit(train_words)

        print("Модель успешно обучена")
        print("Размер словаря:", model.vocab_size)
        print("Коэффициент сглаживания:", model.alpha)
        print("\nПримеры вероятностей:")
        print("P(андрей | князь):",
              model.prob("князь", "андрей"))
        print("P(он | что):",
              model.prob("что", "он"))
        print("P(наполеон | князь):",
              model.prob("князь", "наполеон"))

        # Исходная ячейка №9

        # Температуры для эксперимента
        temperatures = [0.5, 1.0, 2.0]
        # Стартовые слова
        start_words = ["князь", "она"]
        # Фиксируем seed для воспроизводимости
        SEED = 42
        # Список результатов генерации
        generation = []
        # Генерация текстов
        for temperature in temperatures:
            for start in start_words:

                generated_text = model.generate(
                    start=start,
                    length=30,
                    temperature=temperature,
                    seed=SEED
                )
                # Сохранение результата
                generation.append({
                    "temperature": temperature,
                    "start": start,
                    "text": generated_text
                })
                # Вывод результата
                print(f"Температура: {temperature}")
                print(f"Стартовое слово: {start}")
                print(generated_text)
                print("Количество слов:",
                      len(generated_text.split()))
                print("-" * 60)

        # Исходная ячейка №10

        # Устанавливаем коэффициент сглаживания
        model.alpha = 0.01
        # Первые 12 слов проверочной выборки
        sentence_words = val_words[:12]
        # Создаём копию и перемешиваем слова
        shuffled_words = sentence_words.copy()
        rng = random.Random(SEED)
        rng.shuffle(shuffled_words)
        # Расчёт логарифма правдоподобия
        logprob_sentence = model.logprob(sentence_words)
        logprob_shuffled = model.logprob(shuffled_words)
        # Подсчёт пар, отсутствующих в обучающей выборке
        unseen_pairs = sum(
            1
            for v, w in zip(sentence_words, sentence_words[1:])
            if model.bigram_counts.get((v, w), 0) == 0
        )
        # Сохранение результатов для bigram.json
        likelihood = {
            "sentence": " ".join(sentence_words),
            "shuffled": " ".join(shuffled_words),
            "alpha": model.alpha,
            "logprob_sentence": logprob_sentence,
            "logprob_shuffled": logprob_shuffled,
            "unseen_pairs_in_sentence": unseen_pairs
        }
        # Вывод результатов
        print("Исходный текст:")
        print(likelihood["sentence"])

        print("\nПеремешанный текст:")
        print(likelihood["shuffled"])

        print("\nЛогарифм правдоподобия исходного текста:")
        print(round(logprob_sentence, 4))

        print("\nЛогарифм правдоподобия перемешанного текста:")
        print(round(logprob_shuffled, 4))

        print("\nКоличество ранее не встречавшихся пар:")
        print(unseen_pairs)

        print("\nИсходный текст правдоподобнее:")
        print(logprob_sentence > logprob_shuffled)

        # Исходная ячейка №11

        # Функция расчёта перплексии
        def calculate_perplexity(model, words):
            log_probability = model.logprob(words)

            # Первое слово не оценивается
            N = len(words) - 1

            return math.exp(-log_probability / N)

        # Коэффициенты сглаживания
        alphas = [1, 0.1, 0.01]

        # Словарь для сохранения результатов
        perplexity = {}

        print("Количество проверочных слов:",
              len(val_words))

        print("Количество оцениваемых слов:",
              len(val_words) - 1)

        print("\nРезультаты расчёта перплексии:")

        for alpha in alphas:
            model.alpha = alpha

            ppl = calculate_perplexity(
                model, val_words
            )

            perplexity[str(alpha)] = ppl

            print(f"alpha = {alpha}: {ppl:.4f}")

        # Определение лучшего коэффициента
        best_alpha = min(
            perplexity,
            key=perplexity.get
        )

        print("\nЛучший коэффициент сглаживания:",
              best_alpha)

        print("Минимальная перплексия:",
              round(perplexity[best_alpha], 4))

        # Возвращаем исходный коэффициент
        model.alpha = 0.01

        # Исходная ячейка №12

        import json
        # Формирование результатов биграммной модели
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
        # Путь к файлу результатов
        bigram_path = ARTIFACTS / "bigram.json"
        # Сохранение результатов в JSON
        with bigram_path.open("w", encoding="utf-8") as f:
            json.dump(
                bigram_results,
                f,
                ensure_ascii=False,
                indent=2
            )
        # Проверка созданного файла
        print("Файл bigram.json успешно создан")
        print("Путь:", bigram_path)
        print("Размер файла:", bigram_path.stat().st_size,
              "байт")
        print("\nСодержимое файла:")
        print(bigram_path.read_text(encoding="utf-8"))

        # Исходная ячейка №13

        import time
        # Маркер конца слова
        END_WORD = "</w>"

        # Функция объединения пары соседних символов
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

        # Функция обучения токенизатора BPE
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

        # Запоминаем время начала обучения
        start_time = time.perf_counter()

        # Обучаем BPE в полном режиме
        merges, bpe_vocab = train_bpe(
            words,
            num_merges=1500,
            min_freq=5
        )

        # Рассчитываем время обучения
        bpe_training_time = time.perf_counter() - start_time

        print("\nОбучение BPE завершено")
        print("Количество слияний:", len(merges))
        print("Размер словаря BPE:", len(bpe_vocab))
        print("Время обучения:",
              round(bpe_training_time, 2), "секунд")

        print("\nПервые 10 слияний:")

        for i, (a, b) in enumerate(merges[:10], 1):
            print(f"{i}. {a} + {b} = {a + b}")

        # Исходная ячейка №14

        # Создаём словарь рангов слияний
        # Чем меньше ранг, тем раньше была объединена пара
        merge_ranks = {
            pair: i for i, pair in enumerate(merges)
        }
        # Функция токенизации одного слова
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

        # Частоты слов всего корпуса
        frequencies = Counter(words)
        # Токенизация всех уникальных слов
        tokenized_words = {
            word: tokenize_bpe(word, merge_ranks)
            for word in frequencies
        }
        # Общее количество символов без маркеров конца слова
        chars = sum(
            len(word) * count
            for word, count in frequencies.items()
        )
        # Общее количество токенов BPE
        tokens_bpe = sum(
            len(tokenized_words[word]) * count
            for word, count in frequencies.items()
        )
        # Среднее количество токенов на слово
        tokens_per_word = tokens_bpe / len(words)
        # Коэффициент сжатия
        compression_vs_chars = chars / tokens_bpe
        # Статистика токенизации
        bpe_stats = {
            "chars": chars,
            "words": len(words),
            "tokens_bpe": tokens_bpe,
            "tokens_per_word": tokens_per_word,
            "compression_vs_chars": compression_vs_chars
        }

        # Примеры токенизации
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

        # Вывод результатов
        print("Статистика токенизации BPE:")
        print("Количество символов:", chars)
        print("Количество слов:", len(words))
        print("Количество токенов BPE:", tokens_bpe)
        print("Токенов на слово:",
              round(tokens_per_word, 4))
        print("Коэффициент сжатия:",
              round(compression_vs_chars, 4))

        print("\nПримеры токенизации:")

        for word, tokens in bpe_examples.items():
            print(word, "->", tokens)

        # Исходная ячейка №15

        # Формирование результатов токенизатора BPE
        bpe_results = {
            "num_merges": len(merges),
            "vocab_size": len(bpe_vocab),
            "merges": merges,
            "stats": bpe_stats,
            "examples": bpe_examples
        }

        # Путь к файлу результатов
        bpe_path = ARTIFACTS / "bpe.json"
        # Сохранение результатов в JSON
        with bpe_path.open("w", encoding="utf-8") as f:
            json.dump(
                bpe_results,
                f,
                ensure_ascii=False,
                indent=2
            )

        # Проверка созданного файла
        print("Файл bpe.json успешно создан")
        print("Путь:", bpe_path)
        print("Размер файла:", bpe_path.stat().st_size,
              "байт")

        print("\nКоличество сохранённых слияний:",
              len(merges))

        print("\nСтатистика:")
        print(json.dumps(
            bpe_stats,
            ensure_ascii=False,
            indent=2
        ))

        print("\nПримеры токенизации:")
        print(json.dumps(
            bpe_examples,
            ensure_ascii=False,
            indent=2
        ))

        # Проверка наличия обоих файлов
        print("\nФайлы в папке artifacts:")

        for path in sorted(ARTIFACTS.iterdir()):
            print(path.name, "-", path.stat().st_size, "байт")

        # Исходная ячейка №20

        # Подсчёт частоты троек соседних слов
        trigram_counts = Counter(
            zip(
                train_words,
                train_words[1:],
                train_words[2:]
            )
        )

        # Подсчёт частоты контекстов из двух слов
        trigram_context_counts = Counter(
            zip(
                train_words[:-2],
                train_words[1:-1]
            )
        )

        # Вывод общей статистики
        print("Количество обучающих слов:",
              len(train_words))

        print("Общее количество триграмм:",
              sum(trigram_counts.values()))

        print("Количество уникальных триграмм:",
              len(trigram_counts))

        print("Количество уникальных контекстов:",
              len(trigram_context_counts))

        # Вывод наиболее частых триграмм
        print("\n10 самых частых триграмм:")

        for i, ((w1, w2, w3), count) in enumerate(
            trigram_counts.most_common(10), 1
        ):
            print(f"{i:2}. {w1} + {w2} + {w3}: {count}")

        # Исходная ячейка №21

        # Триграммная языковая модель с откатом
        class TrigramModel:

            def __init__(self, bigram_model, discount=0.75):
                self.bigram = bigram_model
                self.discount = discount
            # Обучение триграммной модели
            def fit(self, words):

                self.trigram_counts = Counter(
                    zip(words, words[1:], words[2:])
                )

                self.context_counts = Counter(
                    zip(words[:-2], words[1:-1])
                )

                self.followers = defaultdict(Counter)

                for (u, v, w), count in self.trigram_counts.items():
                    self.followers[(u, v)][w] = count

                # Кэш коэффициентов отката
                self.backoff_cache = {}

                return self
            # Вероятность следующего слова
            def prob(self, u, v, w):

                context = (u, v)
                # Если контекст неизвестен, используем биграмму
                if context not in self.context_counts:
                    return self.bigram.prob(v, w)

                total = self.context_counts[context]
                count = self.trigram_counts.get((u, v, w), 0)
                # Если триграмма встречалась
                if count > 0:
                    return (count - self.discount) / total
                # Если триграмма не встречалась
                followers = self.followers[context]

                cache_key = (context, self.bigram.alpha)

                if cache_key not in self.backoff_cache:

                    reserved_mass = (
                        self.discount * len(followers) / total
                    )

                    known_mass = sum(
                        self.bigram.prob(v, token)
                        for token in followers
                    )

                    self.backoff_cache[cache_key] = (
                        reserved_mass / (1 - known_mass)
                    )

                backoff_weight = self.backoff_cache[cache_key]

                return backoff_weight * self.bigram.prob(v, w)

            # Логарифм правдоподобия текста
            def logprob(self, words):

                return sum(
                    math.log(self.prob(u, v, w))
                    for u, v, w in zip(
                        words, words[1:], words[2:]
                    )
                )
        
        # Создание и обучение модели
        trigram_model = TrigramModel(
            bigram_model=model,
            discount=0.75
        )

        trigram_model.fit(train_words)

        print("Триграммная модель успешно обучена")
        print("Коэффициент дисконтирования:",
              trigram_model.discount)

        print("Уникальных триграмм:",
              len(trigram_model.trigram_counts))

        print("\nПример известной триграммы:")

        print("Частота:",
              trigram_model.trigram_counts.get(
                  ("о", "том", "что"), 0
              ))

        print("P(что | о, том):",
              trigram_model.prob("о", "том", "что"))

        print("\nПример редкого продолжения:")

        print("Частота:",
              trigram_model.trigram_counts.get(
                  ("о", "том", "наполеон"), 0
              ))

        print("P(наполеон | о, том):",
              trigram_model.prob("о", "том", "наполеон"))

        # Проверка нормировки вероятностей
        probability_sum = sum(
            trigram_model.prob("о", "том", word)
            for word in model.vocab + ["<UNK>"]
        )

        print("\nСумма вероятностей:",
              round(probability_sum, 8))

        # Исходная ячейка №22

        # Одинаковый коэффициент сглаживания
        model.alpha = 0.01

        # Количество оцениваемых слов
        N = len(val_words) - 2

        # Логарифм правдоподобия биграммной модели
        bigram_logprob = model.logprob(val_words[1:])

        # Логарифм правдоподобия триграммной модели
        trigram_logprob = trigram_model.logprob(val_words)

        # Расчёт перплексии обеих моделей
        bigram_ppl = math.exp(-bigram_logprob / N)
        trigram_ppl = math.exp(-trigram_logprob / N)

        # Относительное снижение перплексии
        improvement = (
            (bigram_ppl - trigram_ppl)
            / bigram_ppl * 100
        )

        # Подсчёт известных триграмм
        known_trigrams = sum(
            1
            for triple in zip(
                val_words,
                val_words[1:],
                val_words[2:]
            )
            if triple in trigram_model.trigram_counts
        )

        # Количество откатов на биграммную модель
        backoff_count = N - known_trigrams

        backoff_percent = backoff_count / N * 100

        # Вывод результатов
        print("Сравнение языковых моделей")
        print("Количество оцениваемых слов:", N)

        print("\nПерплексия биграммной модели:",
              round(bigram_ppl, 4))

        print("Перплексия триграммной модели:",
              round(trigram_ppl, 4))

        print("\nСнижение перплексии (%):",
              round(improvement, 2))

        print("\nИзвестных триграмм:",
              known_trigrams)

        print("Количество откатов на биграммы:",
              backoff_count)

        print("Доля откатов (%):",
              round(backoff_percent, 2))

        if trigram_ppl < bigram_ppl:
            print("\nТриграммная модель показала лучший результат")
        elif trigram_ppl > bigram_ppl:
            print("\nБиграммная модель показала лучший результат")
        else:
            print("\nМодели показали одинаковый результат")

        # Исходная ячейка №23

        def generate_with_context(language_model, start,
                                  length=30, temperature=1.0,
                                  seed=42):

            rng = random.Random(seed)

            # Начальный контекст из двух слов
            result = start.split()
            # Словарь возможных продолжений
            candidates = model.vocab

            while len(result) < length:

                # Вероятности триграммной модели
                if isinstance(language_model, TrigramModel):

                    u, v = result[-2:]

                    probabilities = [
                        language_model.prob(u, v, w)
                        for w in candidates
                    ]

                # Вероятности биграммной модели
                else:

                    v = result[-1]

                    probabilities = [
                        language_model.prob(v, w)
                        for w in candidates
                    ]

                # Применение температуры
                weights = [
                    p ** (1 / temperature)
                    for p in probabilities
                ]

                # Случайный выбор следующего слова
                next_word = rng.choices(
                    candidates,
                    weights=weights,
                    k=1
                )[0]

                result.append(next_word)

            return " ".join(result)

        # Начальные контексты
        start_contexts = [
            "князь андрей",
            "она была"
        ]

        # Сохранение результатов эксперимента
        trigram_generation = []

        for start in start_contexts:

            bigram_text = generate_with_context(
                model,
                start=start,
                length=30,
                temperature=1.0,
                seed=42
            )

            trigram_text = generate_with_context(
                trigram_model,
                start=start,
                length=30,
                temperature=1.0,
                seed=42
            )

            trigram_generation.append({
                "start": start,
                "temperature": 1.0,
                "bigram": bigram_text,
                "trigram": trigram_text
            })

            print("Начальный контекст:", start)

            print("\nБиграммная модель:")
            print(bigram_text)
            print("Количество слов:",
                  len(bigram_text.split()))

            print("\nТриграммная модель:")
            print(trigram_text)
            print("Количество слов:",
                  len(trigram_text.split()))

            print("=" * 60)

        # Исходная ячейка №24

        # Дополнительный эксперимент с низкой температурой
        for start in start_contexts:
            bigram_text = generate_with_context(
                model,
                start=start,
                length=30,
                temperature=0.5,
                seed=42
            )

            trigram_text = generate_with_context(
                trigram_model,
                start=start,
                length=30,
                temperature=0.5,
                seed=42
            )

            trigram_generation.append({
                "start": start,
                "temperature": 0.5,
                "bigram": bigram_text,
                "trigram": trigram_text
            })

            print("Начальный контекст:", start)
            print("Температура: 0.5")

            print("\nБиграммная модель:")
            print(bigram_text)

            print("\nТриграммная модель:")
            print(trigram_text)

            print("=" * 60)

        # Исходная ячейка №25

        # Бонус №1. Сохранение результатов
        # Создаём отдельную папку для бонусных заданий
        BONUS_DIR = LAB_DIR / "bonus"
        BONUS_DIR.mkdir(exist_ok=True)
        # Формирование результатов эксперимента
        trigram_results = {
            "train_words": len(train_words),
            "val_words": len(val_words),
            "evaluated_words": len(val_words) - 2,

            "alpha": model.alpha,
            "discount": trigram_model.discount,
            "seed": SEED,

            "total_trigrams": sum(
                trigram_counts.values()
            ),

            "unique_trigrams": len(trigram_counts),

            "unique_contexts": len(
                trigram_context_counts
            ),

            "top10": trigram_counts.most_common(10),

            "perplexity": {
                "bigram": bigram_ppl,
                "trigram": trigram_ppl,
                "improvement_percent": improvement
            },

            "backoff": {
                "known_trigrams": known_trigrams,
                "backoff_count": backoff_count,
                "backoff_percent": backoff_percent
            },

            "generation": trigram_generation
        }

        # Путь к файлу
        trigram_path = BONUS_DIR / "trigram.json"

        # Сохранение результатов
        with trigram_path.open("w", encoding="utf-8") as f:
            json.dump(
                trigram_results,
                f,
                ensure_ascii=False,
                indent=2
            )

        # Проверка сохранения
        print("Бонус №1 успешно сохранён")
        print("Путь:", trigram_path)
        print("Размер файла:", trigram_path.stat().st_size,
              "байт")

        print("\nПерплексия:")
        print(json.dumps(
            trigram_results["perplexity"],
            ensure_ascii=False,
            indent=2
        ))

        print("\nКоличество пар сравнения генерации:",
              len(trigram_generation))

        print("Общее количество сгенерированных текстов:",
              len(trigram_generation) * 2)

        print("\nФайлы бонусных заданий:")

        for path in sorted(BONUS_DIR.iterdir()):
            print(path.name)

        # Исходная ячейка №26

        # Бонус №2. Исследование размера словаря BPE
        # Запоминаем время начала обучения
        bonus_bpe_start_time = time.perf_counter()
        # Обучение BPE на 5000 слияниях
        bonus_merges, bonus_bpe_vocab = train_bpe(
            words,
            num_merges=5000,
            min_freq=5
        )
        # Время обучения
        bonus_bpe_training_time = (
            time.perf_counter() - bonus_bpe_start_time
        )
        # Получаем три варианта токенизатора
        bpe_variants = {
            300: bonus_merges[:300],
            1500: bonus_merges[:1500],
            5000: bonus_merges[:5000]
        }

        print("\nОбучение BPE завершено")

        print("Количество слияний:",
              len(bonus_merges))

        print("Размер словаря BPE:",
              len(bonus_bpe_vocab))

        print("Время обучения:",
              round(bonus_bpe_training_time, 2),
              "секунд")

        print("\nПодготовленные варианты BPE:")

        for num, variant in bpe_variants.items():
            print(f"{num} слияний: {len(variant)}")

        # Проверяем воспроизводимость основного эксперимента
        print("\nСовпадение с основным BPE на 1500 слияниях:")

        print(bpe_variants[1500] == merges)

        # Исходная ячейка №27

        # Бонус №2. Сравнение статистики BPE

        frequencies = Counter(words)

        # Начальный словарь символов
        initial_alphabet = set("".join(frequencies))
        initial_alphabet.add(END_WORD)
        # Количество символов в корпусе
        total_chars = sum(
            len(word) * count
            for word, count in frequencies.items()
        )
        # Функция оценки токенизатора
        def evaluate_bpe(variant_merges):

            ranks = {
                pair: i
                for i, pair in enumerate(variant_merges)
            }

            # Формирование словаря токенов
            vocabulary = initial_alphabet.copy()

            for a, b in variant_merges:
                vocabulary.add(a + b)

            # Токенизация уникальных слов
            tokenized = {
                word: tokenize_bpe(word, ranks)
                for word in frequencies
            }

            # Общее количество токенов
            total_tokens = sum(
                len(tokenized[word]) * count
                for word, count in frequencies.items()
            )

            # Количество словоупотреблений,
            # представленных одним токеном
            whole_words = sum(
                count
                for word, count in frequencies.items()
                if len(tokenized[word]) == 1
            )

            return {
                "num_merges": len(variant_merges),
                "vocab_size": len(vocabulary),
                "tokens_bpe": total_tokens,
                "tokens_per_word": total_tokens / len(words),
                "compression_vs_chars": total_chars / total_tokens,
                "whole_words": whole_words,
                "whole_words_percent": whole_words / len(words) * 100
            }
        # Сохранение результатов сравнения
        bpe_comparison = {}

        for num, variant in bpe_variants.items():

            stats = evaluate_bpe(variant)

            bpe_comparison[num] = stats

            print(f"\nBPE: {num} слияний")
            print("Размер словаря:",
                  stats["vocab_size"])

            print("Количество токенов:",
                  stats["tokens_bpe"])

            print("Токенов на слово:",
                  round(stats["tokens_per_word"], 4))

            print("Коэффициент сжатия:",
                  round(stats["compression_vs_chars"], 4))

            print("Словоупотреблений одним токеном:",
                  stats["whole_words"])

            print("Доля целых слов (%):",
                  round(stats["whole_words_percent"], 2))

            print("-" * 50)

        # Проверка совпадения с основным экспериментом
        print("\nСовпадение статистики BPE на 1500 слияниях:")

        print(
            bpe_comparison[1500]["tokens_bpe"]
            == bpe_stats["tokens_bpe"]
        )

        # Исходная ячейка №28

        # Частоты слов корпуса
        frequencies = Counter(words)
        # Фиксируем seed для выбора примеров
        rng = random.Random(SEED)
        # Частые и знакомые нам слова
        selected_words = [
            "и",
            "князь",
            "андрей",
            "наполеон",
            "сказала",
            "неудовольствие"
        ]
        # Выбираем слова с разной частотой
        frequency_ranges = [
            (5, 9),
            (2, 4),
            (1, 1)
        ]

        for low, high in frequency_ranges:

            candidates = sorted(
                word
                for word, count in frequencies.items()
                if low <= count <= high
                and len(word) >= 8
                and word not in selected_words
            )

            # Выбираем два слова из каждой группы
            selected_words.extend(
                rng.sample(
                    candidates,
                    min(2, len(candidates))
                )
            )

        # Ранги слияний для каждого варианта BPE
        variant_ranks = {
            num: {
                pair: i
                for i, pair in enumerate(variant)
            }
            for num, variant in bpe_variants.items()
        }

        # Сохранение результатов сравнения
        bpe_example_comparison = {}

        for word in selected_words:

            print("\nСлово:", word)
            print("Частота в корпусе:", frequencies[word])
            print("Количество символов:", len(word))

            bpe_example_comparison[word] = {
                "frequency": frequencies[word],
                "length": len(word),
                "variants": {}
            }

            for num, ranks in variant_ranks.items():

                tokens = tokenize_bpe(word, ranks)

                bpe_example_comparison[word]["variants"][num] = {
                    "tokens": tokens,
                    "num_tokens": len(tokens)
                }

                print(f"\n{num} слияний:")
                print("Разбиение:", " | ".join(tokens))
                print("Количество токенов:", len(tokens))

            print("-" * 60)

        # Исходная ячейка №29

        # Бонус №2. Визуализация результатов

        import matplotlib.pyplot as plt

        # Количество слияний
        merge_numbers = sorted(bpe_comparison)
        # Функция построения и сохранения графика
        def plot_bpe_metric(metric, title, ylabel, filename):

            values = [
                bpe_comparison[num][metric]
                for num in merge_numbers
            ]

            plt.figure(figsize=(9, 5))

            plt.plot(
                merge_numbers,
                values,
                marker="o",
                linewidth=2,
                markersize=8
            )

            # Подписи значений
            for x, y in zip(merge_numbers, values):
                plt.annotate(
                    f"{y:.2f}",
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center"
                )

            plt.title(title)
            plt.xlabel("Количество слияний BPE")
            plt.ylabel(ylabel)
            plt.xticks(merge_numbers)
            plt.grid(alpha=0.3)
            plt.margins(y=0.15)
            plt.tight_layout()

            # Сохранение графика
            plt.savefig(
                BONUS_DIR / filename,
                dpi=200,
                bbox_inches="tight"
            )

            plt.show()
            plt.close()


        # График 1. Среднее количество токенов на слово
        plot_bpe_metric(
            "tokens_per_word",
            "Зависимость количества токенов на слово от числа слияний",
            "Токенов на слово",
            "bpe_tokens_per_word.png"
        )

        # График 2. Коэффициент сжатия
        plot_bpe_metric(
            "compression_vs_chars",
            "Зависимость коэффициента сжатия от числа слияний",
            "Коэффициент сжатия",
            "bpe_compression.png"
        )

        # График 3. Доля целых слов
        plot_bpe_metric(
            "whole_words_percent",
            "Доля словоупотреблений одним токеном",
            "Доля целых слов (%)",
            "bpe_whole_words.png"
        )

        # График 4. Размер словаря
        plot_bpe_metric(
            "vocab_size",
            "Рост размера словаря BPE",
            "Размер словаря",
            "bpe_vocab_size.png"
        )

        print("Все четыре графика успешно сохранены")
        print("Папка:", BONUS_DIR)

        # Исходная ячейка №30

        # Бонус №2. Сохранение результатов эксперимента
        bpe_bonus_results = {
            "corpus": CORPUS_URL,
            "total_words": len(words),
            "total_chars": total_chars,
            "min_freq": 5,
            "seed": SEED,

            "training_time_seconds": bonus_bpe_training_time,

            "comparison": bpe_comparison,

            "examples": bpe_example_comparison,

            "merges": bonus_merges,

            "reproducibility": {
                "merges_match": bpe_variants[1500] == merges,

                "statistics_match": (
                    bpe_comparison[1500]["tokens_bpe"]
                    == bpe_stats["tokens_bpe"]
                )
            }
        }

        # Путь к файлу результатов
        bpe_bonus_path = BONUS_DIR / "bpe_comparison.json"

        # Сохранение результатов
        with bpe_bonus_path.open("w", encoding="utf-8") as f:
            json.dump(
                bpe_bonus_results,
                f,
                ensure_ascii=False,
                indent=2
            )

        print("Бонус №2 успешно сохранён")
        print("Путь:", bpe_bonus_path)
        print("Размер файла:",
              bpe_bonus_path.stat().st_size, "байт")

        print("\nКоличество вариантов BPE:",
              len(bpe_comparison))

        print("Количество исследованных слов:",
              len(bpe_example_comparison))

        print("Количество сохранённых слияний:",
              len(bonus_merges))

        # Проверка сохранения графиков
        print("\nКоличество сохранённых графиков:",
              len(list(BONUS_DIR.glob("*.png"))))

        # Проверка файлов бонусных заданий
        print("\nФайлы в папке bonus:")

        for path in sorted(BONUS_DIR.iterdir()):
            print(path.name, "-", path.stat().st_size, "байт")

        # Исходная ячейка №31

        from pathlib import Path
        from urllib.request import urlretrieve

        DOSTOEVSKY_URL = (
            "https://raw.githubusercontent.com/nevmenandr/"
            "word2vec-russian-novels/master/books_before/"
            "CrimeAndPunishment.txt"
        )

        # Сохраняем второй корпус отдельно от репозитория
        DOSTOEVSKY_CORPUS = LAB_WORK_DIR / "corpus_dostoevsky.txt"

        # Скачиваем текст, если файла ещё нет
        if not DOSTOEVSKY_CORPUS.exists():
            urlretrieve(
                DOSTOEVSKY_URL,
                DOSTOEVSKY_CORPUS
            )

        # Читаем оригинальные тексты обоих авторов
        tolstoy_text = CORPUS.read_text(
            encoding="utf-8-sig"
        )

        dostoevsky_text = DOSTOEVSKY_CORPUS.read_text(
            encoding="utf-8-sig"
        )

        # Вывод результатов
        print("Оба корпуса успешно загружены")

        print("\nКорпус №1")
        print("Автор: Л. Н. Толстой")
        print("Произведение: Война и мир")
        print("Размер файла:", CORPUS.stat().st_size,
              "байт")

        print("\nКорпус №2")
        print("Автор: Ф. М. Достоевский")
        print("Произведение: Преступление и наказание")
        print("Размер файла:",
              DOSTOEVSKY_CORPUS.stat().st_size,
              "байт")

        print("\nНачало второго корпуса:")
        print(dostoevsky_text[:500])

        # Исходная ячейка №32

        # Бонус №3. Предобработка текстов
        import re
        from collections import Counter
        # Функция обработки текста
        def prepare_sentences(text, min_words=5):

            # Разделение текста на предложения
            raw_sentences = re.split(r"[.!?…]+", text)

            sentences = []

            for sentence in raw_sentences:

                # Нормализация текста
                sentence = sentence.lower().replace("ё", "е")

                # Выделение русских слов
                tokens = re.findall(
                    r"[а-я]+(?:-[а-я]+)*",
                    sentence
                )

                # Исключение коротких предложений
                if len(tokens) >= min_words:
                    sentences.append(tokens)

            return sentences


        # Обработка произведений
        tolstoy_sentences = prepare_sentences(
            tolstoy_text
        )

        dostoevsky_sentences = prepare_sentences(
            dostoevsky_text
        )

        # Разделение на обучение и проверку
        def split_sentences(sentences):

            split_index = int(len(sentences) * 0.9)

            return (
                sentences[:split_index],
                sentences[split_index:]
            )

        # Выборки Толстого
        tolstoy_train, tolstoy_val = split_sentences(
            tolstoy_sentences
        )

        # Выборки Достоевского
        dostoevsky_train, dostoevsky_val = split_sentences(
            dostoevsky_sentences
        )

        # Вывод статистики
        def print_corpus_stats(author, sentences,
                               train, val):

            print("\nАвтор:", author)

            print("Всего предложений:",
                  len(sentences))

            print("Обучающих предложений:",
                  len(train))

            print("Проверочных предложений:",
                  len(val))

            print("Слов в обучающей выборке:",
                  sum(len(s) for s in train))

            print("Слов в проверочной выборке:",
                  sum(len(s) for s in val))

            print("Доля проверочных предложений (%):",
                  round(len(val) / len(sentences) * 100, 2))

            print("\nПример обучающего предложения:")
            print(" ".join(train[0]))

            print("\nПример проверочного предложения:")
            print(" ".join(val[0]))

            print("=" * 60)


        print_corpus_stats(
            "Толстой",
            tolstoy_sentences,
            tolstoy_train,
            tolstoy_val
        )

        print_corpus_stats(
            "Достоевский",
            dostoevsky_sentences,
            dostoevsky_train,
            dostoevsky_val
        )

        # Исходная ячейка №33

        # Бонус №3. Подсчёт частот слов каждого автора

        from collections import Counter
        # Частоты слов Толстого
        tolstoy_word_counts = Counter(
            word
            for sentence in tolstoy_train
            for word in sentence
        )

        # Частоты слов Достоевского
        dostoevsky_word_counts = Counter(
            word
            for sentence in dostoevsky_train
            for word in sentence
        )

        # Общее количество обучающих слов
        tolstoy_total_words = sum(
            tolstoy_word_counts.values()
        )

        dostoevsky_total_words = sum(
            dostoevsky_word_counts.values()
        )

        # Индивидуальные словари авторов
        tolstoy_vocab = set(tolstoy_word_counts)
        dostoevsky_vocab = set(dostoevsky_word_counts)

        # Общий словарь классификатора
        nb_vocab = tolstoy_vocab | dostoevsky_vocab

        # Слова, встречающиеся у обоих авторов
        common_words = tolstoy_vocab & dostoevsky_vocab

        # Слова, встречающиеся только у одного автора
        only_tolstoy = tolstoy_vocab - dostoevsky_vocab
        only_dostoevsky = dostoevsky_vocab - tolstoy_vocab

        # Вывод общей статистики
        print("Статистика словарей авторов")

        print("\nТолстой:")
        print("Всего слов:", tolstoy_total_words)
        print("Уникальных слов:", len(tolstoy_vocab))

        print("\nДостоевский:")
        print("Всего слов:", dostoevsky_total_words)
        print("Уникальных слов:", len(dostoevsky_vocab))

        print("\nСравнение словарей:")
        print("Размер общего словаря:", len(nb_vocab))
        print("Общих слов:", len(common_words))
        print("Только у Толстого:", len(only_tolstoy))
        print("Только у Достоевского:", len(only_dostoevsky))

        # 20 самых частых слов Толстого
        print("\n20 самых частых слов Толстого:")

        for i, (word, count) in enumerate(
            tolstoy_word_counts.most_common(20), 1
        ):
            print(f"{i:2}. {word:<20} {count}")

        # 20 самых частых слов Достоевского
        print("\n20 самых частых слов Достоевского:")

        for i, (word, count) in enumerate(
            dostoevsky_word_counts.most_common(20), 1
        ):
            print(f"{i:2}. {word:<20} {count}")

        # Проверка правильности подсчёта
        print("\nПроверка количества обучающих слов:")

        print("Толстой:",
              tolstoy_total_words ==
              sum(len(s) for s in tolstoy_train))

        print("Достоевский:",
              dostoevsky_total_words ==
              sum(len(s) for s in dostoevsky_train))

        # Исходная ячейка №34

        # Бонус №3. Реализация наивного Байеса

        import math
        from collections import Counter

        class NaiveBayesClassifier:

            def __init__(self, alpha=1.0):

                self.alpha = alpha

            # Обучение классификатора
            def fit(self, training_data):

                self.authors = list(training_data)
                self.word_counts = {}
                self.total_words = {}
                self.vocab = set()

                for author, sentences in training_data.items():

                    counts = Counter(
                        word
                        for sentence in sentences
                        for word in sentence
                    )

                    self.word_counts[author] = counts

                    self.total_words[author] = sum(
                        counts.values()
                    )

                    self.vocab.update(counts)

                self.vocab_size = len(self.vocab)

                # Одинаковые априорные вероятности
                self.priors = {
                    author: 1 / len(self.authors)
                    for author in self.authors
                }

                return self

            # Вероятность слова для конкретного автора
            def word_prob(self, word, author):

                count = self.word_counts[author].get(
                    word, 0
                )

                return (
                    (count + self.alpha)
                    /
                    (
                        self.total_words[author]
                        + self.alpha * self.vocab_size
                    )
                )

            # Логарифмическая оценка предложения
            def log_score(self, sentence, author):

                score = math.log(self.priors[author])

                for word in sentence:

                    # Пропускаем слова, отсутствующие
                    # в общем обучающем словаре
                    if word in self.vocab:

                        score += math.log(
                            self.word_prob(word, author)
                        )

                return score

            # Определение наиболее вероятного автора
            def predict(self, sentence):

                scores = {
                    author: self.log_score(sentence, author)
                    for author in self.authors
                }

                return max(scores, key=scores.get)

            # Расчёт вероятностей принадлежности авторам
            def predict_proba(self, sentence):

                scores = {
                    author: self.log_score(sentence, author)
                    for author in self.authors
                }

                max_score = max(scores.values())

                probabilities = {
                    author: math.exp(score - max_score)
                    for author, score in scores.items()
                }

                total = sum(probabilities.values())

                return {
                    author: value / total
                    for author, value in probabilities.items()
                }

        # Подготовка обучающих данных
        nb_training_data = {
            "Толстой": tolstoy_train,
            "Достоевский": dostoevsky_train
        }

        # Создание и обучение классификатора
        nb_model = NaiveBayesClassifier(alpha=1.0)

        nb_model.fit(nb_training_data)

        # Проверка результатов обучения
        print("Классификатор успешно обучен")

        print("Авторы:", nb_model.authors)

        print("Размер общего словаря:",
              nb_model.vocab_size)

        print("Коэффициент сглаживания:",
              nb_model.alpha)

        print("Априорные вероятности:",
              nb_model.priors)

        # Проверка совпадения словарей
        print("\nСовпадение с предыдущим подсчётом:")
        print(nb_model.vocab == nb_vocab)

        # Примеры вероятностей отдельных слов
        print("\nВероятности слов:")

        for word in ["князь", "раскольников", "я", "и"]:

            print("\nСлово:", word)

            for author in nb_model.authors:

                print(
                    author,
                    round(nb_model.word_prob(word, author), 8)
                )

        # Проверка нормировки распределений
        print("\nПроверка суммы вероятностей:")

        for author in nb_model.authors:

            probability_sum = sum(
                nb_model.word_prob(word, author)
                for word in nb_model.vocab
            )

            print(author, round(probability_sum, 8))

        # Исходная ячейка №35

        # Бонус №3. Оценка качества классификации

        # Проверочные данные
        nb_validation_data = {
            "Толстой": tolstoy_val,
            "Достоевский": dostoevsky_val
        }

        # Сохранение результатов классификации
        nb_predictions = []

        # Классификация проверочных предложений
        for true_author, sentences in nb_validation_data.items():

            for sentence in sentences:

                predicted_author = nb_model.predict(sentence)

                nb_predictions.append({
                    "sentence": " ".join(sentence),
                    "true_author": true_author,
                    "predicted_author": predicted_author,
                    "correct": true_author == predicted_author
                })

        # Общее количество предложений
        nb_total = len(nb_predictions)

        # Количество правильных ответов
        nb_correct = sum(
            result["correct"]
            for result in nb_predictions
        )

        # Количество ошибок
        nb_incorrect = nb_total - nb_correct

        # Общая точность классификации
        nb_accuracy = nb_correct / nb_total

        print("Результаты классификации")

        print("\nВсего проверочных предложений:",
              nb_total)

        print("Правильно классифицировано:",
              nb_correct)

        print("Ошибочно классифицировано:",
              nb_incorrect)

        print("Общая точность (%):",
              round(nb_accuracy * 100, 2))

        # Оценка качества по каждому автору
        nb_author_stats = {}

        for author in nb_model.authors:

            author_results = [
                result
                for result in nb_predictions
                if result["true_author"] == author
            ]

            total = len(author_results)

            correct = sum(
                result["correct"]
                for result in author_results
            )

            accuracy = correct / total

            nb_author_stats[author] = {
                "total": total,
                "correct": correct,
                "incorrect": total - correct,
                "accuracy": accuracy
            }

            print("\nАвтор:", author)
            print("Всего предложений:", total)
            print("Правильных ответов:", correct)
            print("Ошибок:", total - correct)
            print("Доля правильных ответов (%):",
                  round(accuracy * 100, 2))

        # Сбалансированная точность
        nb_balanced_accuracy = sum(
            stats["accuracy"]
            for stats in nb_author_stats.values()
        ) / len(nb_author_stats)

        print("\nСбалансированная точность (%):",
              round(nb_balanced_accuracy * 100, 2))

        # Базовый классификатор:
        # всегда предсказывает самый частый класс
        nb_baseline_accuracy = max(
            len(sentences)
            for sentences in nb_validation_data.values()
        ) / nb_total

        print("Точность базового классификатора (%):",
              round(nb_baseline_accuracy * 100, 2))

        print("Преимущество над базовым методом (п. п.):",
              round(
                  (nb_accuracy - nb_baseline_accuracy) * 100,
                  2
              ))

        # Исходная ячейка №36

        # Бонус №3. Построение матрицы ошибок

        # Список авторов
        nb_authors = nb_model.authors

        # Создание пустой матрицы ошибок
        nb_confusion_matrix = {
            true_author: {
                predicted_author: 0
                for predicted_author in nb_authors
            }
            for true_author in nb_authors
        }

        # Заполнение матрицы результатами классификации
        for result in nb_predictions:

            true_author = result["true_author"]
            predicted_author = result["predicted_author"]

            nb_confusion_matrix[true_author][predicted_author] += 1

        # Вывод матрицы ошибок
        print("Матрица ошибок")
        print("Строки — настоящие авторы")
        print("Столбцы — предсказанные авторы")

        print()
        print(f"{'Автор':<20}", end="")

        for author in nb_authors:
            print(f"{author:>15}", end="")

        print()

        for true_author in nb_authors:

            print(f"{true_author:<20}", end="")

            for predicted_author in nb_authors:

                count = nb_confusion_matrix[
                    true_author
                ][predicted_author]

                print(f"{count:>15}", end="")

            print()

        # Проверка общего количества предложений
        matrix_total = sum(
            sum(row.values())
            for row in nb_confusion_matrix.values()
        )

        # Проверка количества правильных ответов
        matrix_correct = sum(
            nb_confusion_matrix[author][author]
            for author in nb_authors
        )

        print("\nПроверка матрицы ошибок:")

        print("Всего предложений:", matrix_total)
        print("Правильных ответов:", matrix_correct)
        print("Ошибочных ответов:",
              matrix_total - matrix_correct)

        print("\nСовпадение с предыдущими результатами:")

        print("Количество предложений:",
              matrix_total == nb_total)

        print("Правильные ответы:",
              matrix_correct == nb_correct)

        # Исходная ячейка №37

        # Бонус №3. Анализ предсказаний классификатора
        import random
        # Фиксируем seed для воспроизводимости
        nb_rng = random.Random(42)
        # Сохранение примеров для отчёта
        nb_examples = []

        for author in nb_authors:

            print("\nАвтор:", author)
            print("=" * 60)

            for is_correct in [True, False]:

                # Отбираем правильные или ошибочные ответы
                candidates = [
                    result
                    for result in nb_predictions
                    if result["true_author"] == author
                    and result["correct"] == is_correct
                ]

                # Случайно выбираем три примера
                examples = nb_rng.sample(
                    candidates,
                    min(3, len(candidates))
                )

                if is_correct:
                    print("\nПРАВИЛЬНЫЕ ПРЕДСКАЗАНИЯ")
                else:
                    print("\nОШИБОЧНЫЕ ПРЕДСКАЗАНИЯ")

                for i, result in enumerate(examples, 1):

                    sentence = result["sentence"].split()

                    # Вероятности принадлежности авторам
                    probabilities = nb_model.predict_proba(
                        sentence
                    )

                    # Сохраняем пример
                    nb_examples.append({
                        **result,
                        "probabilities": probabilities
                    })

                    print(f"\nПример №{i}")

                    print("Предложение:")
                    print(result["sentence"])

                    print("Настоящий автор:",
                          result["true_author"])

                    print("Предсказанный автор:",
                          result["predicted_author"])

                    print("Вероятности:")

                    for name, probability in probabilities.items():

                        print(
                            f"  {name}: {probability * 100:.2f}%"
                        )

                    print("-" * 60)

        print("\nВсего исследовано примеров:",
              len(nb_examples))

        # Исходная ячейка №38

        # Бонус №3. Визуализация результатов

        import matplotlib.pyplot as plt

        BONUS_DIR.mkdir(parents=True, exist_ok=True)

        # График 1. Сравнение точности

        labels = [
            "Наивный Байес",
            "Сбалансированная\nточность",
            "Базовый\nклассификатор"
        ]

        values = [
            nb_accuracy * 100,
            nb_balanced_accuracy * 100,
            nb_baseline_accuracy * 100
        ]

        plt.figure(figsize=(9, 5))

        bars = plt.bar(
            labels,
            values,
            color=["steelblue", "cornflowerblue", "gray"]
        )

        for bar, value in zip(bars, values):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                value + 1,
                f"{value:.2f}%",
                ha="center"
            )

        plt.title("Сравнение качества классификации")
        plt.ylabel("Точность (%)")
        plt.ylim(0, 100)
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        plt.savefig(
            BONUS_DIR / "nb_accuracy.png",
            dpi=200,
            bbox_inches="tight"
        )

        plt.show()
        plt.close()

        # График 2. Распознавание каждого автора

        correct_percent = [
            nb_author_stats[author]["accuracy"] * 100
            for author in nb_authors
        ]

        incorrect_percent = [
            100 - value
            for value in correct_percent
        ]

        plt.figure(figsize=(8, 5))

        plt.bar(
            nb_authors,
            correct_percent,
            color="seagreen",
            label="Правильные ответы"
        )

        plt.bar(
            nb_authors,
            incorrect_percent,
            bottom=correct_percent,
            color="lightcoral",
            label="Ошибочные ответы"
        )

        for i, value in enumerate(correct_percent):

            plt.text(
                i, value / 2,
                f"{value:.2f}%",
                ha="center",
                color="white"
            )

            plt.text(
                i, value + incorrect_percent[i] / 2,
                f"{incorrect_percent[i]:.2f}%",
                ha="center"
            )

        plt.title("Качество распознавания каждого автора")
        plt.ylabel("Доля предложений (%)")
        plt.ylim(0, 115)
        plt.legend(loc="upper center", ncol=2)
        plt.tight_layout()

        plt.savefig(
            BONUS_DIR / "nb_author_accuracy.png",
            dpi=200,
            bbox_inches="tight"
        )

        plt.show()
        plt.close()

        # График 3. Матрица ошибок

        matrix = [
            [
                nb_confusion_matrix[true_author][predicted_author]
                for predicted_author in nb_authors
            ]
            for true_author in nb_authors
        ]

        plt.figure(figsize=(7, 6))

        plt.imshow(matrix, cmap="Blues")
        plt.colorbar(label="Количество предложений")

        plt.xticks(range(len(nb_authors)), nb_authors)
        plt.yticks(range(len(nb_authors)), nb_authors)

        plt.xlabel("Предсказанный автор")
        plt.ylabel("Настоящий автор")
        plt.title("Матрица ошибок наивного Байеса")

        max_value = max(max(row) for row in matrix)

        for i in range(len(nb_authors)):
            for j in range(len(nb_authors)):

                value = matrix[i][j]

                plt.text(
                    j, i, str(value),
                    ha="center",
                    va="center",
                    fontsize=16,
                    color="white" if value > max_value / 2
                    else "black"
                )

        plt.tight_layout()

        plt.savefig(
            BONUS_DIR / "nb_confusion_matrix.png",
            dpi=200,
            bbox_inches="tight"
        )

        plt.show()
        plt.close()

        # Проверка сохранённых файлов

        print("Все три графика успешно сохранены")

        for path in sorted(BONUS_DIR.glob("nb_*.png")):
            print(path.name, "-", path.stat().st_size, "байт")

        # Исходная ячейка №39

        # Бонус №3. Сохранение результатов наивного Байеса
        import json
        nb_results = {

            # Информация о корпусах
            "corpora": {
                "Толстой": {
                    "work": "Война и мир",
                    "url": CORPUS_URL,
                    "sentences": len(tolstoy_sentences),
                    "train_sentences": len(tolstoy_train),
                    "val_sentences": len(tolstoy_val),
                    "train_words": tolstoy_total_words,
                    "vocab_size": len(tolstoy_vocab),
                    "top20": tolstoy_word_counts.most_common(20)
                },

                "Достоевский": {
                    "work": "Преступление и наказание",
                    "url": DOSTOEVSKY_URL,
                    "sentences": len(dostoevsky_sentences),
                    "train_sentences": len(dostoevsky_train),
                    "val_sentences": len(dostoevsky_val),
                    "train_words": dostoevsky_total_words,
                    "vocab_size": len(dostoevsky_vocab),
                    "top20": dostoevsky_word_counts.most_common(20)
                }
            },

            # Параметры классификатора
            "model": {
                "type": "Multinomial Naive Bayes",
                "alpha": nb_model.alpha,
                "priors": nb_model.priors,
                "vocab_size": nb_model.vocab_size,
                "min_sentence_words": 5,
                "train_ratio": 0.9,
                "validation_ratio": 0.1,
                "unknown_words": "ignored",
                "seed": 42
            },

            # Сравнение словарей
            "vocabulary": {
                "total": len(nb_vocab),
                "common": len(common_words),
                "only_tolstoy": len(only_tolstoy),
                "only_dostoevsky": len(only_dostoevsky)
            },

            # Результаты классификации
            "metrics": {
                "total": nb_total,
                "correct": nb_correct,
                "incorrect": nb_incorrect,
                "accuracy": nb_accuracy,
                "balanced_accuracy": nb_balanced_accuracy,
                "baseline_accuracy": nb_baseline_accuracy,
                "improvement_percentage_points": (
                    nb_accuracy - nb_baseline_accuracy
                ) * 100
            },

            # Статистика по каждому автору
            "author_stats": nb_author_stats,

            # Матрица ошибок
            "confusion_matrix": nb_confusion_matrix,

            # Примеры правильных и ошибочных ответов
            "examples": nb_examples,

            # Сохранённые графики
            "plots": [
                path.name
                for path in sorted(BONUS_DIR.glob("nb_*.png"))
            ]
        }

        # Путь к итоговому файлу
        nb_results_path = BONUS_DIR / "naive_bayes.json"

        # Сохранение JSON
        with nb_results_path.open("w", encoding="utf-8") as f:
            json.dump(
                nb_results,
                f,
                ensure_ascii=False,
                indent=2
            )

        # Проверка сохранённого файла
        with nb_results_path.open("r", encoding="utf-8") as f:
            saved_nb_results = json.load(f)

        print("Бонус №3 успешно сохранён")

        print("Путь:", nb_results_path)
        print("Размер файла:",
              nb_results_path.stat().st_size, "байт")

        print("\nСохранённые метрики:")

        print(json.dumps(
            saved_nb_results["metrics"],
            ensure_ascii=False,
            indent=2
        ))

        print("\nКоличество сохранённых примеров:",
              len(saved_nb_results["examples"]))

        print("Количество сохранённых графиков:",
              len(saved_nb_results["plots"]))

        print("\nПроверка сохранённой точности:",
              saved_nb_results["metrics"]["accuracy"]
              == nb_accuracy)

        print("Проверка матрицы ошибок:",
              saved_nb_results["confusion_matrix"]
              == nb_confusion_matrix)

        print("\nВсе файлы бонусных заданий:")

        for path in sorted(BONUS_DIR.iterdir()):
            print(path.name, "-", path.stat().st_size, "байт")

        print("\nВсего файлов:",
              len(list(BONUS_DIR.iterdir())))

        # Проверяем наличие результатов
        expected_files = [
            "trigram.json",
            "bpe_comparison.json",
            "bpe_compression.png",
            "bpe_tokens_per_word.png",
            "bpe_vocab_size.png",
            "bpe_whole_words.png",
            "naive_bayes.json",
            "nb_accuracy.png",
            "nb_author_accuracy.png",
            "nb_confusion_matrix.png"
        ]

        for filename in expected_files:

            if not (BONUS_DIR / filename).is_file():

                raise FileNotFoundError(filename)

        # Сохраняем итоговые результаты
        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        for filename in expected_files:

            shutil.copy2(
                BONUS_DIR / filename,
                OUTPUT_DIR / filename
            )

        print("\nВсе бонусные задания выполнены")
        print("Результаты сохранены в:", OUTPUT_DIR)
        print("Количество файлов:", len(expected_files))

