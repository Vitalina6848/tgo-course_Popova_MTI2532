import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):

    def __init__(self, d_model, n_heads, context):
        super().__init__()

        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.context = context

        # Матрицы запросов, ключей и значений
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # Выходное линейное преобразование
        self.out_proj = nn.Linear(d_model, d_model)

        # Причинная маска
        mask = torch.tril(
            torch.ones(
                context,
                context,
                dtype=torch.bool
            )
        )

        self.register_buffer(
            "causal_mask",
            mask,
            persistent=False
        )

    def forward(self, x, return_attention=False):

        B, T, C = x.shape

        assert T <= self.context
        assert C == self.d_model

        # Вычисление Q, K и V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Разделение на головы внимания
        q = q.reshape(
            B, T, self.n_heads, self.head_dim
        ).transpose(1, 2)

        k = k.reshape(
            B, T, self.n_heads, self.head_dim
        ).transpose(1, 2)

        v = v.reshape(
            B, T, self.n_heads, self.head_dim
        ).transpose(1, 2)

        # Масштабированные оценки внимания
        scores = (
            q @ k.transpose(-2, -1)
        ) / math.sqrt(self.head_dim)

        # Запрет внимания к будущим токенам
        mask = self.causal_mask[:T, :T]

        scores = scores.masked_fill(
            ~mask,
            float("-inf")
        )

        # Нормализация весов внимания
        attention_weights = F.softmax(
            scores,
            dim=-1
        )

        # Взвешенная сумма значений
        output = attention_weights @ v

        # Объединение голов внимания
        output = output.transpose(
            1, 2
        ).contiguous().reshape(B, T, C)

        output = self.out_proj(output)

        if return_attention:
            return output, attention_weights

        return output

class FeedForward(nn.Module):

    def __init__(self, d_model, d_ff):
        super().__init__()

        # Расширение размерности
        self.linear1 = nn.Linear(d_model, d_ff)

        # Возвращение к исходной размерности
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):

        x = self.linear1(x)

        # Нелинейная функция активации
        x = F.gelu(x)

        x = self.linear2(x)

        return x

class TransformerBlock(nn.Module):

    def __init__(self, d_model, n_heads, d_ff, context):
        super().__init__()

        # Нормализация перед вниманием
        self.norm1 = nn.LayerNorm(d_model)

        # Многоголовое внимание
        self.attention = CausalSelfAttention(
            d_model,
            n_heads,
            context
        )

        # Нормализация перед полносвязной сетью
        self.norm2 = nn.LayerNorm(d_model)

        # Полносвязная сеть
        self.ffn = FeedForward(d_model, d_ff)

    def forward(self, x):

        # Внимание и первая остаточная связь
        x = x + self.attention(self.norm1(x))

        # Полносвязная сеть и вторая остаточная связь
        x = x + self.ffn(self.norm2(x))

        return x

class TokenPositionEmbedding(nn.Module):

    def __init__(self, vocab_size, d_model, context):
        super().__init__()

        self.context = context

        # Таблица эмбеддингов токенов
        self.token_embedding = nn.Embedding(
            vocab_size,
            d_model
        )

        # Таблица эмбеддингов позиций
        self.position_embedding = nn.Embedding(
            context,
            d_model
        )

    def forward(self, ids):

        batch_size, seq_len = ids.shape

        if seq_len > self.context:
            raise ValueError(
                "Превышена длина контекста"
            )

        # Номера позиций
        positions = torch.arange(
            seq_len,
            device=ids.device
        )

        # Эмбеддинги токенов
        token_vectors = self.token_embedding(ids)

        # Эмбеддинги позиций
        position_vectors = self.position_embedding(
            positions
        )

        # Сложение эмбеддингов
        return token_vectors + position_vectors

class TinyTransformer(nn.Module):

    def __init__(
        self,
        vocab_size,
        d_model=128,
        n_layers=4,
        n_heads=4,
        d_ff=512,
        context=128
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.context = context

        # Эмбеддинги токенов и позиций
        self.embedding = TokenPositionEmbedding(
            vocab_size,
            d_model,
            context
        )

        # Блоки трансформера
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model,
                n_heads,
                d_ff,
                context
            )
            for _ in range(n_layers)
        ])

        # Финальная нормализация
        self.final_norm = nn.LayerNorm(d_model)

        # Выходной слой
        self.lm_head = nn.Linear(
            d_model,
            vocab_size
        )

        # Инициализация параметров
        self.apply(self.init_weights)

    @staticmethod
    def init_weights(module):

        if isinstance(module, nn.Linear):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

    def forward(self, ids, targets=None):

        # Эмбеддинги
        x = self.embedding(ids)

        # Последовательное применение блоков
        for block in self.blocks:
            x = block(x)

        # Финальная нормализация
        x = self.final_norm(x)

        # Получение логитов
        logits = self.lm_head(x)

        # Вычисление функции потерь
        loss = None

        if targets is not None:

            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1)
            )

        return logits, loss





# ============================================================
# 1. Данные
# ============================================================

import re
import json
from functools import lru_cache
from pathlib import Path

import torch


class BPETokenizer:

    def __init__(self, bpe_path):

        data = json.loads(
            Path(bpe_path).read_text(encoding="utf-8")
        )

        self.merges = [
            tuple(pair) for pair in data["merges"]
        ]

        self.ranks = {
            pair: i
            for i, pair in enumerate(self.merges)
        }

        symbols = set(
            "абвгдежзийклмнопрстуфхцчшщъыьэюя-"
        )

        symbols.add("</w>")

        for a, b in self.merges:
            symbols.add(a + b)

        self.vocab = sorted(symbols)

        self.token_to_id = {
            token: i
            for i, token in enumerate(self.vocab)
        }

        self.vocab_size = len(self.vocab)

        assert self.vocab_size == data["vocab_size"]

    @staticmethod
    def normalize(text):

        text = text.lower().replace("ё", "е")

        return re.findall(
            r"[а-я]+(?:-[а-я]+)*", text
        )

    @lru_cache(maxsize=None)
    def encode_word(self, word):

        tokens = list(word) + ["</w>"]

        while len(tokens) > 1:

            pairs = list(zip(tokens, tokens[1:]))

            candidates = [
                pair for pair in pairs
                if pair in self.ranks
            ]

            if not candidates:
                break

            best = min(
                candidates,
                key=self.ranks.get
            )

            merged = []
            i = 0

            while i < len(tokens):

                if (
                    i + 1 < len(tokens)
                    and (tokens[i], tokens[i+1]) == best
                ):
                    merged.append(
                        tokens[i] + tokens[i+1]
                    )
                    i += 2

                else:
                    merged.append(tokens[i])
                    i += 1

            tokens = merged

        return tuple(
            self.token_to_id[t] for t in tokens
        )

    def encode_words(self, words):

        return [
            token_id
            for word in words
            for token_id in self.encode_word(word)
        ]

    def encode(self, text):

        return self.encode_words(
            self.normalize(text)
        )

    def decode(self, ids):

        return "".join(
            self.vocab[int(i)] for i in ids
        ).replace("</w>", " ").strip()


def load_data(corpus_path, bpe_path):

    tokenizer = BPETokenizer(bpe_path)

    text = Path(corpus_path).read_text(
        encoding="utf-8"
    )

    words = tokenizer.normalize(text)

    split = int(len(words) * 0.9)

    train_words = words[:split]
    val_words = words[split:]

    train_ids = torch.tensor(
        tokenizer.encode_words(train_words),
        dtype=torch.long
    )

    val_ids = torch.tensor(
        tokenizer.encode_words(val_words),
        dtype=torch.long
    )

    return (
        tokenizer,
        train_words,
        val_words,
        train_ids,
        val_ids
    )




# ============================================================
# 3. ОБУЧЕНИЕ
# 4. ОБРАЗЦЫ ПО ШАГАМ
# 5. ГЕНЕРАЦИЯ РУКАМИ
# ============================================================

import csv
import json
import time

import torch
import torch.nn.functional as F


def make_training_batch(data, batch_size, context, rng, device):
    starts = torch.randint(
        0,
        len(data) - context,
        (batch_size,),
        generator=rng
    )

    positions = (
        starts[:, None]
        + torch.arange(context)[None, :]
    )

    x = data[positions].to(device)
    y = data[positions + 1].to(device)

    return x, y


def transformer_loss(model, x, y):
    logits = model(x)

    if isinstance(logits, tuple):
        logits = logits[0]

    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        y.reshape(-1)
    )


@torch.no_grad()
def validation_loss(model, x, y):
    was_training = model.training
    model.eval()

    try:
        return transformer_loss(model, x, y).item()
    finally:
        model.train(was_training)


def write_json(path, data):
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


@torch.no_grad()
def generate_text(
    model,
    tokenizer,
    prompt,
    max_new_tokens,
    context=128,
    temperature=1.0,
    top_p=None,
    seed=42
):
    assert temperature >= 0

    if top_p is not None:
        assert 0 < top_p <= 1

    device = next(model.parameters()).device

    prompt_ids = list(tokenizer.encode(prompt))
    assert len(prompt_ids) > 0

    ids = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=device
    )

    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    new_ids = []

    was_training = model.training
    model.eval()

    try:
        for _ in range(max_new_tokens):

            x = ids[:, -context:]

            logits = model(x)

            if isinstance(logits, tuple):
                logits = logits[0]

            logits = logits[:, -1, :]

            if temperature == 0:

                next_id = torch.argmax(
                    logits,
                    dim=-1,
                    keepdim=True
                )

            else:

                logits = logits / temperature

                if top_p is not None:

                    sorted_logits, sorted_indices = torch.sort(
                        logits,
                        descending=True,
                        dim=-1
                    )

                    probs = F.softmax(
                        sorted_logits,
                        dim=-1
                    )

                    cumulative = torch.cumsum(
                        probs,
                        dim=-1
                    )

                    remove = cumulative > top_p
                    remove[..., 1:] = remove[..., :-1].clone()
                    remove[..., 0] = False

                    sorted_logits = sorted_logits.masked_fill(
                        remove,
                        float("-inf")
                    )

                    probs = F.softmax(
                        sorted_logits,
                        dim=-1
                    )

                    selected = torch.multinomial(
                        probs,
                        1,
                        generator=rng
                    )

                    next_id = sorted_indices.gather(
                        -1,
                        selected
                    )

                else:

                    probs = F.softmax(logits, dim=-1)

                    next_id = torch.multinomial(
                        probs,
                        1,
                        generator=rng
                    )

            new_ids.append(int(next_id.item()))

            ids = torch.cat(
                [ids, next_id],
                dim=1
            )

    finally:
        model.train(was_training)

    text = tokenizer.decode(prompt_ids + new_ids)

    return text, new_ids


def first_repeat(ids, n=4):
    seen = set()

    for i in range(len(ids) - n + 1):

        fragment = tuple(ids[i:i + n])

        if fragment in seen:
            return i

        seen.add(fragment)

    return None


def train_transformer(
    model,
    tokenizer,
    train_data,
    val_data,
    output_dir,
    checkpoint_dir,
    vocab_size,
    bpe_merges,
    fast=False,
    resume=False,
    seed=42
):
    context = 128
    lr = 0.001

    steps = 200 if fast else 2000
    batch_size = 16 if fast else 32
    interval = 25 if fast else 100

    sample_steps = (
        [0, 50, 100, 200]
        if fast
        else [0, 100, 500, 2000]
    )

    output_dir = Path(output_dir)
    checkpoint_dir = Path(checkpoint_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / "latest.pt"

    device = next(model.parameters()).device

    train_data = train_data.cpu()
    val_data = val_data.cpu()

    torch.manual_seed(seed)

    train_rng = torch.Generator().manual_seed(seed)
    val_rng = torch.Generator().manual_seed(seed + 1)
    initial_rng = torch.Generator().manual_seed(seed + 2)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr
    )

    val_x, val_y = make_training_batch(
        val_data, 20, context, val_rng, device
    )

    initial_x, initial_y = make_training_batch(
        train_data, batch_size, context,
        initial_rng, device
    )

    history = []
    samples = []

    completed_step = 0
    elapsed_seconds = 0.0

    def collect_sample(step):

        text, _ = generate_text(
            model,
            tokenizer,
            prompt="князь андрей",
            max_new_tokens=60,
            context=context,
            temperature=0.8,
            seed=seed
        )

        samples.append({
            "step": step,
            "text": text
        })

        print("Образец на шаге", step)
        print(text)

    if resume and checkpoint_path.is_file():

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True
        )

        assert checkpoint["steps"] == steps
        assert checkpoint["batch_size"] == batch_size

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        train_rng.set_state(
            checkpoint["train_rng_state"]
        )

        history = checkpoint["history"]
        samples = checkpoint["samples"]

        completed_step = checkpoint["step"]
        elapsed_seconds = checkpoint["elapsed_seconds"]

        print("Восстановление с шага:", completed_step)

    else:

        history.append({
            "step": 0,
            "train_loss": validation_loss(
                model, initial_x, initial_y
            ),
            "val_loss": validation_loss(
                model, val_x, val_y
            )
        })

        collect_sample(0)

    start_time = time.perf_counter()

    def save_checkpoint(step):

        if device.type == "cuda":
            torch.cuda.synchronize()

        state = {
            "steps": steps,
            "batch_size": batch_size,
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_rng_state": train_rng.get_state(),
            "history": history,
            "samples": samples,
            "elapsed_seconds": (
                elapsed_seconds
                + time.perf_counter()
                - start_time
            )
        }

        temp_path = checkpoint_dir / "latest.tmp"

        torch.save(state, temp_path)
        temp_path.replace(checkpoint_path)

        with open(
            output_dir / "train_curve.csv",
            "w",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "step",
                    "train_loss",
                    "val_loss"
                ]
            )

            writer.writeheader()
            writer.writerows(history)

        write_json(
            output_dir / "samples.json",
            {
                "prompt": "князь андрей",
                "temperature": 0.8,
                "samples": samples
            }
        )

        print("Контрольная точка сохранена:", step)

    if completed_step == 0:
        save_checkpoint(0)

    model.train()

    loss_sum = 0.0
    loss_count = 0

    for step in range(completed_step + 1, steps + 1):

        x, y = make_training_batch(
            train_data,
            batch_size,
            context,
            train_rng,
            device
        )

        optimizer.zero_grad(set_to_none=True)

        loss = transformer_loss(model, x, y)

        if not torch.isfinite(loss).item():
            raise RuntimeError(
                f"Некорректный loss на шаге {step}"
            )

        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        loss_count += 1

        if step % interval == 0 or step == steps:

            train_loss = loss_sum / loss_count
            val_loss = validation_loss(model, val_x, val_y)

            history.append({
                "step": step,
                "train_loss": train_loss,
                "val_loss": val_loss
            })

            print(
                f"Шаг {step}/{steps} | "
                f"Train loss: {train_loss:.4f} | "
                f"Val loss: {val_loss:.4f}",
                flush=True
            )

            loss_sum = 0.0
            loss_count = 0

        if step in sample_steps:
            collect_sample(step)

        if step % interval == 0 or step == steps:
            save_checkpoint(step)

    if device.type == "cuda":
        torch.cuda.synchronize()

    train_minutes = (
        elapsed_seconds
        + time.perf_counter()
        - start_time
    ) / 60

    config = {
        "vocab_size": vocab_size,
        "d_model": 128,
        "n_layers": 4,
        "n_heads": 4,
        "d_ff": 512,
        "context": context,
        "n_params": sum(
            p.numel() for p in model.parameters()
        ),
        "steps": steps,
        "batch": batch_size,
        "lr": lr,
        "seed": seed,
        "train_tokens": len(train_data),
        "val_tokens": len(val_data),
        "bpe_merges": bpe_merges,
        "train_minutes": train_minutes,
        "device": str(device)
    }

    write_json(
        output_dir / "tiny_config.json",
        config
    )

    return history, samples, config


def run_generation_experiment(
    model,
    tokenizer,
    output_dir,
    seed=42,
    fast=False
):
    prompt = "князь андрей"
    max_new_tokens = 40 if fast else 80

    modes = [
        ("greedy", 0.0, None),
        ("t0.7", 0.7, None),
        ("t1.5", 1.5, None),
        ("top_p0.9", 1.0, 0.9)
    ]

    runs = []

    for name, temperature, top_p in modes:

        text, ids = generate_text(
            model,
            tokenizer,
            prompt,
            max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed
        )

        runs.append({
            "name": name,
            "temperature": temperature,
            "top_p": top_p,
            "text": text,
            "ids": ids,
            "n_tokens": len(ids),
            "loop_start": first_repeat(ids) if name == "greedy" else None
        })

        print("Режим:", name)
        print(text)

    result = {
        "model": "tiny",
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "runs": runs
    }

    write_json(
        Path(output_dir) / "generation.json",
        result
    )

    return result




# ============================================================
# 6. СЧЁТЧИКИ ПРОТИВ ТРАНСФОРМЕРА
# ============================================================

import math
import json

from pathlib import Path
from collections import Counter

import torch
import torch.nn.functional as F


class ComparisonBigram:

    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def fit(self, words):

        self.word_counts = Counter(words)

        self.bigram_counts = Counter(
            zip(words, words[1:])
        )

        self.context_counts = Counter(
            words[:-1]
        )

        self.vocab = sorted(self.word_counts)
        self.vocab_size = len(self.vocab)

        return self

    def prob(self, previous, current):

        pair_count = self.bigram_counts.get(
            (previous, current), 0
        )

        context_count = self.context_counts.get(
            previous, 0
        )

        V = self.vocab_size + 1

        return (
            pair_count + self.alpha
        ) / (
            context_count + self.alpha * V
        )

    def logprob(self, words):

        return sum(
            math.log(self.prob(previous, current))
            for previous, current
            in zip(words, words[1:])
        )


def evaluate_bigram(
    train_words,
    comparison_words,
    alpha=0.01
):

    model = ComparisonBigram(
        alpha=alpha
    ).fit(train_words)

    nats_total = -model.logprob(
        comparison_words
    )

    nats_per_word = (
        nats_total / len(comparison_words)
    )

    result = {
        "tokens": len(comparison_words) - 1,
        "nats_total": nats_total,
        "nats_per_word": nats_per_word,
        "ppl_per_word": math.exp(nats_per_word),
        "alpha": alpha
    }

    return result


@torch.no_grad()
def evaluate_tiny(
    model,
    tokenizer,
    comparison_words,
    context=128,
    batch_size=32
):

    device = next(model.parameters()).device

    text = " ".join(comparison_words)

    token_ids = tokenizer.encode(text)

    ids = torch.tensor(
        token_ids,
        dtype=torch.long
    )

    assert len(ids) >= 2

    was_training = model.training
    model.eval()

    nats_total = 0.0
    evaluated_tokens = 0

    try:

        # Начальная часть последовательности.
        # Первый токен используется как контекст.

        prefix_length = min(
            context,
            len(ids) - 1
        )

        x = ids[:prefix_length].unsqueeze(0).to(device)

        y = ids[1:prefix_length + 1].to(device)

        logits = model(x)

        if isinstance(logits, tuple):
            logits = logits[0]

        losses = F.cross_entropy(
            logits[0].float(),
            y,
            reduction="none"
        )

        nats_total += losses.double().sum().item()

        evaluated_tokens += prefix_length

        # Оценка остальных токенов.
        # Для каждого используется до 128
        # предыдущих токенов.

        for start in range(
            prefix_length + 1,
            len(ids),
            batch_size
        ):

            end = min(
                start + batch_size,
                len(ids)
            )

            positions = torch.arange(start, end)

            offsets = torch.arange(
                -context, 0
            )

            indices = (
                positions[:, None]
                + offsets[None, :]
            )

            x = ids[indices].to(device)

            y = ids[positions].to(device)

            logits = model(x)

            if isinstance(logits, tuple):
                logits = logits[0]

            last_logits = logits[:, -1, :].float()

            losses = F.cross_entropy(
                last_logits,
                y,
                reduction="none"
            )

            nats_total += (
                losses.double().sum().item()
            )

            evaluated_tokens += len(y)

    finally:

        model.train(was_training)

    assert evaluated_tokens == len(ids) - 1

    nats_per_word = (
        nats_total / len(comparison_words)
    )

    result = {
        "tokens": evaluated_tokens,
        "nats_total": nats_total,
        "nats_per_word": nats_per_word,
        "ppl_per_word": math.exp(nats_per_word)
    }

    return result


def run_evaluation(
    model,
    tokenizer,
    train_words,
    val_words,
    output_dir,
    vocab_size,
    n_words=2000
):

    assert len(val_words) >= n_words

    comparison_words = val_words[:n_words]

    print("СРАВНЕНИЕ ЯЗЫКОВЫХ МОДЕЛЕЙ")
    print("=" * 60)

    print(
        "Количество слов:",
        len(comparison_words)
    )

    bigram_result = evaluate_bigram(
        train_words,
        comparison_words,
        alpha=0.01
    )

    tiny_result = evaluate_tiny(
        model,
        tokenizer,
        comparison_words,
        context=128
    )

    result = {
        "words": len(comparison_words),
        "bigram": bigram_result,
        "tiny": tiny_result
    }

    for name in ("bigram", "tiny"):

        metrics = result[name]

        print()
        print("Модель:", name)

        print(
            "Оценённых токенов:",
            metrics["tokens"]
        )

        print(
            "Nats total:",
            metrics["nats_total"]
        )

        print(
            "Nats per word:",
            metrics["nats_per_word"]
        )

        print(
            "Perplexity per word:",
            metrics["ppl_per_word"]
        )

        assert metrics["tokens"] > 0

        assert math.isfinite(
            metrics["nats_per_word"]
        )

        assert math.isfinite(
            metrics["ppl_per_word"]
        )

    uniform = (
        math.log(vocab_size)
        * tiny_result["tokens"]
        / len(comparison_words)
    )

    better_than_uniform = (
        tiny_result["nats_per_word"] < uniform
    )

    print()
    print("Равновероятный выбор:", uniform)

    print(
        "Трансформер лучше равновероятного выбора:",
        better_than_uniform
    )

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = output_dir / "eval.json"

    output_path.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print("Результаты сохранены:", output_path)

    return result




# ============================================================
# БОНУС №1. ТРЕТЬЯ ТОЧКА
# БОНУС №2. ГЕНЕРАЦИЯ QWEN И KV-КЭШ
# ============================================================

import csv
import json
import math
import time

from pathlib import Path

import torch
import torch.nn.functional as F


QWEN_MODEL_NAME = "Qwen/Qwen3-0.6B-Base"


def load_qwen_for_run(model_dir=None, device=None):

    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM
    )

    if device is None:

        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    device = torch.device(device)

    source = QWEN_MODEL_NAME

    if model_dir is not None:

        model_dir = Path(model_dir)

        if (model_dir / "config.json").is_file():
            source = str(model_dir)

    dtype = (
        torch.float16
        if device.type == "cuda"
        else torch.float32
    )

    tokenizer = AutoTokenizer.from_pretrained(
        source
    )

    model = AutoModelForCausalLM.from_pretrained(
        source,
        dtype=dtype
    )

    model = model.to(device)
    model.eval()

    print("Модель:", QWEN_MODEL_NAME)
    print("Устройство:", device)

    print(
        "Количество параметров:",
        sum(p.numel() for p in model.parameters())
    )

    return model, tokenizer


@torch.inference_mode()
def evaluate_qwen(
    model,
    tokenizer,
    comparison_words,
    chunk_size=256
):

    device = next(model.parameters()).device

    text = " ".join(comparison_words)

    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False
    )

    assert len(token_ids) >= 2

    ids = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=device
    )

    model.eval()

    nats_total = 0.0
    evaluated_tokens = 0

    past = None

    for start in range(
        0,
        ids.size(1) - 1,
        chunk_size
    ):

        end = min(
            start + chunk_size,
            ids.size(1) - 1
        )

        x = ids[:, start:end]

        y = ids[:, start + 1:end + 1]

        output = model(
            input_ids=x,
            past_key_values=past,
            use_cache=True
        )

        logits = output.logits.float()

        losses = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            reduction="none"
        )

        nats_total += (
            losses.double().sum().item()
        )

        evaluated_tokens += y.numel()

        past = output.past_key_values

        del output, logits, losses

    assert evaluated_tokens == len(token_ids) - 1

    nats_per_word = (
        nats_total / len(comparison_words)
    )

    result = {
        "model": QWEN_MODEL_NAME,
        "tokens": evaluated_tokens,
        "nats_total": nats_total,
        "nats_per_word": nats_per_word,
        "ppl_per_word": math.exp(nats_per_word)
    }

    print("ОЦЕНКА QWEN")

    for key, value in result.items():
        print(key, ":", value)

    return result


# ============================================================
# САМОСТОЯТЕЛЬНАЯ ЖАДНАЯ ГЕНЕРАЦИЯ
# ============================================================

@torch.inference_mode()
def qwen_greedy_ids(
    model,
    prompt_ids,
    max_new_tokens=30,
    use_cache=True
):

    device = next(model.parameters()).device

    ids = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=device
    )

    model.eval()

    past = None
    new_ids = []

    for step in range(max_new_tokens):

        if use_cache and past is not None:

            x = ids[:, -1:]

        else:

            x = ids

        output = model(
            input_ids=x,
            past_key_values=(
                past if use_cache else None
            ),
            use_cache=use_cache
        )

        logits = output.logits[:, -1, :]

        next_id = torch.argmax(
            logits,
            dim=-1,
            keepdim=True
        )

        if use_cache:

            past = output.past_key_values

        new_ids.append(
            int(next_id.item())
        )

        ids = torch.cat(
            [ids, next_id],
            dim=1
        )

    return new_ids


# ============================================================
# СРАВНЕНИЕ С HUGGING FACE
# ============================================================

@torch.inference_mode()
def qwen_greedy_check(
    model,
    tokenizer,
    prompt="Князь Андрей вошел в комнату и",
    max_new_tokens=30
):

    device = next(model.parameters()).device

    prompt_ids = tokenizer.encode(
        prompt,
        add_special_tokens=False
    )

    ours_ids = qwen_greedy_ids(
        model,
        prompt_ids,
        max_new_tokens=max_new_tokens,
        use_cache=True
    )

    ids = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=device
    )

    output = model.generate(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id
    )

    hf_ids = output[
        0,
        ids.size(1):
    ].tolist()

    match = ours_ids == hf_ids

    result = {
        "model": QWEN_MODEL_NAME,
        "prompt": prompt,
        "prompt_ids": prompt_ids,
        "ours_ids": ours_ids,
        "hf_ids": hf_ids,
        "match": match
    }

    print("СРАВНЕНИЕ ГЕНЕРАЦИИ")

    print(
        "Самостоятельная генерация:",
        tokenizer.decode(
            prompt_ids + ours_ids
        )
    )

    print(
        "Генерация Hugging Face:",
        tokenizer.decode(
            prompt_ids + hf_ids
        )
    )

    print("Совпадение токенов:", match)

    assert match
    assert len(ours_ids) == max_new_tokens

    return result


# ============================================================
# ИЗМЕРЕНИЕ СКОРОСТИ С KV-КЭШЕМ И БЕЗ НЕГО
# ============================================================

def qwen_sync(device):

    if device.type == "cuda":

        torch.cuda.synchronize(device)


def qwen_speed_test(
    model,
    tokenizer,
    prompt="Князь Андрей вошел в комнату и",
    max_new_tokens=30,
    repeats=3
):

    device = next(model.parameters()).device

    prompt_ids = tokenizer.encode(
        prompt,
        add_special_tokens=False
    )

    # Прогрев обоих режимов

    for use_cache in (True, False):

        qwen_greedy_ids(
            model,
            prompt_ids,
            max_new_tokens=max_new_tokens,
            use_cache=use_cache
        )

    timings = {
        "cache": [],
        "no_cache": []
    }

    generated = {}

    # Чередуем порядок режимов,
    # чтобы уменьшить влияние прогрева.

    for repeat in range(repeats):

        modes = (
            [("cache", True), ("no_cache", False)]
            if repeat % 2 == 0
            else [("no_cache", False), ("cache", True)]
        )

        for mode, use_cache in modes:

            qwen_sync(device)

            start = time.perf_counter()

            result_ids = qwen_greedy_ids(
                model,
                prompt_ids,
                max_new_tokens=max_new_tokens,
                use_cache=use_cache
            )

            qwen_sync(device)

            seconds = (
                time.perf_counter() - start
            )

            assert seconds > 0

            timings[mode].append(seconds)

            generated[mode] = result_ids

            print(
                mode,
                "| Повтор:", repeat + 1,
                "| Время:", round(seconds, 4),
                "| Токенов/сек:",
                round(max_new_tokens / seconds, 4)
            )

    assert (
        generated["cache"]
        == generated["no_cache"]
    )

    rows = []

    for mode in ("cache", "no_cache"):

        seconds = (
            sum(timings[mode])
            / len(timings[mode])
        )

        rows.append({
            "mode": mode,
            "new_tokens": max_new_tokens,
            "seconds": seconds,
            "tokens_per_second":
                max_new_tokens / seconds
        })

    print("СРАВНЕНИЕ СКОРОСТИ")

    for row in rows:
        print(row)

    speedup = (
        rows[0]["tokens_per_second"]
        / rows[1]["tokens_per_second"]
    )

    print("Коэффициент ускорения:", speedup)

    print(
        "Генерация с кэшем быстрее:",
        speedup > 1
    )

    return rows


# ============================================================
# ВЫПОЛНЕНИЕ БОНУСОВ №1 И №2
# ============================================================

def run_qwen_bonuses(
    val_words,
    output_dir,
    model_dir=None,
    device=None
):

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    eval_path = output_dir / "eval.json"

    evaluation = json.loads(
        eval_path.read_text(
            encoding="utf-8"
        )
    )

    comparison_words = val_words[
        :evaluation["words"]
    ]

    assert (
        len(comparison_words)
        == evaluation["words"]
    )

    # Загрузка Qwen для оценки
    # и измерения скорости.

    model, tokenizer = load_qwen_for_run(
        model_dir=model_dir,
        device=device
    )

    qwen_result = evaluate_qwen(
        model,
        tokenizer,
        comparison_words
    )

    speed_rows = qwen_speed_test(
        model,
        tokenizer,
        max_new_tokens=30,
        repeats=3
    )

    # Автотест преподавателя проверяет
    # генерацию на CPU с float32.

    if next(model.parameters()).device.type == "cuda":

        del model

        torch.cuda.empty_cache()

        model, tokenizer = load_qwen_for_run(
            model_dir=model_dir,
            device="cpu"
        )

    greedy_result = qwen_greedy_check(
        model,
        tokenizer
    )

    evaluation["qwen"] = qwen_result

    eval_path.write_text(
        json.dumps(
            evaluation,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    greedy_path = (
        output_dir / "greedy_check.json"
    )

    greedy_path.write_text(
        json.dumps(
            greedy_result,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    speed_path = output_dir / "speed.csv"

    with open(
        speed_path,
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "mode",
                "new_tokens",
                "seconds",
                "tokens_per_second"
            ]
        )

        writer.writeheader()
        writer.writerows(speed_rows)

    print("РЕЗУЛЬТАТЫ БОНУСОВ")

    print("eval.json:", eval_path.is_file())
    print("greedy_check.json:", greedy_path.is_file())
    print("speed.csv:", speed_path.is_file())

    return evaluation, greedy_result, speed_rows




# ============================================================
# БОНУС №3. ЗАКОН МАСШТАБИРОВАНИЯ РУКАМИ
# ============================================================

import csv
import json
import math
import time

from pathlib import Path

import torch


def train_scaling_model(
    width,
    vocab_size,
    train_data,
    val_data,
    checkpoint_dir,
    device,
    steps=2000,
    batch_size=32,
    interval=100,
    seed=42,
    resume=False
):

    context = 128
    lr = 0.001
    n_layers = 4
    n_heads = 4
    d_ff = width * 4

    device = torch.device(device)

    train_data = train_data.cpu()
    val_data = val_data.cpu()

    torch.manual_seed(seed)

    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    # Одинаковая последовательность обучающих окон

    train_rng = torch.Generator().manual_seed(seed)

    val_rng = torch.Generator().manual_seed(seed + 1)

    initial_rng = torch.Generator().manual_seed(seed + 2)

    # Создание модели заданной ширины

    model = TinyTransformer(
        vocab_size=vocab_size,
        d_model=width,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        context=context
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr
    )

    n_params = sum(
        p.numel() for p in model.parameters()
    )

    # Одинаковые проверочные окна

    val_x, val_y = make_training_batch(
        val_data,
        20,
        context,
        val_rng,
        device
    )

    initial_x, initial_y = make_training_batch(
        train_data,
        batch_size,
        context,
        initial_rng,
        device
    )

    checkpoint_dir = Path(checkpoint_dir)

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint_path = (
        checkpoint_dir / f"width_{width}.pt"
    )

    history = []
    completed_step = 0
    previous_seconds = 0.0

    settings = {
        "width": width,
        "vocab_size": vocab_size,
        "steps": steps,
        "batch": batch_size,
        "interval": interval,
        "context": context,
        "lr": lr,
        "seed": seed
    }

    print()
    print("=" * 60)
    print("ОБУЧЕНИЕ МОДЕЛИ ШИРИНОЙ", width)
    print("=" * 60)

    print("Количество параметров:", n_params)

    # Восстановление обучения

    if resume and checkpoint_path.is_file():

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True
        )

        assert checkpoint["settings"] == settings

        model.load_state_dict(
            checkpoint["model_state"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state"]
        )

        train_rng.set_state(
            checkpoint["generator_state"]
        )

        history = checkpoint["history"]
        completed_step = checkpoint["step"]
        previous_seconds = checkpoint["seconds"]

        del checkpoint

        print("Восстановлен шаг:", completed_step)

    else:

        initial_train_loss = validation_loss(
            model,
            initial_x,
            initial_y
        )

        initial_val_loss = validation_loss(
            model,
            val_x,
            val_y
        )

        history.append({
            "step": 0,
            "train_loss": initial_train_loss,
            "val_loss": initial_val_loss
        })

        print(
            f"Шаг 0 | "
            f"Train loss: {initial_train_loss:.4f} | "
            f"Val loss: {initial_val_loss:.4f}"
        )

    # Обучение

    model.train()

    start_time = time.perf_counter()

    loss_sum = 0.0
    loss_count = 0

    for step in range(completed_step + 1, steps + 1):

        x, y = make_training_batch(
            train_data,
            batch_size,
            context,
            train_rng,
            device
        )

        optimizer.zero_grad(set_to_none=True)

        loss = transformer_loss(model, x, y)

        if not torch.isfinite(loss).item():

            raise RuntimeError(
                f"Некорректный loss на шаге {step}"
            )

        loss.backward()

        optimizer.step()

        loss_sum += loss.item()
        loss_count += 1

        if step % interval == 0 or step == steps:

            train_loss = loss_sum / loss_count

            val_loss = validation_loss(
                model,
                val_x,
                val_y
            )

            history.append({
                "step": step,
                "train_loss": train_loss,
                "val_loss": val_loss
            })

            print(
                f"Ширина {width} | "
                f"Шаг {step}/{steps} | "
                f"Train loss: {train_loss:.4f} | "
                f"Val loss: {val_loss:.4f}",
                flush=True
            )

            loss_sum = 0.0
            loss_count = 0

            if device.type == "cuda":
                torch.cuda.synchronize()

            # Сохранение контрольной точки

            checkpoint_data = {
                "settings": settings,
                "step": step,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "generator_state": train_rng.get_state(),
                "history": history,
                "seconds": (
                    previous_seconds
                    + time.perf_counter()
                    - start_time
                )
            }

            temporary_path = checkpoint_path.with_suffix(
                ".tmp"
            )

            torch.save(
                checkpoint_data,
                temporary_path
            )

            temporary_path.replace(
                checkpoint_path
            )

            del checkpoint_data

    # Итоговое время

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = previous_seconds

    if completed_step < steps:

        elapsed += (
            time.perf_counter() - start_time
        )

    result = {
        "d_model": width,
        "n_params": n_params,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "d_ff": d_ff,
        "context": context,
        "steps": steps,
        "batch": batch_size,
        "lr": lr,
        "seed": seed,
        "train_tokens": len(train_data),
        "val_tokens": len(val_data),
        "initial_train_loss": history[0]["train_loss"],
        "final_train_loss": history[-1]["train_loss"],
        "initial_val_loss": history[0]["val_loss"],
        "final_val_loss": history[-1]["val_loss"],
        "best_val_loss": min(
            row["val_loss"] for row in history
        ),
        "train_minutes": elapsed / 60,
        "history": history
    }

    print()
    print("РЕЗУЛЬТАТЫ МОДЕЛИ")
    print("Ширина:", width)
    print("Параметров:", n_params)

    print(
        "Конечный val loss:",
        result["final_val_loss"]
    )

    print(
        "Время обучения (мин.):",
        result["train_minutes"]
    )

    return result


# ============================================================
# ПОСТРОЕНИЕ ГРАФИКОВ
# ============================================================

def plot_scaling_results(results, output_dir):

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)

    results = sorted(
        results,
        key=lambda r: r["d_model"]
    )

    widths = [
        str(r["d_model"])
        for r in results
    ]

    colors = [
        "orange",
        "steelblue",
        "seagreen"
    ]

    # График 1. Три кривые Validation loss

    plt.figure(figsize=(12, 6))

    for result, color in zip(results, colors):

        plt.plot(
            [
                row["step"]
                for row in result["history"]
            ],
            [
                row["val_loss"]
                for row in result["history"]
            ],
            marker="o",
            markersize=4,
            color=color,
            label=f"Ширина {result['d_model']}"
        )

    plt.title(
        "Влияние ширины трансформера на Validation loss"
    )

    plt.xlabel("Шаг обучения")
    plt.ylabel("Validation loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        output_dir / "scaling_val_loss.png",
        dpi=150
    )

    plt.close()

    # График 2. Итоговые функции потерь

    final_train = [
        r["final_train_loss"]
        for r in results
    ]

    final_val = [
        r["final_val_loss"]
        for r in results
    ]

    positions = list(range(len(results)))

    plt.figure(figsize=(9, 5))

    train_bars = plt.bar(
        [x - 0.18 for x in positions],
        final_train,
        width=0.36,
        color="steelblue",
        label="Train loss"
    )

    val_bars = plt.bar(
        [x + 0.18 for x in positions],
        final_val,
        width=0.36,
        color="orange",
        label="Validation loss"
    )

    plt.bar_label(train_bars, fmt="%.4f", padding=3)
    plt.bar_label(val_bars, fmt="%.4f", padding=3)

    plt.xticks(positions, widths)
    plt.xlabel("Ширина модели")
    plt.ylabel("Loss")
    plt.title("Итоговые потери трансформеров")
    plt.ylim(0, max(final_val + final_train) * 1.15)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        output_dir / "scaling_final_loss.png",
        dpi=150
    )

    plt.close()

    # График 3. Количество параметров

    params = [
        r["n_params"]
        for r in results
    ]

    plt.figure(figsize=(9, 5))

    bars = plt.bar(
        widths,
        params,
        color=colors[:len(results)]
    )

    plt.bar_label(bars, padding=4, fmt="%.0f")

    plt.xlabel("Ширина модели")
    plt.ylabel("Количество параметров")
    plt.title("Количество параметров трансформера")
    plt.ylim(0, max(params) * 1.2)
    plt.ticklabel_format(style="plain", axis="y")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        output_dir / "scaling_parameters.png",
        dpi=150
    )

    plt.close()

    # График 4. Время обучения

    minutes = [
        r["train_minutes"]
        for r in results
    ]

    plt.figure(figsize=(9, 5))

    bars = plt.bar(
        widths,
        minutes,
        color=colors[:len(results)]
    )

    plt.bar_label(bars, fmt="%.4f", padding=4)

    plt.xlabel("Ширина модели")
    plt.ylabel("Время обучения, минут")
    plt.title("Время обучения трансформеров")
    plt.ylim(0, max(minutes) * 1.2)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        output_dir / "scaling_time.png",
        dpi=150
    )

    plt.close()


# ============================================================
# ВЫПОЛНЕНИЕ БОНУСА №3
# ============================================================

def run_scaling_bonus(
    train_data,
    val_data,
    vocab_size,
    output_dir,
    checkpoint_dir,
    device,
    widths=(64, 128, 256),
    steps=2000,
    batch_size=32,
    interval=100,
    seed=42,
    resume=False,
    make_plots=True
):

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    results = []

    for width in widths:

        result = train_scaling_model(
            width=width,
            vocab_size=vocab_size,
            train_data=train_data,
            val_data=val_data,
            checkpoint_dir=checkpoint_dir,
            device=device,
            steps=steps,
            batch_size=batch_size,
            interval=interval,
            seed=seed,
            resume=resume
        )

        results.append(result)

        # Сохранение индивидуальной кривой

        curve_path = (
            output_dir / f"scaling_{width}.csv"
        )

        with open(
            curve_path,
            "w",
            encoding="utf-8",
            newline=""
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "step",
                    "train_loss",
                    "val_loss"
                ]
            )

            writer.writeheader()
            writer.writerows(result["history"])

        # Сохранение промежуточных результатов

        write_json(
            output_dir / "scaling_results.json",
            results
        )

        if torch.device(device).type == "cuda":
            torch.cuda.empty_cache()

    # Сводная таблица

    columns = [
        "d_model",
        "n_params",
        "steps",
        "initial_train_loss",
        "initial_val_loss",
        "final_train_loss",
        "final_val_loss",
        "best_val_loss",
        "train_minutes",
        "generalization_gap"
    ]

    with open(
        output_dir / "scaling_summary.csv",
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=columns
        )

        writer.writeheader()

        for result in results:

            row = {
                key: result[key]
                for key in columns
                if key != "generalization_gap"
            }

            row["generalization_gap"] = (
                result["final_val_loss"]
                - result["final_train_loss"]
            )

            writer.writerow(row)

    if make_plots:

        plot_scaling_results(
            results,
            output_dir
        )

    print()
    print("ИТОГИ МАСШТАБИРОВАНИЯ")
    print("=" * 60)

    for result in results:

        print(
            f"Ширина: {result['d_model']} | "
            f"Параметров: {result['n_params']} | "
            f"Val loss: {result['final_val_loss']:.4f}"
        )

    best = min(
        results,
        key=lambda r: r["final_val_loss"]
    )

    print("Лучшая ширина:", best["d_model"])

    print(
        "Лучший Validation loss:",
        best["final_val_loss"]
    )

    assert len(results) == len(widths)

    assert all(
        r["history"][-1]["step"] == steps
        for r in results
    )

    assert all(
        math.isfinite(row["train_loss"])
        and math.isfinite(row["val_loss"])
        for r in results
        for row in r["history"]
    )

    return results




# ============================================================
# БОНУС №4. ДРУГОЙ КОРПУС — PYTHON
# ============================================================

import ast
import csv
import hashlib
import json
import math
import random
import re
import subprocess
import tempfile
import time

from pathlib import Path
from collections import Counter, defaultdict
from functools import lru_cache

import torch
import torch.nn.functional as F


DJANGO_REPO = "https://github.com/django/django.git"

DJANGO_COMMIT = (
    "9e7cc2b628fe8fd3895986af9b7fc9525034c1b0"
)


# ============================================================
# 1. ПОДГОТОВКА КОРПУСА PYTHON
# ============================================================

def prepare_python_corpus(
    source_dir,
    output_dir,
    target_size=3000000,
    seed=42
):

    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    if not source_dir.exists():

        source_dir.mkdir(parents=True)

        subprocess.run(
            ["git", "init", str(source_dir)],
            check=True
        )

        subprocess.run(
            [
                "git", "-C", str(source_dir),
                "fetch", "--depth", "1",
                DJANGO_REPO, DJANGO_COMMIT
            ],
            check=True
        )

        subprocess.run(
            [
                "git", "-C", str(source_dir),
                "checkout", "--detach", "FETCH_HEAD"
            ],
            check=True
        )

    commit = subprocess.check_output(
        [
            "git", "-C", str(source_dir),
            "rev-parse", "HEAD"
        ],
        text=True
    ).strip()

    assert commit == DJANGO_COMMIT

    files = sorted(
        (source_dir / "django").rglob("*.py")
    )

    random.Random(seed).shuffle(files)

    parts = []
    manifest = []
    total_bytes = 0

    for path in files:

        try:
            text = path.read_text(encoding="utf-8")

        except UnicodeDecodeError:
            continue

        if not text.strip():
            continue

        if not text.endswith("\n"):
            text += "\n"

        text += "\n"

        size = len(text.encode("utf-8"))

        if total_bytes + size > 5000000:
            continue

        manifest.append({
            "file": str(path.relative_to(source_dir)),
            "bytes": size
        })

        parts.append(text)
        total_bytes += size

        if total_bytes >= target_size:
            break

    corpus = "".join(parts)

    split = int(len(parts) * 0.9)

    train_text = "".join(parts[:split])
    val_text = "".join(parts[split:])

    assert train_text and val_text
    assert train_text + val_text == corpus
    assert 2000000 <= total_bytes <= 5000000

    stats = {
        "repository": DJANGO_REPO,
        "version": "5.2",
        "commit": commit,
        "seed": seed,
        "files": len(parts),
        "train_files": split,
        "val_files": len(parts) - split,
        "bytes": total_bytes,
        "characters": len(corpus),
        "lines": len(corpus.splitlines()),
        "unique_characters": len(set(corpus)),
        "spaces": corpus.count(" "),
        "newlines": corpus.count("\n"),
        "tabs": corpus.count("\t"),
        "train_characters": len(train_text),
        "val_characters": len(val_text),
        "sha256": hashlib.sha256(
            corpus.encode("utf-8")
        ).hexdigest()
    }

    write_json(
        output_dir / "python_corpus_stats.json",
        stats
    )

    write_json(
        output_dir / "source_files.json",
        manifest
    )

    print("КОРПУС PYTHON")
    print("Файлов:", len(parts))
    print("Размер корпуса:", total_bytes)
    print("Commit:", commit)

    return train_text, val_text, stats


# ============================================================
# 2. БАЙТОВЫЙ BPE
# ============================================================

class PythonBPETokenizer:

    pattern = (
        r"[^\W\d]\w*|\d+|[ \t]+|\r\n|\n|\r|[^\w\s]+|."
    )

    regex = re.compile(pattern, flags=re.DOTALL)

    def __init__(self, merges):

        self.merges = [
            tuple(pair) for pair in merges
        ]

        self.ranks = {
            pair: rank
            for rank, pair in enumerate(self.merges)
        }

        self.id_to_bytes = {
            i: bytes([i])
            for i in range(256)
        }

        for rank, (a, b) in enumerate(self.merges):

            self.id_to_bytes[256 + rank] = (
                self.id_to_bytes[a]
                + self.id_to_bytes[b]
            )

        self.vocab_size = 256 + len(self.merges)

    @staticmethod
    def merge_pair(sequence, pair, new_id):

        result = []
        i = 0

        while i < len(sequence):

            if (
                i + 1 < len(sequence)
                and (sequence[i], sequence[i + 1]) == pair
            ):

                result.append(new_id)
                i += 2

            else:

                result.append(sequence[i])
                i += 1

        return tuple(result)

    @classmethod
    def fit(cls, text, num_merges=1500):

        parts = cls.regex.findall(text)

        assert "".join(parts) == text

        frequencies = Counter(
            tuple(part.encode("utf-8"))
            for part in parts
        )

        del parts

        sequences = list(frequencies.keys())
        weights = list(frequencies.values())

        pair_counts = Counter()
        pair_locations = defaultdict(set)

        for index, sequence in enumerate(sequences):

            counts = Counter(
                zip(sequence, sequence[1:])
            )

            for pair, count in counts.items():

                pair_counts[pair] += (
                    count * weights[index]
                )

                pair_locations[pair].add(index)

        merges = []

        for step in range(num_merges):

            if not pair_counts:
                break

            best_pair = min(
                pair_counts,
                key=lambda pair: (
                    -pair_counts[pair],
                    pair
                )
            )

            new_id = 256 + step

            affected = list(
                pair_locations[best_pair]
            )

            for index in affected:

                old = sequences[index]
                weight = weights[index]

                for pair, count in Counter(
                    zip(old, old[1:])
                ).items():

                    pair_counts[pair] -= count * weight

                    pair_locations[pair].discard(index)

                    if pair_counts[pair] == 0:

                        del pair_counts[pair]
                        del pair_locations[pair]

                new = cls.merge_pair(
                    old, best_pair, new_id
                )

                sequences[index] = new

                for pair, count in Counter(
                    zip(new, new[1:])
                ).items():

                    pair_counts[pair] += count * weight

                    pair_locations[pair].add(index)

            merges.append(best_pair)

            if (step + 1) % 100 == 0:

                print(
                    "BPE:",
                    step + 1,
                    "/",
                    num_merges
                )

        return cls(merges)

    @lru_cache(maxsize=100000)
    def encode_part(self, part):

        sequence = tuple(
            part.encode("utf-8")
        )

        while len(sequence) > 1:

            pairs = list(
                zip(sequence, sequence[1:])
            )

            candidates = [
                pair for pair in pairs
                if pair in self.ranks
            ]

            if not candidates:
                break

            best = min(
                candidates,
                key=self.ranks.get
            )

            sequence = self.merge_pair(
                sequence,
                best,
                256 + self.ranks[best]
            )

        return sequence

    def encode(self, text):

        return [
            token_id
            for part in self.regex.findall(text)
            for token_id in self.encode_part(part)
        ]

    def decode(self, ids, errors="strict"):

        return b"".join(
            self.id_to_bytes[int(i)]
            for i in ids
        ).decode("utf-8", errors=errors)

    def save(self, path, train_seconds):

        write_json(
            Path(path),
            {
                "algorithm": "byte_level_bpe",
                "base_vocab_size": 256,
                "num_merges": len(self.merges),
                "vocab_size": self.vocab_size,
                "pretoken_pattern": self.pattern,
                "merges": [
                    list(pair)
                    for pair in self.merges
                ],
                "train_seconds": train_seconds
            }
        )


# ============================================================
# 3. ГЕНЕРАЦИЯ PYTHON
# ============================================================

@torch.no_grad()
def generate_python_code(
    model,
    tokenizer,
    prompt,
    max_new_tokens=100,
    temperature=0.8,
    context=128,
    seed=42
):

    device = next(model.parameters()).device

    ids = list(tokenizer.encode(prompt))

    assert ids
    assert temperature >= 0

    rng = torch.Generator(
        device="cpu"
    ).manual_seed(seed)

    was_training = model.training
    model.eval()

    try:

        for _ in range(max_new_tokens):

            x = torch.tensor(
                [ids[-context:]],
                dtype=torch.long,
                device=device
            )

            logits = model(x)

            if isinstance(logits, tuple):
                logits = logits[0]

            logits = logits[0, -1]

            if temperature == 0:

                next_id = logits.argmax().item()

            else:

                probabilities = F.softmax(
                    logits.float() / temperature,
                    dim=-1
                ).cpu()

                next_id = torch.multinomial(
                    probabilities,
                    1,
                    generator=rng
                ).item()

            ids.append(next_id)

    finally:

        model.train(was_training)

    return tokenizer.decode(
        ids,
        errors="replace"
    )


# ============================================================
# 4. ОБУЧЕНИЕ ТРАНСФОРМЕРА
# ============================================================

def train_python_transformer(
    tokenizer,
    train_data,
    val_data,
    output_dir,
    checkpoint_dir,
    device,
    steps=2000,
    batch_size=32,
    interval=100,
    sample_tokens=100,
    seed=42,
    resume=False
):

    output_dir = Path(output_dir)
    checkpoint_dir = Path(checkpoint_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(device)

    train_data = train_data.cpu()
    val_data = val_data.cpu()

    context = 128

    torch.manual_seed(seed)

    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model = TinyTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        n_layers=4,
        n_heads=4,
        d_ff=512,
        context=context
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001
    )

    train_rng = torch.Generator().manual_seed(seed)

    val_rng = torch.Generator().manual_seed(seed + 1)

    initial_rng = torch.Generator().manual_seed(seed)

    val_x, val_y = make_training_batch(
        val_data, 20, context, val_rng, device
    )

    x0, y0 = make_training_batch(
        train_data,
        batch_size,
        context,
        initial_rng,
        device
    )

    prompts = [
        "def ",
        "class ",
        "def calculate_sum(a, b):\n    "
    ]

    sample_steps = sorted({
        step
        for step in [0, 100, 500, steps]
        if step <= steps
    })

    history = []
    samples = []

    start_step = 0
    previous_seconds = 0.0

    checkpoint_path = checkpoint_dir / "latest.pt"

    settings = {
        "vocab_size": tokenizer.vocab_size,
        "steps": steps,
        "batch_size": batch_size,
        "interval": interval,
        "sample_tokens": sample_tokens,
        "seed": seed,
        "train_tokens": len(train_data),
        "val_tokens": len(val_data),
        "data_sha256": hashlib.sha256(
            train_data.numpy().tobytes()
            + val_data.numpy().tobytes()
        ).hexdigest()
    }

    if resume and checkpoint_path.is_file():

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True
        )

        assert checkpoint["settings"] == settings

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        train_rng.set_state(
            checkpoint["train_rng_state"]
        )

        history = checkpoint["history"]
        samples = checkpoint["samples"]

        start_step = checkpoint["step"]

        previous_seconds = checkpoint["seconds"]

        del checkpoint

        print("Восстановлен шаг:", start_step)

    start_time = time.perf_counter()

    def collect_samples(step):

        for prompt in prompts:

            text = generate_python_code(
                model,
                tokenizer,
                prompt,
                max_new_tokens=sample_tokens,
                temperature=0.8,
                seed=seed
            )

            samples.append({
                "step": step,
                "prompt": prompt,
                "text": text
            })

    def save_checkpoint(step):

        if device.type == "cuda":
            torch.cuda.synchronize()

        state = {
            "settings": settings,
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_rng_state": train_rng.get_state(),
            "history": history,
            "samples": samples,
            "seconds": (
                previous_seconds
                + time.perf_counter()
                - start_time
            )
        }

        temporary = checkpoint_dir / "latest.tmp"

        torch.save(state, temporary)

        temporary.replace(checkpoint_path)

    if not history:

        history.append({
            "step": 0,
            "train_loss": validation_loss(model, x0, y0),
            "val_loss": validation_loss(model, val_x, val_y)
        })

        collect_samples(0)

        save_checkpoint(0)

    model.train()

    accumulated = []

    for step in range(start_step + 1, steps + 1):

        x, y = make_training_batch(
            train_data,
            batch_size,
            context,
            train_rng,
            device
        )

        optimizer.zero_grad(set_to_none=True)

        loss = transformer_loss(model, x, y)

        if not torch.isfinite(loss).item():

            raise RuntimeError(
                f"Некорректный loss на шаге {step}"
            )

        loss.backward()

        optimizer.step()

        accumulated.append(loss.item())

        if step in sample_steps:

            collect_samples(step)

        if step % interval == 0 or step == steps:

            train_loss = (
                sum(accumulated) / len(accumulated)
            )

            val_loss = validation_loss(
                model,
                val_x,
                val_y
            )

            history.append({
                "step": step,
                "train_loss": train_loss,
                "val_loss": val_loss
            })

            print(
                f"Python | Шаг {step}/{steps} | "
                f"Train loss: {train_loss:.4f} | "
                f"Val loss: {val_loss:.4f}",
                flush=True
            )

            accumulated = []

            save_checkpoint(step)

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = previous_seconds

    if start_step < steps:

        elapsed += (
            time.perf_counter() - start_time
        )

    config = {
        "corpus": "Django 5.2",
        "vocab_size": tokenizer.vocab_size,
        "d_model": 128,
        "n_layers": 4,
        "n_heads": 4,
        "d_ff": 512,
        "context": context,
        "n_params": sum(
            p.numel() for p in model.parameters()
        ),
        "steps": steps,
        "batch": batch_size,
        "lr": 0.001,
        "seed": seed,
        "train_tokens": len(train_data),
        "val_tokens": len(val_data),
        "bpe_merges": len(tokenizer.merges),
        "train_minutes": elapsed / 60,
        "device": str(device)
    }

    with open(
        output_dir / "train_curve.csv",
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step", "train_loss", "val_loss"
            ]
        )

        writer.writeheader()
        writer.writerows(history)

    write_json(
        output_dir / "samples.json",
        {
            "temperature": 0.8,
            "max_new_tokens": sample_tokens,
            "prompts": prompts,
            "samples": samples
        }
    )

    write_json(
        output_dir / "tiny_config.json",
        config
    )

    assert history[-1]["step"] == steps

    return model, history, samples, config


# ============================================================
# 5. ЭКСПЕРИМЕНТ С РЕЖИМАМИ ГЕНЕРАЦИИ
# ============================================================

def evaluate_python_generation(
    model,
    tokenizer,
    output_dir,
    steps=2000,
    max_new_tokens=100,
    seed=42
):

    prompts = [
        "def ",
        "class ",
        "def calculate_sum(a, b):\n    ",
        "import "
    ]

    modes = [
        ("greedy", 0.0),
        ("temperature_0.8", 0.8)
    ]

    runs = []

    for prompt in prompts:

        for name, temperature in modes:

            text = generate_python_code(
                model,
                tokenizer,
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                seed=seed
            )

            try:

                ast.parse(text)

                syntax_correct = True
                syntax_error = None

            except (SyntaxError, ValueError) as error:

                syntax_correct = False
                syntax_error = str(error)

            continuation = text[len(prompt):]

            runs.append({
                "prompt": prompt,
                "mode": name,
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "text": text,
                "syntax_correct": syntax_correct,
                "syntax_error": syntax_error,
                "newlines": continuation.count("\n"),
                "indented_lines": sum(
                    line.startswith(" ")
                    and bool(line.strip())
                    for line in continuation.splitlines()
                ),
                "replacement_characters":
                    continuation.count("\ufffd")
            })

    summary = []

    for name, _ in modes:

        records = [
            row for row in runs
            if row["mode"] == name
        ]

        valid = sum(
            row["syntax_correct"]
            for row in records
        )

        summary.append({
            "mode": name,
            "samples": len(records),
            "valid_syntax": valid,
            "invalid_syntax": len(records) - valid,
            "newlines": sum(
                row["newlines"] for row in records
            ),
            "indented_lines": sum(
                row["indented_lines"] for row in records
            )
        })

    result = {
        "corpus": "Django 5.2",
        "model": "TinyTransformer",
        "training_steps": steps,
        "seed": seed,
        "max_new_tokens": max_new_tokens,
        "runs": runs,
        "summary": summary
    }

    write_json(
        Path(output_dir) / "python_generation.json",
        result
    )

    return result


# ============================================================
# 6. ГРАФИКИ
# ============================================================

def plot_python_bonus(
    history,
    generation,
    output_dir,
    literature_curve_path=None
):

    import matplotlib
    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)

    steps = [
        row["step"] for row in history
    ]

    train_losses = [
        row["train_loss"] for row in history
    ]

    val_losses = [
        row["val_loss"] for row in history
    ]

    plt.figure(figsize=(12, 6))

    plt.plot(
        steps, train_losses,
        marker="o", label="Train loss"
    )

    plt.plot(
        steps, val_losses,
        marker="s", label="Validation loss"
    )

    plt.xlabel("Шаг обучения")
    plt.ylabel("Loss")
    plt.title("Обучение трансформера на Python")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        output_dir / "python_train_curve.png",
        dpi=150
    )

    plt.close()

    # Сравнение с художественным текстом

    if literature_curve_path is not None:

        with open(
            literature_curve_path,
            encoding="utf-8"
        ) as f:

            literature = list(csv.DictReader(f))

        plt.figure(figsize=(12, 6))

        plt.plot(
            [int(r["step"]) for r in literature],
            [float(r["val_loss"]) for r in literature],
            marker="o",
            label="Война и мир"
        )

        plt.plot(
            steps,
            val_losses,
            marker="s",
            label="Python (Django)"
        )

        plt.xlabel("Шаг обучения")
        plt.ylabel("Validation loss на токен")
        plt.title("Динамика обучения на двух корпусах")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        plt.savefig(
            output_dir / "python_literature_comparison.png",
            dpi=150
        )

        plt.close()

    # Сравнение режимов генерации

    rows = generation["summary"]

    positions = list(range(len(rows)))

    plt.figure(figsize=(9, 5))

    plt.bar(
        [x - 0.18 for x in positions],
        [r["newlines"] for r in rows],
        width=0.36,
        label="Переносы строк"
    )

    plt.bar(
        [x + 0.18 for x in positions],
        [r["indented_lines"] for r in rows],
        width=0.36,
        label="Строки с отступами"
    )

    plt.xticks(
        positions,
        [r["mode"] for r in rows]
    )

    plt.title("Форматирование сгенерированного кода")
    plt.ylabel("Количество")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        output_dir / "python_generation_comparison.png",
        dpi=150
    )

    plt.close()


# ============================================================
# 7. ВЫПОЛНЕНИЕ БОНУСА №4
# ============================================================

def run_python_bonus(
    output_dir,
    checkpoint_dir,
    device,
    source_dir=None,
    literature_curve_path=None,
    steps=2000,
    resume=False
):

    output_dir = Path(output_dir)
    checkpoint_dir = Path(checkpoint_dir)

    corpus_dir = output_dir / "python_corpus"
    results_dir = corpus_dir / "results"

    corpus_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Подготовка воспроизводимого корпуса

    if source_dir is None:

        with tempfile.TemporaryDirectory() as temp:

            train_text, val_text, corpus_stats = (
                prepare_python_corpus(
                    Path(temp) / "django",
                    corpus_dir
                )
            )

    else:

        train_text, val_text, corpus_stats = (
            prepare_python_corpus(
                source_dir,
                corpus_dir
            )
        )

    # Обучение BPE

    start = time.perf_counter()

    tokenizer = PythonBPETokenizer.fit(
        train_text,
        num_merges=1500
    )

    bpe_seconds = time.perf_counter() - start

    assert len(tokenizer.merges) == 1500
    assert tokenizer.vocab_size == 1756

    tokenizer.save(
        corpus_dir / "python_bpe.json",
        bpe_seconds
    )

    # Токенизация корпуса

    start = time.perf_counter()

    train_data = torch.tensor(
        tokenizer.encode(train_text),
        dtype=torch.long
    )

    val_data = torch.tensor(
        tokenizer.encode(val_text),
        dtype=torch.long
    )

    tokenization_seconds = (
        time.perf_counter() - start
    )

    assert tokenizer.decode(
        train_data.tolist()
    ) == train_text

    assert tokenizer.decode(
        val_data.tolist()
    ) == val_text

    token_stats = {
        "vocab_size": tokenizer.vocab_size,
        "bpe_merges": len(tokenizer.merges),
        "train_chars": len(train_text),
        "val_chars": len(val_text),
        "total_chars": len(train_text) + len(val_text),
        "train_tokens": len(train_data),
        "val_tokens": len(val_data),
        "total_tokens": len(train_data) + len(val_data),
        "chars_per_token": (
            (len(train_text) + len(val_text))
            / (len(train_data) + len(val_data))
        ),
        "tokenization_seconds": tokenization_seconds
    }

    write_json(
        corpus_dir / "tokenization_stats.json",
        token_stats
    )

    # Обучение модели

    model, history, samples, config = (
        train_python_transformer(
            tokenizer,
            train_data,
            val_data,
            results_dir,
            checkpoint_dir,
            device,
            steps=steps,
            resume=resume
        )
    )

    # Проверка генерации

    generation = evaluate_python_generation(
        model,
        tokenizer,
        results_dir,
        steps=steps
    )

    # Построение графиков

    plot_python_bonus(
        history,
        generation,
        output_dir,
        literature_curve_path
    )

    # Итоговые результаты

    summary = {
        "corpus": corpus_stats,
        "tokenization": token_stats,
        "config": config,
        "initial_train_loss":
            history[0]["train_loss"],
        "final_train_loss":
            history[-1]["train_loss"],
        "initial_val_loss":
            history[0]["val_loss"],
        "final_val_loss":
            history[-1]["val_loss"],
        "generation": generation["summary"]
    }

    write_json(
        output_dir / "python_bonus_summary.json",
        summary
    )

    print("БОНУС №4 ВЫПОЛНЕН")

    print(
        "Конечный Train loss:",
        history[-1]["train_loss"]
    )

    print(
        "Конечный Val loss:",
        history[-1]["val_loss"]
    )

    print("Образцов:", len(samples))

    return summary



# ============================================================
# ТОЧКА ВХОДА В ПРОГРАММУ
# ЛАБОРАТОРНАЯ РАБОТА №2
# ============================================================

import argparse
import os
import urllib.request

from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent

ARTIFACTS = HERE / "artifacts"

BPE = HERE.parent / "pt01" / "artifacts" / "bpe.json"

CORPUS = HERE / "corpus.txt"

CORPUS_URL = (
    "https://raw.githubusercontent.com/"
    "nevmenandr/word2vec-russian-novels/"
    "master/books_before/WarAndPeace.txt"
)


# ============================================================
# ЗАГРУЗКА КОРПУСА
# ============================================================

def download_corpus(path):

    path = Path(path)

    if path.is_file() and path.stat().st_size > 0:

        print("Корпус уже доступен:", path)

        return path

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Загрузка корпуса «Война и мир»")

    temporary = path.with_suffix(".download")

    urllib.request.urlretrieve(
        CORPUS_URL,
        temporary
    )

    assert temporary.stat().st_size > 0

    temporary.replace(path)

    print(
        "Корпус сохранён:",
        path
    )

    print(
        "Размер корпуса:",
        path.stat().st_size,
        "байт"
    )

    return path


# ============================================================
# АРГУМЕНТЫ КОМАНДНОЙ СТРОКИ
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Лабораторная работа №2. "
            "Свой трансформер: обучение и генерация"
        )
    )

    parser.add_argument(
        "--fast",
        action="store_true",
        help="Быстрый режим: 200 шагов обучения"
    )

    parser.add_argument(
        "--bpe",
        type=Path,
        default=BPE,
        help="Путь к BPE из первой лабораторной"
    )

    parser.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help="Путь к локальному корпусу"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ARTIFACTS,
        help="Папка для сохранения артефактов"
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Папка контрольных точек"
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Устройство для обучения"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed генератора случайных чисел"
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="Количество потоков CPU"
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Продолжить обучение из контрольной точки"
    )

    parser.add_argument(
        "--bonuses",
        nargs="+",
        choices=["qwen", "scaling", "python", "all"],
        default=[],
        help="Дополнительные бонусные эксперименты"
    )

    parser.add_argument(
        "--qwen-dir",
        type=Path,
        default=None,
        help="Путь к локальной модели Qwen"
    )

    parser.add_argument(
        "--django-dir",
        type=Path,
        default=None,
        help="Путь к локальному репозиторию Django"
    )

    args = parser.parse_args()

    if args.fast and args.bonuses:

        parser.error(
            "Бонусы запускаются отдельно от режима --fast"
        )

    if args.threads < 1:

        parser.error(
            "--threads должен быть положительным числом"
        )

    return args


# ============================================================
# ОСНОВНАЯ ПРОГРАММА
# ============================================================

def main():

    args = parse_arguments()

    mode = "fast" if args.fast else "full"

    output_dir = args.output_dir.resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    if args.checkpoint_dir is None:

        checkpoint_dir = (
            output_dir.parent
            / (output_dir.name + "_checkpoints")
            / mode
        )

    else:

        checkpoint_dir = (
            args.checkpoint_dir.resolve()
            / mode
        )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Выбор устройства

    if args.device == "auto":

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    else:

        device = torch.device(args.device)

    if (
        device.type == "cuda"
        and not torch.cuda.is_available()
    ):

        raise RuntimeError(
            "CUDA недоступна"
        )

    torch.set_num_threads(args.threads)

    torch.manual_seed(args.seed)

    if device.type == "cuda":

        torch.cuda.manual_seed_all(args.seed)

    print("=" * 60)

    print(
        "ЛАБОРАТОРНАЯ РАБОТА №2"
    )

    print(
        "СВОЙ ТРАНСФОРМЕР: ОБУЧЕНИЕ И ГЕНЕРАЦИЯ"
    )

    print("=" * 60)

    print("Режим:", mode)
    print("Устройство:", device)
    print("Seed:", args.seed)
    print("Потоки CPU:", args.threads)
    print("Артефакты:", output_dir)
    print("Контрольные точки:", checkpoint_dir)

    if device.type == "cuda":

        print(
            "Видеокарта:",
            torch.cuda.get_device_name(device)
        )

    # --------------------------------------------------------
    # ЗАДАНИЕ №1. ДАННЫЕ
    # --------------------------------------------------------

    bpe_path = args.bpe.resolve()

    if not bpe_path.is_file():

        raise FileNotFoundError(
            f"Не найден BPE: {bpe_path}"
        )

    if args.corpus is None:

        corpus_path = download_corpus(CORPUS)

    else:

        corpus_path = args.corpus.resolve()

        if not corpus_path.is_file():

            raise FileNotFoundError(
                f"Не найден корпус: {corpus_path}"
            )

    (
        tokenizer,
        train_words,
        val_words,
        train_data,
        val_data

    ) = load_data(
        corpus_path,
        bpe_path
    )

    print("\nЗАДАНИЕ №1. ДАННЫЕ")
    print("=" * 60)

    print("Размер словаря:", tokenizer.vocab_size)
    print("Слияний BPE:", len(tokenizer.merges))
    print("Обучающих слов:", len(train_words))
    print("Проверочных слов:", len(val_words))
    print("Обучающих токенов:", len(train_data))
    print("Проверочных токенов:", len(val_data))

    assert len(tokenizer.merges) >= 200

    # --------------------------------------------------------
    # ЗАДАНИЕ №2. МОДЕЛЬ
    # --------------------------------------------------------

    torch.manual_seed(args.seed)

    model = TinyTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        n_layers=4,
        n_heads=4,
        d_ff=512,
        context=128
    ).to(device)

    n_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print("\nЗАДАНИЕ №2. МОДЕЛЬ")
    print("=" * 60)

    print("Количество параметров:", n_params)

    # --------------------------------------------------------
    # ЗАДАНИЯ №3 И №4. ОБУЧЕНИЕ И ОБРАЗЦЫ
    # --------------------------------------------------------

    print("\nЗАДАНИЯ №3 И №4")
    print("=" * 60)

    history, samples, config = train_transformer(
        model=model,
        tokenizer=tokenizer,
        train_data=train_data,
        val_data=val_data,
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        vocab_size=tokenizer.vocab_size,
        bpe_merges=len(tokenizer.merges),
        fast=args.fast,
        resume=args.resume,
        seed=args.seed
    )

    print(
        "Обучение завершено на шаге:",
        history[-1]["step"]
    )

    print(
        "Начальный Val loss:",
        history[0]["val_loss"]
    )

    print(
        "Конечный Val loss:",
        history[-1]["val_loss"]
    )

    # --------------------------------------------------------
    # ЗАДАНИЕ №5. ГЕНЕРАЦИЯ РУКАМИ
    # --------------------------------------------------------

    print("\nЗАДАНИЕ №5. ГЕНЕРАЦИЯ")
    print("=" * 60)

    generation = run_generation_experiment(
        model,
        tokenizer,
        output_dir,
        seed=args.seed,
        fast=args.fast
    )

    # --------------------------------------------------------
    # ЗАДАНИЕ №6. СРАВНЕНИЕ МОДЕЛЕЙ
    # --------------------------------------------------------

    print("\nЗАДАНИЕ №6. СРАВНЕНИЕ МОДЕЛЕЙ")
    print("=" * 60)

    evaluation = run_evaluation(
        model=model,
        tokenizer=tokenizer,
        train_words=train_words,
        val_words=val_words,
        output_dir=output_dir,
        vocab_size=tokenizer.vocab_size,
        n_words=300 if args.fast else 2000
    )

    # --------------------------------------------------------
    # БОНУСНЫЕ ЗАДАНИЯ
    # --------------------------------------------------------

    selected = set(args.bonuses)

    if "all" in selected:

        selected = {
            "qwen", "scaling", "python"
        }

    bonus_dir = output_dir / "bonus"

    if selected:

        bonus_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    if "qwen" in selected:

        print("\nБОНУСЫ №1 И №2. QWEN")
        print("=" * 60)

        run_qwen_bonuses(
            val_words=val_words,
            output_dir=output_dir,
            model_dir=args.qwen_dir,
            device=device
        )

    if "scaling" in selected:

        print("\nБОНУС №3. МАСШТАБИРОВАНИЕ")
        print("=" * 60)

        run_scaling_bonus(
            train_data=train_data,
            val_data=val_data,
            vocab_size=tokenizer.vocab_size,
            output_dir=bonus_dir,
            checkpoint_dir=checkpoint_dir / "scaling",
            device=device,
            widths=(64, 128, 256),
            steps=2000,
            batch_size=32,
            seed=args.seed,
            resume=args.resume
        )

    if "python" in selected:

        print("\nБОНУС №4. ДРУГОЙ КОРПУС")
        print("=" * 60)

        run_python_bonus(
            output_dir=bonus_dir,
            checkpoint_dir=checkpoint_dir / "python",
            device=device,
            source_dir=args.django_dir,
            literature_curve_path=(
                output_dir / "train_curve.csv"
            ),
            steps=2000,
            resume=args.resume
        )

    # --------------------------------------------------------
    # ИТОГОВАЯ ПРОВЕРКА
    # --------------------------------------------------------

    required = [
        "tiny_config.json",
        "train_curve.csv",
        "samples.json",
        "generation.json",
        "eval.json"
    ]

    print("\nИТОГОВАЯ ПРОВЕРКА")
    print("=" * 60)

    for name in required:

        exists = (
            output_dir / name
        ).is_file()

        print(name + ":", exists)

        assert exists

    assert history[-1]["step"] == (
        200 if args.fast else 2000
    )

    assert len(samples) == 4

    assert len(generation["runs"]) == 4

    print("\nЛАБОРАТОРНАЯ РАБОТА №2 ВЫПОЛНЕНА")

    print("Результаты:", output_dir)


if __name__ == "__main__":

    main()
