"""Ядро NameForge: генерация → фильтр → скоринг → ранжирование.

Класс Pipeline не знает ни о каком UI. Он возвращает полные данные
(ScoredName с метаданными), а UI решает, что показывать пользователю.
Writer-режим покажет только name, defense-режим — всё остальное.
"""
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch

from ml.classifier import load_classifier, predict_proba
from ml.lstm_gen import STYLES, load_lstm
from ml.lstm_gen import sample as lstm_sample
from ml.novelty import NoveltyScorer
from ml.plausibility import BigramPlausibility
from ml.report import count_syllables, cv_pattern
from ml.transformer_gen import load_transformer
from ml.transformer_gen import sample as transformer_sample


@dataclass
class ScoredName:
    """Одно сгенерированное имя со всеми метаданными."""
    name: str
    scores: dict[str, float]          # style, novelty, plausibility, total
    breakdown: dict = field(default_factory=dict)  # syllables, pattern, style_probs


def _load_scorers(models_dir: Path):
    """Восстанавливает classifier + plausibility + novelty из чекпоинтов."""
    classifier, stoi_cls = load_classifier(models_dir / "classifier.pt")
    ckpt = torch.load(models_dir / "scorers.pt",
                      map_location="cpu", weights_only=False)

    p_data = ckpt["plausibility"]
    plausibility = BigramPlausibility(smoothing=p_data["smoothing"])
    plausibility.counts = p_data["counts"]
    plausibility.context_counts = p_data["context_counts"]
    plausibility.vocab = set(p_data["vocab"])
    plausibility.mean_lp = p_data["mean_lp"]
    plausibility.std_lp = p_data["std_lp"]

    n_data = ckpt["novelty"]
    novelty = NoveltyScorer(ngram_sizes=n_data["ngram_sizes"])
    novelty.train_ngrams = n_data["train_ngrams"]

    return classifier, stoi_cls, plausibility, novelty


class Pipeline:
    """Единая точка входа: генерация + скоринг + ранжирование."""

    def __init__(self, model_type: str = "lstm",
                 models_dir: Optional[Path] = None):
        if model_type not in ("lstm", "transformer"):
            raise ValueError(
                f"model_type должен быть 'lstm' или 'transformer', "
                f"получено {model_type!r}"
            )
        self.model_type = model_type
        self.models_dir = models_dir or (
            Path(__file__).resolve().parent.parent / "models"
        )
        if not self.models_dir.exists():
            raise FileNotFoundError(
                f"Папка моделей не найдена: {self.models_dir}. "
                f"Запусти python -m ml.train_lstm и python -m ml.train_scorers"
            )

        # Генератор
        if model_type == "lstm":
            gen_path = self.models_dir / "lstm.pt"
            if not gen_path.exists():
                raise FileNotFoundError(
                    f"Модель не найдена: {gen_path}. "
                    f"Запусти python -m ml.train_lstm"
                )
            self.gen_model, self.itos, self.stoi_gen = load_lstm(gen_path)
        else:
            gen_path = self.models_dir / "transformer.pt"
            if not gen_path.exists():
                raise FileNotFoundError(
                    f"Модель не найдена: {gen_path}. "
                    f"Запусти python -m ml.train_transformer"
                )
            self.gen_model, self.itos, self.stoi_gen = load_transformer(gen_path)

        # Скореры
        for name in ("classifier.pt", "scorers.pt"):
            if not (self.models_dir / name).exists():
                raise FileNotFoundError(
                    f"Скор-модель не найдена: {self.models_dir / name}. "
                    f"Запусти python -m ml.train_scorers"
                )
        self.classifier, self.stoi_cls, self.plausibility, self.novelty = (
            _load_scorers(self.models_dir)
        )

    # ---------- внутренние методы ----------

    def _sample_one(self, style: str, temperature: float, top_k: int,
                    top_p: float, seed: int) -> str:
        if self.model_type == "lstm":
            return lstm_sample(self.gen_model, self.itos, self.stoi_gen,
                               style, temperature, top_k, top_p, seed)
        return transformer_sample(self.gen_model, self.itos, self.stoi_gen,
                                  style, temperature, top_k, top_p, seed)

    def _score_one(self, name: str, target_style: str) -> ScoredName:
        style_probs = predict_proba(self.classifier, self.stoi_cls, name)
        style_score = style_probs[target_style]
        novelty_score = self.novelty.score(name)
        plaus_score = self.plausibility.score(name)
        total = 0.5 * style_score + 0.25 * novelty_score + 0.25 * plaus_score
        return ScoredName(
            name=name,
            scores={
                "style": style_score,
                "novelty": novelty_score,
                "plausibility": plaus_score,
                "total": total,
            },
            breakdown={
                "syllables": count_syllables(name),
                "pattern": cv_pattern(name),
                "style_probs": style_probs,
            },
        )

    # ---------- публичный API ----------

    def generate(self, style: str, syllables: int, count: int,
                 seed: Optional[int] = None, temperature: float = 0.8,
                 top_k: int = 40, top_p: float = 0.9,
                 n_candidates: int = 100, max_iterations: int = 3
                 ) -> list[ScoredName]:
        """Генерирует `count` имён с заданными параметрами.

        Параметры:
            style: один из STYLES
            syllables: желаемое число слогов (1..10)
            count: сколько имён вернуть (1..50)
            seed: для воспроизводимости (None = случайный)
            temperature, top_k, top_p: параметры сэмплирования
            n_candidates: сколько генерировать за одну итерацию
            max_iterations: сколько раз повторить, если не набрали count

        Возвращает список ScoredName, отсортированный по total (убыв.).
        Может вернуть меньше `count` имён, если после всех итераций
        не удалось набрать нужное количество — это не ошибка.
        """
        # ---- валидация ----
        if style not in STYLES:
            raise ValueError(
                f"Неизвестный стиль: {style!r}. Допустимые: {STYLES}"
            )
        if not (1 <= syllables <= 10):
            raise ValueError(
                f"syllables должен быть 1..10, получено {syllables}"
            )
        if not (1 <= count <= 50):
            raise ValueError(f"count должен быть 1..50, получено {count}")
        if not (0.0 < temperature <= 2.0):
            raise ValueError(
                f"temperature должен быть 0 < t <= 2, получено {temperature}"
            )
        if top_k < 0:
            raise ValueError(f"top_k должен быть >= 0, получено {top_k}")
        if not (0.0 < top_p <= 1.0):
            raise ValueError(
                f"top_p должен быть 0 < p <= 1, получено {top_p}"
            )
        if n_candidates < 1:
            raise ValueError(
                f"n_candidates должен быть >= 1, получено {n_candidates}"
            )

        # ---- базовый seed для воспроизводимости ----
        if seed is not None:
            base_seed = seed
        else:
            base_seed = random.Random().randint(0, 2**31 - 1)

        # ---- итеративная генерация ----
        all_scored: list[ScoredName] = []
        seen_names: set[str] = set()
        iteration = 0

        while len(all_scored) < count and iteration < max_iterations:
            for i in range(n_candidates):
                cand_seed = base_seed + iteration * n_candidates + i
                name = self._sample_one(
                    style, temperature, top_k, top_p, cand_seed
                )
                if not name:
                    continue
                name = name[0].upper() + name[1:].lower()
                if name in seen_names:
                    continue
                if count_syllables(name) != syllables:
                    continue
                seen_names.add(name)
                all_scored.append(self._score_one(name, style))
            iteration += 1

        # ---- ранжирование ----
        all_scored.sort(key=lambda s: -s.scores["total"])
        return all_scored[:count]
