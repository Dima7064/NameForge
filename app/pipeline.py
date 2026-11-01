
"""Ядро NameForge: генерация → фильтр → скоринг → ранжирование + объяснения."""
import random
import re
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

VOWELS = set("аеёиоуыэюяАЕЁИОУЫЭЮЯ")


@dataclass
class ScoredName:
    name: str
    scores: dict[str, float]
    breakdown: dict = field(default_factory=dict)


def _load_scorers(models_dir: Path):
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


def _split_syllables(name: str) -> list[str]:
    """Грубое разбиение на слоги: каждый слог = согласные + одна гласная."""
    if not name:
        return []
    parts: list[str] = []
    current = ""
    for ch in name:
        current += ch
        if ch in VOWELS:
            parts.append(current)
            current = ""
    if current:
        if parts:
            parts[-1] += current
        else:
            parts.append(current)
    return parts


class Pipeline:
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
                f"Папка моделей не найдена: {self.models_dir}"
            )

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

        for name in ("classifier.pt", "scorers.pt"):
            if not (self.models_dir / name).exists():
                raise FileNotFoundError(
                    f"Скор-модель не найдена: {self.models_dir / name}"
                )
        self.classifier, self.stoi_cls, self.plausibility, self.novelty = (
            _load_scorers(self.models_dir)
        )

        # train-имена для проверки «не из train»
        train_path = self.models_dir.parent / "data" / "train.json"
        self.train_names: set[str] = set()
        if train_path.exists():
            import json
            with train_path.open("r", encoding="utf-8") as f:
                self.train_names = {d["name"].lower() for d in json.load(f)}

    # ---------- внутренние ----------

    def _sample_one(self, style: str, temperature: float, top_k: int,
                    top_p: float, seed: int) -> str:
        if self.model_type == "lstm":
            return lstm_sample(self.gen_model, self.itos, self.stoi_gen,
                               style, temperature, top_k, top_p, seed)
        return transformer_sample(self.gen_model, self.itos, self.stoi_gen,
                                  style, temperature, top_k, top_p, seed)

    def _score_one(self, name: str, target_style: str,
                   w_style: float, w_novelty: float, w_plaus: float) -> ScoredName:
        style_probs = predict_proba(self.classifier, self.stoi_cls, name)
        style_score = style_probs[target_style]
        novelty_score = self.novelty.score(name)
        plaus_score = self.plausibility.score(name)
        total = (w_style * style_score
                 + w_novelty * novelty_score
                 + w_plaus * plaus_score)
        return ScoredName(
            name=name,
            scores={
                "style": style_score, "novelty": novelty_score,
                "plausibility": plaus_score, "total": total,
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
                 n_candidates: int = 100, max_iterations: int = 3,
                 w_style: float = 0.5, w_novelty: float = 0.25,
                 w_plaus: float = 0.25) -> list[ScoredName]:
        """Генерирует count имён с заданными параметрами и весами."""
        if style not in STYLES:
            raise ValueError(f"Неизвестный стиль: {style!r}")
        if not (1 <= syllables <= 10):
            raise ValueError(f"syllables должен быть 1..10")
        if not (1 <= count <= 50):
            raise ValueError(f"count должен быть 1..50")
        if not (0.0 < temperature <= 2.0):
            raise ValueError(f"temperature должна быть 0 < t <= 2")
        if top_k < 0:
            raise ValueError(f"top_k должен быть >= 0")
        if not (0.0 < top_p <= 1.0):
            raise ValueError(f"top_p должен быть 0 < p <= 1")

        base_seed = seed if seed is not None else random.Random().randint(0, 2**31 - 1)

        all_scored: list[ScoredName] = []
        seen: set[str] = set()
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
                if name in seen:
                    continue
                if count_syllables(name) != syllables:
                    continue
                seen.add(name)
                all_scored.append(
                    self._score_one(name, style, w_style, w_novelty, w_plaus)
                )
            iteration += 1

        all_scored.sort(key=lambda s: -s.scores["total"])
        return all_scored[:count]

    def explain(self, name: str, target_style: str,
                w_style: float = 0.5, w_novelty: float = 0.25,
                w_plaus: float = 0.25) -> dict:
        """Полный разбор имени: слоги, паттерн, дифтонги, окончание,
        комментарии по стилю, скоры. Возвращает dict."""
        style_probs = predict_proba(self.classifier, self.stoi_cls, name)
        style_score = style_probs[target_style]
        nov = self.novelty.score(name)
        plaus = self.plausibility.score(name)
        total = w_style * style_score + w_novelty * nov + w_plaus * plaus

        syl_parts = _split_syllables(name)
        syl_count = count_syllables(name)
        pattern = cv_pattern(name)

        # дифтонги (подряд идущие гласные)
        vowels_lc = set("аеёиоуыэюя")
        dips: list[str] = []
        for i in range(len(name) - 1):
            a, b = name[i].lower(), name[i + 1].lower()
            if a in vowels_lc and b in vowels_lc:
                dips.append(name[i:i + 2])

        ending = name[-3:] if len(name) >= 3 else name

        # что прошло фильтр
        filters_passed = [f"слогов = {syl_count}"]
        if name.lower() not in self.train_names:
            filters_passed.append("не из train")
        else:
            filters_passed.append("⚠️ совпадает с train")

        # эвристические комментарии по стилю
        comments: list[str] = []
        nm = name.lower()
        if target_style == "elven":
            if any(d in ["аэ", "эа", "иэ", "эи", "иа", "аи", "ио", "ои"] for d in dips):
                comments.append("эльфийские дифтонги (аэ, иэ, ио)")
            if nm.endswith(("эль", "иэль", "иль", "эл")):
                comments.append("окончание -эль/-иэль/-ил — характерно для эльфийских")
            if "ль" in nm:
                comments.append("плавное -ль- между согласными")
            if nm.startswith(("а", "э", "и", "л", "с", "ф", "т")):
                comments.append("типичная начальная буква")
        elif target_style == "orcish":
            if any(c in nm for c in "гкхрзшщж"):
                comments.append("гортанные/рычащие согласные")
            if nm.endswith(("г", "к", "ш", "з", "х", "р")):
                comments.append("жёсткое окончание")
            if "рр" in nm or "гр" in nm or "кр" in nm:
                comments.append("сочетание -гр-/-кр-/-рр- — орочья черта")
        elif target_style == "northern":
            if nm.endswith(("ар", "ир", "ур", "альд", "грим", "хейм")):
                comments.append("скандинавское окончание")
            if nm.startswith(("т", "х", "б", "с", "и", "о", "м", "р", "э")):
                comments.append("характерная начальная буква")
            if "рг" in nm or "ль" in nm or "рн" in nm:
                comments.append("сочетание -рг-/-рн- — северное")
        elif target_style == "southern":
            if nm.endswith(("о", "а", "ина", "елла", "ино")):
                comments.append("романское окончание -о/-а/-ина")
            if "нь" in nm or "ль" in nm or "нь" in nm or "чч" in nm:
                comments.append("мягкие сочетания (нь, ль)")
            if any(c in nm for c in "чнь"): 
                comments.append("типичные южные звуки (ч, нь)")
        elif target_style == "human":
            if nm.endswith(("д", "н", "р", "т", "с", "льф", "ард")):
                comments.append("западноевропейское окончание")
            if any(c in nm for c in "аеиоуы"):
                comments.append("обычные гласные")

        return {
            "name": name,
            "syllable_split": "·".join(syl_parts) if syl_parts else name,
            "syllable_count": syl_count,
            "pattern": pattern,
            "diphthongs": dips,
            "ending": ending,
            "filters_passed": filters_passed,
            "comments": comments,
            "style_probs": style_probs,
            "scores": {
                "style": style_score, "novelty": nov,
                "plausibility": plaus, "total": total,
            },
        }