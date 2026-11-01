
"""Пайплайн для городов. Без гендера, без скор-моделей."""
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ml.lstm_gen import STYLES, load_lstm
from ml.lstm_gen import sample as lstm_sample
from ml.report import count_syllables, cv_pattern


@dataclass
class GeneratedCity:
    name: str
    syllables: int
    pattern: str


class CityPipeline:
    def __init__(self, models_dir: Optional[Path] = None):
        self.models_dir = models_dir or (
            Path(__file__).resolve().parent.parent / "models"
        )
        ckpt_path = self.models_dir / "city_lstm.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Модель городов не найдена: {ckpt_path}. "
                f"Запусти python -m ml.train_city_lstm"
            )
        self.model, self.itos, self.stoi = load_lstm(ckpt_path)

    def generate(self, style: str, syllables: int, count: int,
                 seed: Optional[int] = None, temperature: float = 0.8,
                 top_k: int = 40, top_p: float = 0.9,
                 n_candidates: int = 100, max_iterations: int = 3
                 ) -> list[GeneratedCity]:
        if style not in STYLES:
            raise ValueError(f"Неизвестный стиль: {style!r}")
        if not (1 <= syllables <= 8):
            raise ValueError("syllables должен быть 1..8")
        if not (1 <= count <= 50):
            raise ValueError("count должен быть 1..50")

        base_seed = seed if seed is not None else random.Random().randint(0, 2**31 - 1)

        results: list[GeneratedCity] = []
        seen: set[str] = set()
        iteration = 0

        while len(results) < count and iteration < max_iterations:
            for i in range(n_candidates):
                cand_seed = base_seed + iteration * n_candidates + i
                # Города гендера не имеют — передаём "male" как заглушку
                name = lstm_sample(self.model, self.itos, self.stoi,
                                   style, "male", temperature, top_k,
                                   top_p, cand_seed)
                if not name:
                    continue
                name = name[0].upper() + name[1:].lower()
                if name in seen:
                    continue
                if count_syllables(name) != syllables:
                    continue
                seen.add(name)
                results.append(GeneratedCity(
                    name=name,
                    syllables=count_syllables(name),
                    pattern=cv_pattern(name),
                ))
            iteration += 1

        random.Random(base_seed).shuffle(results)
        return results[:count]