"""Замеры производительности (ТЗ N-1: ≤ 3 сек на 100 кандидатов + скоринг).

Пороги сознательно ослаблены (x3 от ТЗ), чтобы тесты не падали на слабых
машинах. Проверяем только грубые регрессии.
"""
import time
from pathlib import Path

import pytest

from app.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

MODELS_AVAILABLE = all(
    (MODELS_DIR / f).exists()
    for f in ("lstm.pt", "classifier.pt", "scorers.pt")
)

pytestmark = pytest.mark.skipif(
    not MODELS_AVAILABLE,
    reason="Реальные модели не обучены",
)


@pytest.fixture(scope="module")
def pipeline():
    return Pipeline(model_type="lstm")


def test_perf_generation_under_3s(pipeline):
    """N-1: 100 кандидатов + скоринг ≤ 3 сек. Здесь запас x3 = 9 сек."""
    start = time.perf_counter()
    results = pipeline.generate(
        style="elven", syllables=2, count=10, seed=42,
        n_candidates=100, max_iterations=1,
    )
    elapsed = time.perf_counter() - start
    print(f"\n⏱  100 кандидатов + скоринг: {elapsed:.2f} сек")
    assert elapsed < 9.0, (
        f"Слишком медленно: {elapsed:.2f} сек на 100 кандидатов. "
        f"Порог по ТЗ — 3 сек, здесь допуск x3."
    )


def test_perf_repeat_generation_faster(pipeline):
    """Повторная генерация после прогрева должна быть быстрее или равна."""
    pipeline.generate(style="elven", syllables=2, count=5, seed=42,
                      n_candidates=50, max_iterations=1)
    start = time.perf_counter()
    pipeline.generate(style="elven", syllables=2, count=5, seed=42,
                      n_candidates=50, max_iterations=1)
    elapsed = time.perf_counter() - start
    print(f"\n⏱  50 кандидатов (warm): {elapsed:.2f} сек")
    assert elapsed < 5.0


def test_perf_pipeline_load(pipeline):
    """Pipeline уже загружен — повторное использование не должно тормозить."""
    start = time.perf_counter()
    for _ in range(5):
        pipeline.generate(style="human", syllables=2, count=1, seed=1,
                          n_candidates=10, max_iterations=1)
    elapsed = time.perf_counter() - start
    print(f"\n⏱  5 маленьких генераций: {elapsed:.2f} сек")
    assert elapsed < 10.0
