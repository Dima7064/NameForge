"""CLI-интерфейс NameForge.

Запуск (из корня проекта):
    python -m app.cli --style elven --syllables 3 --count 10
    python -m app.cli --style orcish --syllables 2 --count 5 --seed 42
    python -m app.cli --style northern --syllables 2 --model transformer
"""
import argparse
import sys

from app.pipeline import Pipeline
from ml.lstm_gen import STYLES


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NameForge — генератор фэнтези-имён с ML-пайплайном",
    )
    parser.add_argument("--style", required=True, choices=STYLES,
                        help="Стиль имён")
    parser.add_argument("--syllables", type=int, default=3,
                        help="Желаемое число слогов (1..10)")
    parser.add_argument("--count", type=int, default=10,
                        help="Сколько имён вернуть (1..50)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed для воспроизводимости")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Temperature (0 < t <= 2)")
    parser.add_argument("--top-k", type=int, default=40,
                        help="top-k сэмплирования (0 = отключить)")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="top-p сэмплирования (1.0 = отключить)")
    parser.add_argument("--model", choices=["lstm", "transformer"],
                        default="lstm", help="Тип генератора")
    parser.add_argument("--n-candidates", type=int, default=100,
                        help="Кандидатов за одну итерацию")
    parser.add_argument("--no-scores", action="store_true",
                        help="Показывать только имена (writer-режим)")

    args = parser.parse_args()

    try:
        pipeline = Pipeline(model_type=args.model)
    except (FileNotFoundError, ValueError) as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        results = pipeline.generate(
            style=args.style, syllables=args.syllables, count=args.count,
            seed=args.seed, temperature=args.temperature,
            top_k=args.top_k, top_p=args.top_p,
            n_candidates=args.n_candidates,
        )
    except ValueError as e:
        print(f"Ошибка параметров: {e}", file=sys.stderr)
        sys.exit(1)

    if not results:
        print(f"\nНе удалось сгенерировать имена с {args.syllables} слогами "
              f"в стиле {args.style}. Попробуй другое число слогов.")
        sys.exit(0)

    seed_info = f"seed={args.seed}" if args.seed is not None else "seed=random"
    print(f"\n{len(results)} имён | стиль: {args.style} | "
          f"слогов: {args.syllables} | модель: {args.model} | {seed_info}\n")

    if args.no_scores:
        for i, s in enumerate(results, 1):
            print(f"{i:2d}. {s.name}")
    else:
        for i, s in enumerate(results, 1):
            sc = s.scores
            print(
                f"{i:2d}. {s.name:16s} | "
                f"style={sc['style']:.2f} | "
                f"novelty={sc['novelty']:.2f} | "
                f"plaus={sc['plausibility']:.2f} | "
                f"total={sc['total']:.2f} | "
                f"[{s.breakdown['pattern']}]"
            )


if __name__ == "__main__":
    main()
