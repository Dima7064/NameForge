# NameForge — Кузница Имён

Генератор фэнтези-имён с ML-пайплайном. Два режима: «Для себя» (только имена) и «Для защиты» (с метриками).

## Статус

🚧 В разработке. Этапы 0–7 готовы. Writer-mode UI работает.

## Стек

- **ML:** PyTorch, ONNX, scikit-learn
- **UI:** Gradio (десктоп), HTML/JS + onnxruntime-web (веб, план)
- **Сборка:** PyInstaller (план)
- **Хранение:** JSON

## Установка

```powershell
git clone https://github.com/ТВОЙ_НИК/NameForge.git
Set-Location NameForge
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
Запуск UI
powershell
python -m app.main
Откроется браузер с writer-режимом: фильтры слева, карточки в центре, избранное справа. Избранное сохраняется в data/favorites.json.

Запуск CLI
powershell
python -m app.cli --style elven --syllables 2 --count 5
Структура проекта
text
NameForge/
├── app/         # UI и пайплайн (Gradio)
├── ml/          # Обучение, модели, скоринг
├── data/        # Датасет, избранное
├── models/      # Обученные веса (.pt)
├── web/         # Браузерная версия (план)
├── build/       # Скрипты сборки .exe (план)
├── assets/      # Тема, иконки
├── tests/       # Тесты (pytest)
├── notebooks/   # Jupyter-эксперименты (план)
└── docs/        # Документация
Отклонения от ТЗ
Слоги: 1–6 вместо 2–4. Обосновано данными (docs/data_report.md): в реальных именах 1–6 слогов.

Количество: 1–50 по ТЗ; UI выдерживает 50 слотов.

Шрифты: используются системные (Georgia, Consolas) вместо Cinzel/EB Garamond — Cinzel не поддерживает кириллицу.

Переключатель режимов: появится на этапе 8 вместе с defense-mode.

План разработки
См. docs/roadmap.md (будет добавлено).

Лицензия
MIT
```
