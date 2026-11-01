# NameForge — Кузница Имён

Генератор фэнтези-имён с ML-пайплайном и двумя режимами работы: «Для себя» (только имена) и «Для защиты» (с метриками и объяснениями).

## Статус

🚧 В разработке. Этап 0: инфраструктура готова.

## Стек

- **ML:** PyTorch, ONNX, scikit-learn
- **UI:** Gradio (десктоп), HTML/JS + onnxruntime-web (веб)
- **Сборка:** PyInstaller
- **Хранение:** JSON / IndexedDB

## Установка (для разработчика)

```powershell
git clone https://github.com/ТВОЙ_НИК/NameForge.git
Set-Location NameForge
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

Запуск
(будет добавлено на этапе 7)


Структура проекта
NameForge/
├── app/         # UI и пайплайн (Gradio)
├── ml/          # Обучение, модели, экспорт в ONNX
├── data/        # Датасет имён (JSON)
├── models/      # Обученные веса (.pt, .onnx)
├── web/         # Браузерная версия
├── build/       # Скрипты сборки .exe
├── assets/      # Иконки, тема, шрифты
├── tests/       # Тесты (pytest)
├── notebooks/   # Jupyter-эксперименты
└── docs/        # Документация

План разработки
Полный план в 14 этапов — см. docs/roadmap.md (будет добавлено).

Лицензия
MIT
```
