# Сборка NameForge.exe

## Что нужно

- Обученные модели в `models/` (запусти `python -m ml.full_retrain`)
- PyInstaller: `pip install pyinstaller`

## Сборка

```powershell
python build/build_exe.py
```

Результат: dist/NameForge/NameForge.exe + папка \_internal/ с зависимостями.

Как распространять
Заархивируй папку dist/NameForge/ в ZIP (~700 МБ)

Отправь другу / перенеси на флешку

Друг распаковывает → двойной клик NameForge.exe

Python не нужен, работает офлайн

Как пользоваться
Двойной клик NameForge.exe:

Откроется чёрное окно консоли (это нормально — для диагностики)

Через 3-5 секунд откроется браузер с приложением

Для закрытия: закрой и окно браузера, и чёрное окно консоли

Где хранятся данные
Избранное, feedback, экспорты — dist/NameForge/user_data/ (рядом с exe)

Модели, датасет, тема — внутри \_internal/ (не трогай)

Пересобрать после изменения моделей
powershell
python -m ml.full_retrain
python build/build_exe.py
text

---

## Шаг 13. Прогоняем тесты (быстрая проверка)

```powershell
pytest tests\ -v
Ожидаемо: ~100 passed, 2 skipped (E2E, если моделей нет — но у тебя есть).

Если что-то падает (особенно test_storage — там могло сломаться из-за смены путей) — пришли вывод.

```
