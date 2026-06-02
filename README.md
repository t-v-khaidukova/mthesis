# Онтологическая модель морфологической дизамбигуации удмуртского языка

Репозиторий содержит прототип системы для построения и применения онтологической модели лингвистических знаний на основе функционального графа знаний. Проект является адаптацией двухэтапной архитектуры SOLAR к задаче морфологической разметки малоресурсного языка, а именно удмуртского языка.

Основная задача системы — не заменить морфологический анализатор UniParser, а выбрать наиболее подходящий морфологический разбор среди кандидатов, уже предложенных UniParser. Для этого используется онтологический слой: TBox задает общие классы, свойства и правила, а ABox описывает конкретное предложение, его токены, кандидатные анализы и контекстные признаки.

## Настройка окружения

Создайте файл `.env` в корне проекта и добавьте ключ для доступа к модели:

```
OPENAI_API_KEY=
FIREWORKS_API_KEY=
ANTHROPIC_API_KEY=
GOOGLE_API_KEY=
```

## Структура репозитория

| Путь | Назначение |
|------|------------|
| `stage_udmurt_one_sourcegrounded.py` | первый этап: построение TBox и Python-интерпретатора на основе грамматики, CG3 и обучающей выборки |
| `stage_udmurt_two_sourcegrounded.py` | второй этап: построение ABox для тестовых предложений, запуск интерпретатора и расчет метрик |
| `udmurt_corpus_parser_sourcegrounded.py` | преобразование корпусных данных и кандидатов UniParser во внутреннее ABox-представление |
| `runs/interpreter_claude_fixed.py` | доработанная версия интерпретатора для удмуртской морфологической дизамбигуации |
| `runs/tbox_claude.json` | пример сгенерированного TBox |
| `USAGE.md` | инструкция по проверке онтологии в Protégé/HermiT и загрузке RDF/OWL в Neo4j |
| `eval/train.csv` | обучающая выборка |
| `eval/test.csv` | тестовая выборка |
| `eval/udmurt_grammar.txt` | грамматическое описание удмуртского языка |
| `eval/udmurt_disambiguation.cg3` | правила Constraint Grammar для удмуртского языка |
| `runs/` | результаты запусков разных моделей, сгенерированные TBox, интерпретаторы и сводные метрики |
| `solar/` | базовые модули SOLAR, используемые для извлечения и валидации знаний |

## Использование

### Этап I: Получение знаний

Сгенерируйте онтологию TBox и интерпретатор из грамматического описания удмуртского языка, правил CG3 и примеров из обучающей выборки:

```bash
uv run python stage_udmurt_one_sourcegrounded.py \
  --model claude \
  --statute ./eval/udmurt_grammar.txt \
  --cg3 ./eval/udmurt_disambiguation.cg3 \
  --train ./eval/train.csv \
  --output-dir runs \
  --output-prefix udmurt \
  --save-prompt runs/stage1_prompt_claude.txt \
  --no-cache
```

Arguments:

- `--model`: Модель для использования (gpt-5.1, claude etc.)
- `--statute`: Путь к входному файлу с грамматическим описанием удмуртского языка
- `--cg3`: Путь к входному файлу с правилами CG3
- `--train`: Путь к обучающему набору
- `--output-dir`: Выходной путь для сгенерированных TBox JSON и кода интерпретатора
- `--no-cache`: Отключить кэш LLM

## Этап II: Применение знаний

Оцените дизамбигуацию морфологической разметки на тестовом наборе данных:

```bash
uv run python stage_udmurt_two_sourcegrounded.py \
  --model claude \
  --corpus ./eval/test.csv \
  --tbox-path runs/tbox_udmurt_claude.json \
  --tbox-interpreter-path runs/interpreter_udmurt_claude.py \
  --output-dir runs \
  --no-cache
```

Arguments:

- `--model`: Модель для оценивания
- `--test`: Тестовый набор данных
- `--tbox-path`: Путь к TBox JSON (for solar mode)
- `--tbox-interpreter-path`: Путь к интерпретеру TBox
- `--output-dir`: Выходной путь для сгенерированных JSON
- `--no-cache`: Отключить кэш LLM
