# RAG по ТФКП

Роман Фельдман, Б05-305

Python 3.11. PDF → блоки текста → BM25 → Qwen3.5-4B → ответ, обоснование и ссылки. Поиск работает на CPU, модель — через Chat Completions API.

## Подготовка

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python download_textbook.py --output .local/textbook.pdf
python prepare_data.py --pdf .local/textbook.pdf --corpus .local/corpus.jsonl --questions dev_questions.jsonl --gold dev_gold.jsonl
python prepare_own.py --corpus .local/corpus.jsonl
python build_index.py --corpus .local/corpus.jsonl --output ./index --config config.json
```

[Источник](https://new.mipt.ru/upload/mathematics/study/uchebniki/%D0%9B_%D0%A2%D0%A4%D0%9A%D0%9F_%D0%93%D0%BE%D1%80%D1%8F%D0%B9%D0%BD%D0%BE%D0%B2_%D0%9F%D0%BE%D0%BB%D0%BE%D0%B2%D0%B8%D0%BD%D0%BA%D0%B8%D0%BD_2025.pdf). Корпус, индекс, веса и окружение в Git не включены. Зависимости, веса и токенизатор скачиваются до предсказаний.

## Локальная модель на Mac

```bash
python -m pip install -r requirements-local.txt
python setup_local.py --bits 16 --output .local/qwen-mlx-16 --cache .cache/huggingface
python serve_local.py --model .local/qwen-mlx-16 --port 8082 --audit .local/endpoint-bf16.jsonl
```

Qwen работает в BF16. Ревизия, параметры генерации и лимит памяти 16 GB заданы в `runtime.json`.

## Предсказание

В другом терминале:

```bash
source .venv/bin/activate
export LLM_BASE_URL=http://127.0.0.1:8082/v1
export LLM_MODEL=Qwen/Qwen3.5-4B
export LLM_API_KEY=local
export SUBMISSION_ID=feldman-hw01
python predict.py --index ./index --questions dev_questions.jsonl --output dev_answers.jsonl --trace dev_trace.jsonl --config config.json
python validate_submission.py --questions dev_questions.jsonl --answers dev_answers.jsonl --trace dev_trace.jsonl --corpus .local/corpus.jsonl
```

Для другого сервера замените переменные окружения. Запросы содержат `X-Question-ID` и `X-Submission-ID`. Во время предсказаний сеть используется только для API.

## Оценка и эксперименты

Оценка сохраненных ответов с проверкой обоснований:

```bash
python evaluate.py --questions dev_questions.jsonl --gold dev_gold.jsonl --answers dev_answers.jsonl --trace dev_trace.jsonl --corpus .local/corpus.jsonl --name final --rationale-audit rationale_audit.csv
```

Новые прогоны и тесты:

```bash
python run_experiments.py --index ./index --questions dev_questions.jsonl --gold dev_gold.jsonl --config config.json
python -m unittest discover -s tests
```

Режимы: `no_context`, `baseline`, `full`, `no_math`, `no_stemming`, `final`. Для новых ответов проверку обоснований нужно повторить.

## Ресурсы

Проверено на M4 16 CPU, 64 GB RAM. Индексы с токенизатором — 16.13 MiB; пик памяти LLM — 10.214 GB. Архитектура, результаты и ошибки — в [отчете](report.md).
