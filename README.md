# Knowledge Graph Embeddings & Completion (Russian Language)

Этот проект предназначен для построения, обучения и расширения графа знаний на русском языке. Он включает обработку данных, генерацию обучающего датасета, обучение модели TransE, а также интеллектуальное дополнение триплетов с помощью LLM.

---

## 📁 Структура проекта

- **build_russian_dataset.py**  
  Преобразует URI в русскоязычные метки на основе файла меток.
- **generate_train_dataset.py**  
  Генерирует обучающий датасет с позитивными и негативными примерами, выполняет стратифицированное уменьшение положительных примеров.
- **embeddings.py**  
  Обучает модель TransE, реализует hard negative sampling, сохраняет эмбеддинги, оценивает качество.
- **generate_llm_finetune_dataset.py**  
  Генерирует датасет для дообучения модели
- **train_llm.py**  
  Доубучение модели
- **main.py**  
  Извлекает кандидатов для tail, строит промпт для LLM, парсит ответ, рассчитывает различные метрики.
- **dataset/**  
  Каталог с исходными и промежуточными файлами данных.
- **embeddings/**  
  Каталог с эмбеддингами и отображениями.
- **finetuned_model/**  
  Каталог с дообученной моделью

---

## 🚀 Запуск по шагам

### 1. Подготовка данных

1. Поместите исходные файлы `labels_ru.txt` (метки) и `relations_en.txt` (триплеты в URI) в папку `dataset/`.

2. Преобразуйте URI в русскоязычные метки:
   - Запустите:
     ```
     python build_russian_dataset.py
     ```
   - Результат: файл `dataset/relations_ru.txt` с триплетами на русском языке.

### 2. Генерация обучающего датасета

1. Сгенерируйте обучающий датасет с позитивными и негативными примерами:
   - Запустите:
     ```
     python generate_train_dataset.py
     ```
   - Скрипт выполнит стратифицированное уменьшение положительных примеров, сгенерирует негативные примеры и создаст файл `dataset/relations_ru_train.tsv`.

2. (Опционально) Проанализируйте баланс классов и уникальность триплетов:
   - Используйте функцию `analyze_dataset` внутри скрипта или добавьте вызов в конец файла.

### 3. Обучение эмбеддингов

1. Обучите модель TransE на подготовленном датасете:
   - Запустите:
     ```
     python embeddings.py
     ```
   - Скрипт обучит модель, сохранит эмбеддинги сущностей и отношений в папку `embeddings/`, а также выведет accuracy на тестовой выборке.

### 4. Генерация датасета для обучения LLM

1. Генерация датасета для обучения:
   - Запустите:
     ```
     python generate_llm_finetune_dataset.py
     ```
   - Скрипт генерирует датасет для дообучения модели в папку `dataset/`

### 5. Обучение LLM

1. Обучите LLM:
   - Запустите:
     ```
     python train_llm.py
     ```
   - Скрипт выгрузит файлы модели в папку `finetuned_model/`

### 6. Завершение графа знаний и интеграция с LLM

1. Запустите основной скрипт для дополнения триплетов и ранжирования кандидатов с помощью LLM:
   - Запустите:
     ```
     python main.py
     ```
   - Скрипт:
     - Загружает эмбеддинги и отображения.
     - Извлекает кандидатов для tail на основе эмбеддингов.
     - Формирует промпт для LLM.
     - Отправляет запрос к LLM (проверьте и настройте адрес API в функции `rerank_with_llm`).
     - Парсит ответ LLM и рассчитывает метрики.

---

## 🛠️ Требования

- Python 3.7+
- Библиотеки:  
  - torch  
  - numpy  
  - requests  
  - scikit-learn
