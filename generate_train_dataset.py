import random
from collections import defaultdict

def load_dataset(file_path):
    """
    Загружает датасет из TSV файла с тройками (subject, predicate, object).
    Возвращает список кортежей.
    """
    triples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                triples.append(tuple(parts))
    return triples

def stratified_downsample(positive_samples, fraction=0.3):
    """
    Выполняет стратифицированное уменьшение положительных примеров по признаку predicate.

    positive_samples: список кортежей (subject, predicate, object, label=1)
    fraction: доля, которую нужно оставить в каждой группе

    Возвращает список уменьшенных положительных примеров.
    """

    groups = defaultdict(list)
    for sample in positive_samples:
        predicate = sample[1]
        groups[predicate].append(sample)

    downsampled = []
    for predicate, samples in groups.items():
        sample_size = max(1, int(len(samples) * fraction))  # минимум 1 пример из группы
        chosen = random.sample(samples, sample_size)
        downsampled.extend(chosen)

    return downsampled

def generate_negative_samples(triples, entities, num_samples=1000):
    """
    Генерирует негативные примеры через подмену head или tail.
    """
    existing_triples = set(triples)
    negative_samples = set()
    triples_list = list(triples)
    while len(negative_samples) < num_samples:
        h, r, t = random.choice(triples_list)
        if random.random() < 0.5:
            h_corrupt = random.choice(entities)
            triple = (h_corrupt, r, t)
        else:
            t_corrupt = random.choice(entities)
            triple = (h, r, t_corrupt)
        if triple not in existing_triples and triple not in negative_samples:
            negative_samples.add(triple)
    return list(negative_samples)

def save_dataset(triples_with_labels, output_file):
    """
    Сохраняет датасет в TSV файл с четырьмя колонками:
    subject, predicate, object, label
    """
    with open(output_file, 'w+', encoding='utf-8') as f:
        for subj, pred, obj, label in triples_with_labels:
            f.write(f"{subj}\t{pred}\t{obj}\t{label}\n")

def main():
    input_file = 'dataset/relations_ru.txt'
    output_file = 'dataset/relations_ru_train.tsv'

    print(f"Загрузка датасета из {input_file}...")
    triples = load_dataset(input_file)

    positive_samples = [(s, p, o, 1) for s, p, o in triples]
    print(f"Всего положительных примеров: {len(positive_samples)}")

    print("Выполняем стратифицированное уменьшение положительных примеров...")
    positive_samples_downsampled = stratified_downsample(positive_samples, fraction=0.7)
    print(f"После уменьшения положительных примеров: {len(positive_samples_downsampled)}")

    entities = list(set([t[0] for t in triples] + [t[2] for t in triples]))
    negative_triples = generate_negative_samples(triples, entities, num_samples=len(positive_samples_downsampled))

    negative_samples = [(s, p, o, 0) for s, p, o in negative_triples]

    merged = positive_samples_downsampled + negative_samples
    save_dataset(merged, output_file)

    print(f"Объединённый датасет сохранён в {output_file}")

    positives = set(triples)
    negatives = set(negative_triples)

    intersect = positives & negatives
    print(f"Совпадающих троек: {len(intersect)}")

def analyze_dataset(tsv_path):
    triples = []
    labels = []
    with open(tsv_path, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 4:
                h, r, t, label = parts
                triples.append((h, r, t))
                labels.append(int(label))
    print("Всего примеров:", len(triples))
    print("Положительных:", sum(labels))
    print("Отрицательных:", len(labels) - sum(labels))
    print("Уникальных триплетов:", len(set(triples)))
    print("Дубликатов:", len(triples) - len(set(triples)))

if __name__ == "__main__":
    main()
    analyze_dataset('dataset/relations_ru_train.tsv')
