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

def generate_negative_samples(triples, num_samples=1000):
    """
    Генерирует отрицательные примеры - тройки, которых нет в исходном датасете.
    """
    subjects = list(set([t[0] for t in triples]))
    predicates = list(set([t[1] for t in triples]))
    objects = list(set([t[2] for t in triples]))

    existing_triples = set(triples)
    negative_samples = set()

    while len(negative_samples) < num_samples:
        subj = random.choice(subjects)
        pred = random.choice(predicates)
        obj = random.choice(objects)
        triple = (subj, pred, obj)
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
    positive_samples_downsampled = stratified_downsample(positive_samples, fraction=0.3)
    print(f"После уменьшения положительных примеров: {len(positive_samples_downsampled)}")

    negative_triples = generate_negative_samples(triples, num_samples=1000)
    negative_samples = [(s, p, o, 0) for s, p, o in negative_triples]

    merged = positive_samples_downsampled + negative_samples
    save_dataset(merged, output_file)

    print(f"Объединённый датасет сохранён в {output_file}")

if __name__ == "__main__":
    main()
