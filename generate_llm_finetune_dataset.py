import json

from main import retrieve_candidates, build_prompt
from triplets import load_labeled_triples, load_triples, load_json_from_file

def generate_llm_finetune_dataset(triples, entity2id, relation2id, triples_raw_all, entities, top_m=5, output_file="llm_finetune_dataset.jsonl"):
    """
    Формирует пары (prompt, completion) для дообучения LLM.
    """
    from tqdm import tqdm
    dataset = []
    for h, r, t in tqdm(triples, desc="Генерация датасета"):
        head_id = entity2id.get(h)
        relation_id = relation2id.get(r)
        if head_id is None or relation_id is None:
            continue

        candidates = retrieve_candidates(h, r, top_m=top_m)
        if t not in candidates:
            continue
        prompt = build_prompt(h, r, candidates, triples_raw_all)
        dataset.append({
            "prompt": prompt,
            "completion": t
        })
    with open(output_file, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Сохранено {len(dataset)} примеров в {output_file}")

triples_raw_all, entities, relations = load_triples('dataset/relations_ru.txt')
entity2id = load_json_from_file('embeddings/entity2id.txt')
relation2id = load_json_from_file('embeddings/relation2id.txt')
triples, labels = load_labeled_triples('dataset/relations_ru_train.tsv')

positive_triples = [tr for tr, label in zip(triples, labels) if label == 1]

generate_llm_finetune_dataset(
    triples=positive_triples,
    entity2id=entity2id,
    relation2id=relation2id,
    triples_raw_all=triples_raw_all,
    entities=entities,
    top_m=5,
    output_file="dataset/llm_finetune_dataset.jsonl"
)
