import numpy as np
import requests
from triplets import load_labeled_triples, load_triples
from triplets import load_json_from_file

triples_raw_all, entities, relations = load_triples('dataset/relations_ru.txt')
relation_embeddings = np.load('embeddings/relation_embeddings.npy')
entity_embeddings = np.load('embeddings/entity_embeddings.npy')

entity2id = load_json_from_file('embeddings/entity2id.txt')
relation2id = load_json_from_file('embeddings/relation2id.txt')

triples_raw, labels = load_labeled_triples('dataset/relations_ru_train.tsv', entity2id, relation2id)

def retrieve_candidates(head, relation, top_m=5):
    """Извлекает кандидатов на основе эмбеддингов."""
    if head not in entity2id or relation not in relation2id:
        return []

    head_id = entity2id[head]
    relation_id = relation2id[relation]
    head_emb = entity_embeddings[head_id]
    relation_emb = relation_embeddings[relation_id]
    query = head_emb + relation_emb
    dists = np.linalg.norm(entity_embeddings - query, axis=1)

    candidates = []
    for idx in np.argsort(dists):
        cand = entities[idx]

        if cand != head:
            candidates.append(cand)
        if len(candidates) == top_m:
            break
    return candidates


def get_ego_graph(head):
    """Получает эго-граф для заданной сущности."""
    return [f"{h}-{r}-{t}" for h, r, t in triples_raw if h == head][:3]


def build_prompt(head, relation, candidates):
    """
    Строит промпт для LLM с четкой инструкцией и примером формата ответа.
    """
    context = "\n".join(get_ego_graph(head))
    prompt = (
        f"Контекст:\n{context}\n\n"
        f"Задание:\n"
        f"Дана неполная тройка: {head} - {relation} - ?\n"
        f"Варианты кандидатов для объекта:\n"
        + "\n".join(f"- {c}" for c in candidates) +
        "\n\n"
        "Выбери ОДНОГО наиболее вероятного кандидата (URI) из списка и верни ТОЛЬКО его URI без комментариев и пояснений.\n"
        "Если не можешь выбрать, верни только 'None'.\n"
        "Пример формата ответа:\n"
        "http://ru.dbpedia.org/resource/Some_Entity\n"
    )
    return prompt


def rerank_with_llm(prompt):
    """Отправляет запрос к LLM и получает ответ."""
    try:
        response = requests.post(
            "http://172.25.96.1:1234/v1/completions",
            json={
                "prompt": prompt,
                "temperature": 0,
                "max_tokens": 150
            },
            timeout=10
        )
        response.raise_for_status()
        text = response.json()['choices'][0]['text'].strip()
        return text
    except Exception as e:
        print(f"Ошибка при запросе к LLM: {e}")
        return "Ошибка при ранжировании кандидатов."

import re

def parse_llm_response(response_text, candidates):
    """
    Парсит ответ LLM и возвращает наиболее вероятного tail-кандидата.
    """
    response_text = response_text.strip()
    if response_text.lower() == 'none':
        return None

    response_text = re.sub(r'[\[\]\(\)\'\"\`]', '', response_text)

    for cand in candidates:
        if cand == response_text:
            return cand

    for cand in candidates:
        if cand in response_text:
            return cand

    for line in response_text.splitlines():
        line = line.strip()
        for cand in candidates:
            if cand == line:
                return cand
            if cand in line:
                return cand

    return None

def add_triple_to_dataset(head, relation, tail, file_path):
    """
    Добавляет новый триплет в датасет и сохраняет его.
    """
    if (head, relation, tail) not in triples_raw:
        triples_raw.append((head, relation, tail))
        with open(file_path, 'w+', encoding='utf-8') as f:
            f.write(f"{head}\t{relation}\t{tail}\n")

def knowledge_graph_completion_and_add(head, relation, file_path):
    tail = knowledge_graph_completion(head, relation)

    if tail:
        add_triple_to_dataset(head, relation, tail, file_path)

        return tail
    else:
        print("Не удалось распарсить ответ LLM.")

def knowledge_graph_completion(head, relation):
    """Основная функция для завершения графа знаний."""
    existing = [t for h, r, t in triples_raw if h == head and r == relation]
    if existing:
        return existing[0]

    candidates = retrieve_candidates(head, relation)

    if not candidates:
        return None

    prompt = build_prompt(head, relation, candidates)

    llm_response = rerank_with_llm(prompt)
    tail = parse_llm_response(llm_response, candidates)

    return tail

output_file = 'dataset/relations_ru_completed.txt'

for triple in triples_raw_all:
    result = knowledge_graph_completion_and_add(triple[0], triple[1], output_file)