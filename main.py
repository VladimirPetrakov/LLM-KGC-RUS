import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
from triplets import load_triples
from triplets import load_json_from_file

# Загрузка данных
triples_raw, entities, relations = load_triples('dataset/relations_ru.txt')
relation_embeddings = np.load('embeddings/relation_embeddings.npy')
entity_embeddings = np.load('embeddings/entity_embeddings.npy')

entity2id = load_json_from_file('embeddings/entity2id.txt')
relation2id = load_json_from_file('embeddings/relation2id.txt')

def retrieve_candidates(head, relation, top_m=5):
    """Извлекает кандидатов на основе эмбеддингов."""
    if head not in entity2id or relation not in relation2id:
        return []

    head_id = entity2id[head]
    relation_id = relation2id[relation]
    head_emb = entity_embeddings[head_id]
    relation_emb = relation_embeddings[relation_id]
    query_emb = head_emb + relation_emb

    sims = cosine_similarity(query_emb.reshape(1, -1), entity_embeddings)[0]

    known_tails = set(t for h, r, t in triples_raw if h == head and r == relation)

    candidates = []
    for idx in sims.argsort()[::-1]:
        cand = entities[idx]

        if cand != head and cand not in known_tails:
            candidates.append(cand)
        if len(candidates) == top_m:
            break
    return candidates


def get_ego_graph(head):
    """Получает эго-граф для заданной сущности."""
    return [f"{h}-{r}-{t}" for h, r, t in triples_raw if h == head][:3]


def build_prompt(head, relation, candidates):
    """Строит промпт для LLM."""
    context = "\n".join(get_ego_graph(head))
    prompt = (
        f"Контекст:\n{context}\n\n"
        f"Запрос: {head} - {relation} - ?\n"
        f"Варианты кандидатов: {', '.join(candidates)}\n\n"
        "Пожалуйста, отранжируйте кандидатов по вероятности и укажите один самый вероятный ответ."
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
    # 1. Если в ответе есть строка, совпадающая с кандидатом - берем ее
    for cand in candidates:
        if cand in response_text:
            return cand

    # 2. Если ответ - просто URI (одна строка)
    lines = [line.strip() for line in response_text.splitlines() if line.strip()]
    for line in lines:
        if line in candidates:
            return line

    # 3. Если ответ - "1. ..." или "1) ..."
    for line in lines:
        m = re.match(r"1[\.\)]\s*(\S+)", line)
        if m and m.group(1) in candidates:
            return m.group(1)

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
    candidates = retrieve_candidates(head, relation)

    if not candidates:
        return None

    prompt = build_prompt(head, relation, candidates)

    llm_response = rerank_with_llm(prompt)
    tail = parse_llm_response(llm_response, candidates)

    return tail

output_file = 'dataset/relations_ru_completed.txt'

result = knowledge_graph_completion_and_add("http://ru.dbpedia.org/resource/Interview_(альбом)",
                                    "http://dbpedia.org/ontology/artist",
                                            output_file)

print("\nРезультат завершения графа знаний:")
print(result)