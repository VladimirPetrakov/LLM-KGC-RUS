import numpy as nm
from sklearn.metrics.pairwise import cosine_similarity
import requests
from triplets import load_triples
from triplets import load_json_from_file

triples_raw, entities, relations = load_triples('dataset/relations_ru.txt')

relation_embeddings = nm.load('embeddings/relation_embeddings.npy')
entity_embeddings = nm.load('embeddings/entity_embeddings.npy')

entity2id = load_json_from_file('embeddings/entity2id.txt')
relation2id = load_json_from_file('embeddings/relation2id.txt')

def retrieve_candidates(head, relation, top_m=5):
    if head not in entity2id or relation not in relation2id:
        return []

    head_id = entity2id[head]
    relation_id = relation2id[relation]

    head_emb = entity_embeddings[head_id]
    relation_emb = relation_embeddings[relation_id]

    query_emb = head_emb + relation_emb

    sims = cosine_similarity(query_emb.reshape(1, -1), entity_embeddings)[0]
    top_indices = sims.argsort()[-top_m:][::-1]

    candidates = [entities[i] for i in top_indices]

    return candidates

def build_prompt(head, relation, candidates):
    prompt = f"Дано: {head} - {relation} - ?\n"
    prompt += "Выберите наиболее вероятные варианты из списка:\n"
    for i, cand in enumerate(candidates, 1):
        prompt += f"{i}. {cand}\n"
    prompt += "Отранжируйте кандидатов по вероятности."
    return prompt

def rerank_with_llm(prompt):
    try:
        response = requests.post(
            "http://172.19.176.1:1234/v1/completions",
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

def knowledge_graph_completion(head, relation):
    candidates = retrieve_candidates(head, relation)
    if not candidates:
        return "Нет кандидатов для данного запроса."
    prompt = build_prompt(head, relation, candidates)
    ranked_candidates = rerank_with_llm(prompt)
    return ranked_candidates

result = knowledge_graph_completion("http://ru.dbpedia.org/resource/Unity_(альбом_Rage)", "http://dbpedia.org/ontology/producer")

print("\nРезультат завершения графа знаний:")

print(result)