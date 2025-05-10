import numpy as np
import requests
from triplets import load_labeled_triples, load_triples, load_json_from_file
from sklearn.model_selection import train_test_split
import asyncio
import re

triples_raw_all, entities, relations = load_triples('dataset/relations_ru.txt')
relation_embeddings = np.load('embeddings/relation_embeddings.npy')
entity_embeddings = np.load('embeddings/entity_embeddings.npy')

entity2id = load_json_from_file('embeddings/entity2id.txt')
relation2id = load_json_from_file('embeddings/relation2id.txt')

triples_raw, labels = load_labeled_triples('dataset/relations_ru_train.tsv')

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


def get_ego_graph(entity, triples, max_edges=3):
    """Получает эго-граф (до max_edges) для заданной сущности."""
    return [f"{h}-{r}-{t}" for h, r, t in triples if h == entity or t == entity][:max_edges]

def build_prompt(head, relation, candidates, triples=triples_raw_all):
    """
    Строит промпт для LLM:
    - Включает эго-граф head и каждого кандидата (tail)
    - Явно просит выбрать только из списка
    - Добавляет пример формата ответа
    """
    context_head = "\n".join(get_ego_graph(head, triples))
    context_tails = []
    for cand in candidates:
        ego = get_ego_graph(cand, triples)
        if ego:
            context_tails.append(f"Эго-граф кандидата {cand}:\n" + "\n".join(ego))
    context_tails_str = "\n\n".join(context_tails)
    prompt = (
        f"Контекст:\n"
        f"Эго-граф head-сущности ({head}):\n{context_head}\n\n"
        f"{context_tails_str}\n\n"
        f"Задание:\n"
        f"Дана неполная тройка: {head} - {relation} - ?\n"
        f"Варианты кандидатов для объекта:\n"
        + "\n".join(f"- {c}" for c in candidates) +
        "\n\n"
        "Выбери ОДНОГО наиболее вероятного кандидата (URI) из списка и верни ТОЛЬКО его.\n"
        "Пример формата ответа:\n"
        f"{candidates[0]}"
    )
    return prompt

def evaluate_candidate_retrieval(test_triples, top_n_values=[1, 3, 5, 10]):
    """
    Оценивает качество функции retrieve_candidates на тестовых триплетах.
    Возвращает метрики Hits@N, MRR и Mean Rank.
    """
    hits_at_n = {n: 0 for n in top_n_values}
    reciprocal_ranks = []
    mean_ranks = []
    total = 0
    for h, r, t in test_triples:
        if h not in entity2id or r not in relation2id or t not in entity2id:
            continue
        all_candidates = retrieve_candidates(h, r, top_m=len(entities)-1)
        if t in all_candidates:
            rank = all_candidates.index(t) + 1
            reciprocal_ranks.append(1.0 / rank)
            mean_ranks.append(rank)
            for n in top_n_values:
                if rank <= n:
                    hits_at_n[n] += 1
        else:
            mean_ranks.append(len(entities))
            reciprocal_ranks.append(0)
        total += 1
    metrics = {
        "MRR": sum(reciprocal_ranks) / total if total else 0,
        "MeanRank": sum(mean_ranks) / total if total else 0
    }
    for n in top_n_values:
        metrics[f"Hits@{n}"] = hits_at_n[n] / total if total else 0
    return metrics

async def send_to_llm(prompt, api_url):
    """Отправляет промпт в LLM API и возвращает ответ."""
    try:
        resp = requests.post(api_url, json={"prompt": prompt, "max_tokens": 50})
        resp.raise_for_status()
        return resp.json()["choices"][0]["text"].strip()
    except Exception as e:
        print(f"Ошибка при обращении к LLM API: {e}")
        return ""

def parse_llm_answer(response, candidates):
    """
    Извлекает одного кандидата из списка candidates из ответа LLM.
    Работает устойчиво к разным форматам (одиночный URI, пункт списка, кавычки, лишние слова).
    """
    response = response.strip().lower()

    for c in candidates:
        if response == c.lower():
            return c

    for c in candidates:
        pattern = r'[\s"\'>-]' + re.escape(c.lower()) + r'[\s"\'.<,-]'
        if re.search(pattern, f' {response} '):
            return c

    for c in candidates:
        if c.lower() in response:
            return c

    lines = response.splitlines()
    for line in lines:
        line = line.strip('-:> ').strip()
        for c in candidates:
            if line == c.lower():
                return c
            if c.lower() in line:
                return c

    return None

async def evaluate_llm_with_transe_candidates(test_triples, llm_api_url, top_m=5):
    """Оценивает, насколько хорошо LLM выбирает правильный ответ из кандидатов TransE."""
    correct_count = 0
    total = 0
    for h, r, t in test_triples:
        if h not in entity2id or r not in relation2id or t not in entity2id:
            continue
        candidates = retrieve_candidates(h, r, top_m=top_m)
        if t not in candidates:
            continue
        prompt = build_prompt(h, r, candidates)
        response = await send_to_llm(prompt, llm_api_url)
        chosen = parse_llm_answer(response, candidates)
        if chosen is not None and chosen == t:
            correct_count += 1
        total += 1
    llm_accuracy = correct_count / total if total > 0 else 0
    return {
        "LLM_Accuracy": llm_accuracy,
        "Total_Evaluated": total,
        "Correct_Count": correct_count
    }

def main():
    train_triples, test_triples, train_labels, test_labels = train_test_split(
        triples_raw, labels, test_size=0.2, random_state=42
    )
    print("Оценка качества извлечения кандидатов (TransE)...")
    metrics = evaluate_candidate_retrieval(test_triples, top_n_values=[1, 3, 5, 10])
    print("\nМетрики TransE:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    llm_api_url = "http://172.21.32.1:1234/v1/completions"
    print("\nОценка качества LLM с кандидатами TransE...")
    llm_metrics = asyncio.run(evaluate_llm_with_transe_candidates(
    test_triples[:1000],
        llm_api_url, top_m=5
    ))

    print("\nМетрики LLM:")

    for metric, value in llm_metrics.items():
         print(f"{metric}: {value}")

if __name__ == "__main__":
    main()
