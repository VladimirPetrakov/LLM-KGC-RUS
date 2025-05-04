from sklearn.metrics.pairwise import cosine_similarity
import requests
import torch
import torch.nn as nn
import torch.optim as optim
import random

def load_triples(file_path):
    triples = []
    entities = set()
    relations = set()
    with open(file_path, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 3:
                continue
            h, r, t = parts
            triples.append((h, r, t))
            entities.add(h)
            entities.add(t)
            relations.add(r)
    return triples, sorted(entities), sorted(relations)

triples_raw, entities, relations = load_triples('dataset/relations_ru.txt')
entity2id = {e: i for i, e in enumerate(entities)}
relation2id = {r: i for i, r in enumerate(relations)}
triples = [(entity2id[h], relation2id[r], entity2id[t]) for h, r, t in triples_raw]
num_entities = len(entities)
num_relations = len(relations)

class TransE(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(TransE, self).__init__()
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        nn.init.xavier_uniform_(self.entity_embeddings.weight.data)
        nn.init.xavier_uniform_(self.relation_embeddings.weight.data)

    def forward(self, head, relation, tail):
        head_emb = self.entity_embeddings(head)
        relation_emb = self.relation_embeddings(relation)
        tail_emb = self.entity_embeddings(tail)
        score = (head_emb + relation_emb - tail_emb).norm(p=2, dim=1)
        return score

def generate_negative_samples(triples, num_entities):
    corrupted_triples = []
    for h, r, t in triples:
        if random.random() < 0.5:
            h_corrupt = random.randint(0, num_entities - 1)
            corrupted_triples.append((h_corrupt, r, t))
        else:
            t_corrupt = random.randint(0, num_entities - 1)
            corrupted_triples.append((h, r, t_corrupt))
    return corrupted_triples

def train_transe(triples, num_entities, num_relations, embedding_dim=50, learning_rate=0.01, epochs=10, batch_size=128):
    model = TransE(num_entities, num_relations, embedding_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MarginRankingLoss(margin=1.0)

    triples_tensor = torch.tensor(triples, dtype=torch.long)

    for epoch in range(epochs):
        model.train()

        perm = torch.randperm(len(triples))
        triples_tensor = triples_tensor[perm]

        epoch_loss = 0
        for i in range(0, len(triples), batch_size):
            batch = triples_tensor[i:i+batch_size]
            head = batch[:,0]
            relation = batch[:,1]
            tail = batch[:,2]

            corrupted_batch = generate_negative_samples(batch.tolist(), num_entities)
            corrupted_batch = torch.tensor(corrupted_batch, dtype=torch.long)
            head_corrupt = corrupted_batch[:,0]
            relation_corrupt = corrupted_batch[:,1]
            tail_corrupt = corrupted_batch[:,2]

            optimizer.zero_grad()
            score_pos = model(head, relation, tail)
            score_neg = model(head_corrupt, relation_corrupt, tail_corrupt)

            y = torch.ones(len(score_pos))
            loss = criterion(score_pos, score_neg, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / (len(triples) / batch_size)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    return model.entity_embeddings.weight.data.cpu().numpy(), model.relation_embeddings.weight.data.cpu().numpy()

entity_embeddings, relation_embeddings = train_transe(
    triples, num_entities, num_relations, embedding_dim=50, learning_rate=0.01, epochs=20
)

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
