import numpy as nm
import torch
import torch.nn as nn
import torch.optim as optim
import random
from triplets import load_triples
from triplets import save_json_to_file

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



triples_raw, entities, relations = load_triples('dataset/relations_ru.txt')

entity2id = {e: i for i, e in enumerate(entities)}
relation2id = {r: i for i, r in enumerate(relations)}

save_json_to_file('embeddings/entity2id.txt', entity2id)
save_json_to_file('embeddings/relation2id.txt', relation2id)

triples = [(entity2id[h], relation2id[r], entity2id[t]) for h, r, t in triples_raw]
num_entities = len(entities)
num_relations = len(relations)

entity_embeddings, relation_embeddings = train_transe(
    triples, num_entities, num_relations, embedding_dim=50, learning_rate=0.01, epochs=20
)

nm.save('embeddings/relation_embeddings.npy', relation_embeddings)
nm.save('embeddings/entity_embeddings.npy', entity_embeddings)

