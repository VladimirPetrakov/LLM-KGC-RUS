import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

class TransE(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(TransE, self).__init__()
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        nn.init.xavier_uniform_(self.entity_embeddings.weight.data)
        nn.init.xavier_uniform_(self.relation_embeddings.weight.data)

    def forward(self, head, relation, tail):
        head_emb = self.entity_embeddings(head)
        rel_emb = self.relation_embeddings(relation)
        tail_emb = self.entity_embeddings(tail)
        score = (head_emb + rel_emb - tail_emb).norm(p=2, dim=1)
        return score

class TriplesDataset(Dataset):
    def __init__(self, triples):
        self.triples = triples

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        triple = self.triples[idx]
        return torch.tensor(triple, dtype=torch.long)

def generate_hard_negative_samples_vectorized(model, pos_triples, num_entities, triples_set,
                                             num_hard_neg=1, max_candidates=20, device='cpu'):
    model.eval()
    hard_negatives = []

    entity_emb = model.entity_embeddings.weight.data.to(device)
    relation_emb = model.relation_embeddings.weight.data.to(device)

    for h, r, t in pos_triples:
        tail_candidates = [i for i in range(num_entities) if i != t and (h, r, i) not in triples_set]
        if len(tail_candidates) > max_candidates:
            tail_candidates = random.sample(tail_candidates, max_candidates)
        if tail_candidates:
            head_vec = entity_emb[h].unsqueeze(0)
            rel_vec = relation_emb[r].unsqueeze(0)
            tail_vecs = entity_emb[tail_candidates]

            scores = torch.norm(head_vec + rel_vec - tail_vecs, p=2, dim=1)
            topk = torch.topk(scores, k=min(num_hard_neg, len(scores)), largest=False)
            for idx in topk.indices.cpu().numpy():
                hard_negatives.append((h, r, tail_candidates[idx]))

        head_candidates = [i for i in range(num_entities) if i != h and (i, r, t) not in triples_set]
        if len(head_candidates) > max_candidates:
            head_candidates = random.sample(head_candidates, max_candidates)
        if head_candidates:
            tail_vec = entity_emb[t].unsqueeze(0)
            rel_vec = relation_emb[r].unsqueeze(0)
            head_vecs = entity_emb[head_candidates]

            scores = torch.norm(head_vecs + rel_vec - tail_vec, p=2, dim=1)
            topk = torch.topk(scores, k=min(num_hard_neg, len(scores)), largest=False)
            for idx in topk.indices.cpu().numpy():
                hard_negatives.append((head_candidates[idx], r, t))

    return hard_negatives

def train_transe(triples, num_entities, num_relations,
                 embedding_dim=50, learning_rate=0.01, epochs=10, batch_size=128, device='cpu'):
    model = TransE(num_entities, num_relations, embedding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MarginRankingLoss(margin=1.0)

    triples_set = set(tuple(triple) for triple in triples)
    dataset = TriplesDataset(triples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for pos_batch in dataloader:
            if not isinstance(pos_batch, torch.Tensor):
                pos_batch = torch.tensor(pos_batch, dtype=torch.long)
            pos_batch = pos_batch.long().to(device)

            head_pos = pos_batch[:, 0]
            rel_pos = pos_batch[:, 1]
            tail_pos = pos_batch[:, 2]

            pos_triples = [tuple(x) for x in pos_batch.cpu().tolist()]

            neg_batch = generate_hard_negative_samples_vectorized(
                model, pos_triples, num_entities, triples_set,
                num_hard_neg=1, max_candidates=20, device=device
            )
            if not neg_batch:
                continue

            neg_batch_tensor = torch.tensor(neg_batch, dtype=torch.long, device=device)
            head_neg = neg_batch_tensor[:, 0]
            rel_neg = neg_batch_tensor[:, 1]
            tail_neg = neg_batch_tensor[:, 2]

            optimizer.zero_grad()
            pos_scores = model(head_pos, rel_pos, tail_pos)
            neg_scores = model(head_neg, rel_neg, tail_neg)
            target = torch.ones_like(pos_scores)

            min_len = min(len(pos_scores), len(neg_scores))
            loss = loss_fn(neg_scores[:min_len], pos_scores[:min_len], target[:min_len])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    return model.entity_embeddings.weight.data.cpu().numpy(), model.relation_embeddings.weight.data.cpu().numpy(), model

def evaluate(model, triples, labels, batch_size=128, device='cpu'):
    model.eval()
    triples_tensor = torch.tensor(triples, dtype=torch.long, device=device)
    labels_tensor = torch.tensor(labels, dtype=torch.float, device=device)
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(0, len(triples), batch_size):
            batch = triples_tensor[i:i + batch_size]
            batch_labels = labels_tensor[i:i + batch_size]
            head = batch[:, 0]
            relation = batch[:, 1]
            tail = batch[:, 2]
            scores = model(head, relation, tail)
            preds = (torch.sigmoid(-scores) > 0.5).float()
            correct += (preds == batch_labels).sum().item()
            total += len(batch_labels)
    return correct / total

if __name__ == "__main__":
    import random
    from sklearn.model_selection import train_test_split
    from triplets import load_triples, load_labeled_triples, save_json_to_file

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    triples_raw, entities, relations = load_triples('dataset/relations_ru.txt')
    entity2id = {e: i for i, e in enumerate(entities)}
    relation2id = {r: i for i, r in enumerate(relations)}
    save_json_to_file('embeddings/entity2id.txt', entity2id)
    save_json_to_file('embeddings/relation2id.txt', relation2id)

    triples, labels = load_labeled_triples('dataset/relations_ru_train.tsv', entity2id, relation2id)
    num_entities = len(entities)
    num_relations = len(relations)

    train_triples, test_triples, train_labels, test_labels = train_test_split(
        triples, labels, test_size=0.2, random_state=42
    )

    entity_embeddings, relation_embeddings, model = train_transe(
        train_triples, num_entities, num_relations,
        embedding_dim=100, learning_rate=0.005, epochs=40, batch_size=128, device=device
    )

    test_acc = evaluate(model, test_triples, test_labels, device=device)
    print(f"Test Accuracy: {test_acc:.4f}")

    np.save('embeddings/relation_embeddings.npy', relation_embeddings)
    np.save('embeddings/entity_embeddings.npy', entity_embeddings)
