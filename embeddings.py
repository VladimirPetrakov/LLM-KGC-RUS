import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from sklearn.model_selection import train_test_split
from triplets import load_triples, load_labeled_triples, save_json_to_file

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

def generate_hard_negative_samples_batch(
    model, pos_batch, num_entities, triples_set,
    num_hard_neg=1, max_candidates=20, device='cpu'
):
    """
    Быстрая батчевая генерация hard negative samples для TransE.
    Для каждого триплета в батче генерирует по num_hard_neg hard negative,
    меняя head или tail, выбирает наиболее "трудные" негативы по скору модели.
    """
    model.eval()
    batch_size = pos_batch.size(0)
    pos_batch = pos_batch.to(device)

    hard_negatives = []

    for i in range(batch_size):
        h, r, t = pos_batch[i].tolist()

        candidates = []
        while len(candidates) < max_candidates:
            candidate = random.randint(0, num_entities - 1)
            if candidate != h and (candidate, r, t) not in triples_set:
                candidates.append(candidate)
        head_cands = torch.tensor(candidates, device=device)
        rel = torch.tensor([r]*max_candidates, device=device)
        tail = torch.tensor([t]*max_candidates, device=device)
        scores = model(head_cands, rel, tail)
        topk = torch.topk(scores, k=num_hard_neg, largest=False)
        for idx in topk.indices.cpu().numpy():
            hard_negatives.append((candidates[idx], r, t))

    for i in range(batch_size):
        h, r, t = pos_batch[i].tolist()
        candidates = []
        while len(candidates) < max_candidates:
            candidate = random.randint(0, num_entities - 1)
            if candidate != t and (h, r, candidate) not in triples_set:
                candidates.append(candidate)
        head = torch.tensor([h]*max_candidates, device=device)
        rel = torch.tensor([r]*max_candidates, device=device)
        tail_cands = torch.tensor(candidates, device=device)
        scores = model(head, rel, tail_cands)
        topk = torch.topk(scores, k=num_hard_neg, largest=False)
        for idx in topk.indices.cpu().numpy():
            hard_negatives.append((h, r, candidates[idx]))

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

            neg_batch = generate_hard_negative_samples_batch(
                model, pos_batch, num_entities, triples_set,
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
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    triples_raw, entities, relations = load_triples('dataset/relations_ru.txt')
    entity2id = {e: i for i, e in enumerate(entities)}
    relation2id = {r: i for i, r in enumerate(relations)}
    save_json_to_file('embeddings/entity2id.txt', entity2id)
    save_json_to_file('embeddings/relation2id.txt', relation2id)

    triples_train, labels = load_labeled_triples('dataset/relations_ru_train.tsv')

    triples = []
    for h, r, t in triples_train:
        if h in entity2id and r in relation2id and t in entity2id:
            triples.append((entity2id[h], relation2id[r], entity2id[t]))

    num_entities = len(entities)
    num_relations = len(relations)

    train_triples, test_triples, train_labels, test_labels = train_test_split(
        triples, labels, test_size=0.2, random_state=42
    )

    entity_embeddings, relation_embeddings, model = train_transe(
        train_triples, num_entities, num_relations,
        embedding_dim=200, learning_rate=0.001, epochs=300, batch_size=128, device=device
    )

    test_acc = evaluate(model, test_triples, test_labels, device=device)
    print(f"Test Accuracy: {test_acc:.4f}")

    np.save('embeddings/relation_embeddings.npy', relation_embeddings)
    np.save('embeddings/entity_embeddings.npy', entity_embeddings)
