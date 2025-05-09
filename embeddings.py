import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from sklearn.model_selection import train_test_split
from triplets import load_triples, load_labeled_triples, save_json_to_file, triples_to_ids
import matplotlib.pyplot as plt

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
    max_candidates=50, device='cpu'
):
    """
    Для каждого триплета в батче генерирует по одному hard negative,
    меняя head или tail случайно.
    """
    model.eval()
    batch_size = pos_batch.size(0)
    pos_batch = pos_batch.to(device)
    hard_negatives = []
    for i in range(batch_size):
        h, r, t = pos_batch[i].tolist()
        if random.random() < 0.5:
            candidates = []
            while len(candidates) < max_candidates:
                candidate = random.randint(0, num_entities - 1)
                if candidate != h and (candidate, r, t) not in triples_set:
                    candidates.append(candidate)
            head_cands = torch.tensor(candidates, device=device)
            rel = torch.tensor([r]*max_candidates, device=device)
            tail = torch.tensor([t]*max_candidates, device=device)
            scores = model(head_cands, rel, tail)
            topk = torch.topk(scores, k=1, largest=False)
            idx = topk.indices[0].item()
            hard_negatives.append((candidates[idx], r, t))
        else:
            candidates = []
            while len(candidates) < max_candidates:
                candidate = random.randint(0, num_entities - 1)
                if candidate != t and (h, r, candidate) not in triples_set:
                    candidates.append(candidate)
            head = torch.tensor([h]*max_candidates, device=device)
            rel = torch.tensor([r]*max_candidates, device=device)
            tail_cands = torch.tensor(candidates, device=device)
            scores = model(head, rel, tail_cands)
            topk = torch.topk(scores, k=1, largest=False)
            idx = topk.indices[0].item()
            hard_negatives.append((h, r, candidates[idx]))
    return hard_negatives

def train_transe(triples, num_entities, num_relations,
                 embedding_dim=50, learning_rate=0.01, epochs=10, batch_size=128, device='cpu'):
    model = TransE(num_entities, num_relations, embedding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    loss_fn = nn.SoftMarginLoss()

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
                max_candidates=50, device=device
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

            loss = loss_fn(pos_scores - neg_scores, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
        val_acc = evaluate(model, val_triples, val_labels, device=device)
        print(f"Validation Accuracy: {val_acc:.4f}")

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
def plot_score_distribution(model, triples, labels, entity2id=None, relation2id=None, device='cpu'):
    """
    Если triples - список строковых троек, entity2id и relation2id - словари для перевода.
    Если triples - список кортежей чисел, entity2id и relation2id можно не передавать.
    """
    model.eval()
    if entity2id is not None and relation2id is not None and len(triples) > 0 and isinstance(triples[0][0], str):
        triples_ids = []
        filtered_labels = []
        for (h, r, t), label in zip(triples, labels):
            if h in entity2id and r in relation2id and t in entity2id:
                triples_ids.append((entity2id[h], relation2id[r], entity2id[t]))
                filtered_labels.append(label)
        triples_tensor = torch.tensor(triples_ids, dtype=torch.long, device=device)
        labels_array = np.array(filtered_labels)
    else:
        triples_tensor = torch.tensor(triples, dtype=torch.long, device=device)
        labels_array = np.array(labels)

    scores = []
    batch_size = 128
    with torch.no_grad():
        for i in range(0, len(triples_tensor), batch_size):
            batch = triples_tensor[i:i+batch_size]
            head = batch[:, 0]
            relation = batch[:, 1]
            tail = batch[:, 2]
            batch_scores = model(head, relation, tail)
            scores.extend(batch_scores.cpu().numpy())

    scores = np.array(scores)

    plt.figure(figsize=(8,6))
    plt.hist(scores[labels_array == 1], bins=50, alpha=0.6, label='Позитивные')
    plt.hist(scores[labels_array == 0], bins=50, alpha=0.6, label='Негативные')
    plt.xlabel('Score (расстояние)')
    plt.ylabel('Количество')
    plt.title('Распределение скорингов модели TransE')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    triples_raw, entities, relations = load_triples('dataset/relations_ru.txt')
    entity2id = {e: i for i, e in enumerate(entities)}
    relation2id = {r: i for i, r in enumerate(relations)}
    save_json_to_file('embeddings/entity2id.txt', entity2id)
    save_json_to_file('embeddings/relation2id.txt', relation2id)

    triples, labels = load_labeled_triples('dataset/relations_ru_train.tsv')

    triples_ids = triples_to_ids(triples, entity2id, relation2id)

    train_triples, test_triples, train_labels, test_labels = train_test_split(
        triples_ids, labels, test_size=0.2, random_state=42
    )
    train_triples, val_triples, train_labels, val_labels = train_test_split(
        train_triples, train_labels, test_size=0.2, random_state=42
    )

    num_entities = len(entities)
    num_relations = len(relations)

    entity_embeddings, relation_embeddings, model = train_transe(
        train_triples, num_entities, num_relations,
        embedding_dim=300, learning_rate=0.0005, epochs=10, batch_size=128, device=device
    )

    plot_score_distribution(model, val_triples, val_labels, device=device)

    np.save('embeddings/relation_embeddings.npy', relation_embeddings)
    np.save('embeddings/entity_embeddings.npy', entity_embeddings)
