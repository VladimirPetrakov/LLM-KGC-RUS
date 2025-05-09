import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from sklearn.model_selection import train_test_split
from triplets import load_triples, load_labeled_triples, save_json_to_file, triples_to_ids
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_curve

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

def train_transe(
    train_triples, train_labels,
    val_triples, val_labels,
    num_entities, num_relations,
    embedding_dim=200, learning_rate=0.001, epochs=300, batch_size=128, device='cpu'
):
    model = TransE(num_entities, num_relations, embedding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_dataset = TriplesDataset(train_triples)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    triples_set = set(map(tuple, train_triples))

    margin = 1
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, pos_batch in enumerate(train_loader):
            pos_batch = pos_batch.to(device)

            head_pos = pos_batch[:, 0]
            relation_pos = pos_batch[:, 1]
            tail_pos = pos_batch[:, 2]

            neg_batch = generate_hard_negative_samples_batch(
                model, pos_batch, num_entities, triples_set, device=device
            )

            neg_batch = torch.tensor(neg_batch, dtype=torch.long).to(device)
            head_neg = neg_batch[:, 0]
            relation_neg = neg_batch[:, 1]
            tail_neg = neg_batch[:, 2]

            optimizer.zero_grad()

            pos_scores = model(head_pos, relation_pos, tail_pos)
            neg_scores = model(head_neg, relation_neg, tail_neg)

            loss = torch.relu(pos_scores - neg_scores + margin).mean()

            running_loss += loss.item()

            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_acc = evaluate(model, val_triples, val_labels, device=device)
            train_acc = evaluate(model, train_triples, train_labels, device=device)

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {(running_loss / len(train_loader)):.4f},"
                  f" Validation Accuracy: {val_acc:.4f}, Train Accuracy: {train_acc:.4f}")

    entity_embeddings = model.entity_embeddings.weight.data.cpu().numpy()
    relation_embeddings = model.relation_embeddings.weight.data.cpu().numpy()

    return entity_embeddings, relation_embeddings, model

def evaluate(model, triples, labels, device='cpu'):
    model.eval()
    triples_tensor = torch.tensor(triples, dtype=torch.long, device=device)
    with torch.no_grad():
        head = triples_tensor[:, 0]
        relation = triples_tensor[:, 1]
        tail = triples_tensor[:, 2]
        scores = model(head, relation, tail).cpu().numpy()
    labels = np.array(labels)
    fpr, tpr, thresholds = roc_curve(labels, -scores)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    preds = (scores < -optimal_threshold).astype(int)
    acc = (preds == labels).mean()
    return acc

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

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    save_json_to_file('embeddings/entity2id.txt', entity2id)
    save_json_to_file('embeddings/relation2id.txt', relation2id)

    triples, labels = load_labeled_triples('dataset/relations_ru_train.tsv')
    train_triples, test_triples, train_labels, test_labels = train_test_split(
        triples, labels, test_size=0.2, random_state=42
    )
    train_triples, val_triples, train_labels, val_labels = train_test_split(
        train_triples, train_labels, test_size=0.2, random_state=42
    )

    train_triples_ids = triples_to_ids(train_triples, entity2id, relation2id)
    val_triples_ids = triples_to_ids(val_triples, entity2id, relation2id)
    test_triples_ids = triples_to_ids(test_triples, entity2id, relation2id)

    entity_embeddings, relation_embeddings, model = train_transe(
        train_triples_ids, train_labels,
        val_triples_ids, val_labels,
        num_entities=len(entity2id),
        num_relations=len(relation2id),
        embedding_dim=200,
        learning_rate=0.001,
        epochs=10,
        batch_size=128,
        device='cuda'
    )

    test_acc = evaluate(model, test_triples_ids, test_labels, device=device)
    print(f"Test Accuracy: {test_acc:.4f}")

    plot_score_distribution(model, val_triples_ids, val_labels, device=device)

    np.save('embeddings/relation_embeddings.npy', relation_embeddings)
    np.save('embeddings/entity_embeddings.npy', entity_embeddings)


triples_raw, entities, relations = load_triples('dataset/relations_ru.txt')
entity2id = {e: i for i, e in enumerate(entities)}
relation2id = {r: i for i, r in enumerate(relations)}

if __name__ == "__main__":
    main()