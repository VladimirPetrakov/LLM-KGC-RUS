import json

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

def load_labeled_triples(file_path, entity2id, relation2id):
    triples = []
    labels = []
    with open(file_path, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 4:
                continue
            h, r, t, label = parts
            if h in entity2id and r in relation2id and t in entity2id:
                triples.append((entity2id[h], relation2id[r], entity2id[t]))
                labels.append(int(label))
    return triples, labels

def save_json_to_file(file_path, jsonData):
    with open(file_path, 'w') as file:
        json.dump(jsonData, file)

def load_json_from_file(file_path):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return {}
