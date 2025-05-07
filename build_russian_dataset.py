import re

def parse_label_file(label_file):
    label_dict = {}
    pattern = re.compile(
        r'<([^>]+)>\s+<[^>]+>\s+"((?:[^"\\]|\\.)*)"\s*@ru\b'
    )

    with open(label_file, encoding='utf-8') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                uri, label_escaped = match.groups()
                label = label_escaped.replace(r'\"', '"').replace(r"\'", "'")
                label_dict[uri] = label
            else:
                print(f"Не удалось распарсить label строку: {line.strip()}")
    return label_dict

def replace_uris_with_labels_no_brackets(relations_file, label_dict, output_file):
    with open(relations_file, encoding='utf-8') as fin, \
         open(output_file, 'w+', encoding='utf-8') as fout:
        for line in fin:
            uris = re.findall(r'(http://[^\s]+)', line)
            if not uris:
                print(f"URI не найдены в строке: {line.strip()}")
            new_line = line
            for uri in uris:
                label = label_dict.get(uri)
                if label:
                    new_line = new_line.replace(uri, f'"{label}"')
                else:
                    new_line = new_line.replace(uri, uri)
            fout.write(new_line)

if __name__ == "__main__":
    labels_file = 'dataset/labels_ru.txt'
    relations_file = 'dataset/relations_en.txt'
    output_file = 'dataset/relations_ru.txt'

    label_dict = parse_label_file(labels_file)
    print(f"Загружено {len(label_dict)} label'ов")
    replace_uris_with_labels_no_brackets(relations_file, label_dict, output_file)
