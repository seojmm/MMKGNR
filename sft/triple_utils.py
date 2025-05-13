import json
from datasets import Dataset
import re
from typing import List, Tuple


# Load teacher triples from JSON
def load_teacher_triples(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    dataset_list = []
    for title, triples in data.items():
        response = '\n'.join([' | '.join(t) for t in triples])
        dataset_list.append({
            "prompt": title,
            "response": response,
            "labels_triples": triples
        })

    return Dataset.from_list(dataset_list)


# Triple Parsing
def extract_student_triples(text: str) -> List[Tuple[str, str, str]]:
    pattern = re.findall(r'\(?\s*([^|()\n]+?)\s*\|\s*([^|()\n]+?)\s*\|\s*([^|()\n]+?)\s*\)?', text)
    return [(s.strip(), p.strip(), o.strip()) for s, p, o in pattern]
