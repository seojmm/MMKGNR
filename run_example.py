from triple_extractor.extractor import extract_batch_triples

if __name__ == "__main__":
    results = extract_batch_triples(
        model="gpt-4o-mini-2024-07-18",
        jsonl_path="behaviors_test.jsonl",
        save_output=True,
    )
