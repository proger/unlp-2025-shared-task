import llm
import json

model = llm.get_embedding_model("3-large")

file_path = "data/train.clean.jsonl"
output_file = "data/train.emb.jsonl"
data = {}

with open(file_path, "r") as file:
    for line in file:
        entry = json.loads(line)
        data[entry["id"]] = entry

with open(output_file, "r") as file:
    for line in file:
        entry = json.loads(line)
        data[entry["id"]] = entry

with open(output_file, "a") as file:
    for key in data:
        entry = data[key]
        emb_key = "embedding:3-large"
        if emb_key in entry:
            print("skip", key)
            continue
        content = entry["cleaned_content"]
        x = model.embed(content)
        entry[emb_key] = list(x)
        print("#", entry.get('id'))
        print()
        print(content)
        print()
        print(json.dumps(entry, ensure_ascii=False), file=file)

