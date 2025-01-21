import io
import llm
import json

model = llm.get_model("gpt-4o")

file_path = "data/train.clean.jsonl"
output_file = "data/train.explain.jsonl"
data = {}

with open(file_path, "r") as file:
    for line in file:
        entry = json.loads(line)
        data[entry["id"]] = entry

with open(output_file, "r") as file:
    for line in file:
        entry = json.loads(line)
        data[entry["id"]] = entry

def describe_message(entry, print_labels=True):
    buf = io.StringIO()
    content = entry['content']
    techniques = entry.get('techniques', [])
    trigger_words = [
        content[start:end] for start, end in entry.get('trigger_words', []) or []
    ]
    content = entry['cleaned_content']
    print(f"Consider message enclosed in <message/> tag:\n<message>{content}</message>\n", file=buf)

    if trigger_words:
        print("This message contains manipulative phrases:", file=buf)
        for word in trigger_words:
            word = word.strip()
            print(f"- {word}", file=buf)

    if print_labels:
        if techniques:
            print(f"\nFound manipulation techniques: {', '.join(techniques)}", file=buf)
        else:
            print("\nNo manipulation techniques have been found.", file=buf)
    return buf.getvalue()

def explain_entry(entry):
    prompt = (
        describe_message(entry) + "\nFor each snippet justify its manipulation technique by choosing from the provided list and deliberating."
    )
    response = model.prompt(prompt)
    return response.text().strip()

with open(output_file, "a") as file:
    for key in data:
        entry = data[key]
        try:
            explanation = explain_entry(entry)
        except Exception as e:
            print("error", key, e)
            continue
        entry["explanation"] = explanation
        print("#", entry.get('id'))
        print("### before")
        print(entry["cleaned_content"])
        print("### after")
        print(explanation)
        print()
        print()
        print(json.dumps(entry, ensure_ascii=False), file=file)

