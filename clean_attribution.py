import llm
import json

# Initialize the model
model = llm.get_model("gpt-4o-mini")

# Load the JSONL dataset
file_path = "data/train.jsonl"
output_file = "data/train.clean.jsonl"
data = {}

with open(file_path, "r") as file:
    for line in file:
        entry = json.loads(line)
        data[entry["id"]] = entry

with open(output_file, "r") as file:
    for line in file:
        entry = json.loads(line)
        data[entry["id"]] = entry

# Function to clean a message using the model
def clean_message(content):
    prompt = (
        f"These are social media messages that contain footers with calls to action and channel attribution. Judiciously, keep only the content from this message. Discard emojis. Message: \n\n{content}"
    )
    response = model.prompt(prompt)
    return response.text().strip()

with open(output_file, "a") as file:
    for key in data:
        entry = data[key]
        if "cleaned_content" in entry:
            print("skip", key)
            continue
        content = entry["content"]
        try:
            cleaned_content = clean_message(content)
        except Exception as e:
            print("error", key, e)
            continue
        entry["cleaned_content"] = cleaned_content  # Add cleaned content to the entry
        print("#", entry.get('id'))
        print("### before")
        print(content)
        print("### after")
        print(cleaned_content)
        print()
        print()
        print(json.dumps(entry, ensure_ascii=False), file=file)

