import json

file_path = 'data/train.jsonl'
data = []

with open(file_path, 'r') as file:
    for line in file:
        data.append(json.loads(line))

# Define the function to process and print each message
def describe_message(entry):
    content = entry['content']
    techniques = entry.get('techniques', [])
    trigger_words = [
        content[start:end] for start, end in entry.get('trigger_words', []) or []
    ]
    print(f"Given message:\n{content}\n")

    if trigger_words:
        print("The above message contains manipulative phrases:")
        for word in trigger_words:
            word = word.strip()
            print(f"- {word}")

    if techniques:
        print(f"\nFound manipulation techniques: {', '.join(techniques)}")
    else:
        print("\nNo manipulation techniques have been found.")

    print("\n" + "-" * 80 + "\n")

# Loop over all messages and describe them
for message in data:
    describe_message(message)
