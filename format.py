import re
import json
import sys

def describe_message(entry, print_labels=True):
    content = entry['content']
    techniques = entry.get('techniques', [])
    trigger_words = [
        content[start:end] for start, end in entry.get('trigger_words', []) or []
    ]
    print(f"Consider message enclosed in <message/> tag:\n<message>{content}</message>\n")

    if trigger_words:
        print("This message contains manipulative phrases:")
        for word in trigger_words:
            word = word.strip()
            print(f"- {word}")

    if print_labels:
        if techniques:
            print(f"\nFound manipulation techniques: {', '.join(techniques)}")
        else:
            print("\nNo manipulation techniques have been found.")

if sys.stdin.isatty():
    file_path = 'data/train.jsonl'
    file = open(file_path, 'r')
else:
    file = sys.stdin


# Loop over all messages and describe them
for message in file:
    message = json.loads(message)
    describe_message(message)
