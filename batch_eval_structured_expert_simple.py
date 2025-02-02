import sys
import time
import re
import json
import pandas as pd
import numpy as np
from openai import OpenAI
from pathlib import Path
client = OpenAI()

file_path = "data/train.jsonl"
target_label = (sys.argv[2:3] or ["loaded_language"])[0]
name_prefix = target_label + "_simple_"

if sys.argv[1:2] == ["mini"]:
    batch_file_path = f"exp/{name_prefix}expert_request_mini.jsonl"
    batch_result = f"exp/{name_prefix}expert_response_mini.jsonl"
    batch_id = f"exp/{name_prefix}expert_batch_id_mini"
    model_name = "gpt-4o-mini-2024-07-18"
elif sys.argv[1:2] == ["o1"]:
    batch_file_path = f"exp/{name_prefix}expert_request_o.jsonl"
    batch_result = f"exp/{name_prefix}expert_response_o.jsonl"
    batch_id = f"exp/{name_prefix}expert_batch_id_o"
    model_name = "o1-2024-12-17"
elif sys.argv[1:2] == ["o3-mini"]:
    #model_name = "o3‐mini‐2025‐01‐31"
    model_name = "o3‐mini"
    batch_file_path = f"exp/{name_prefix}expert_request_{model_name}.jsonl"
    batch_result = f"exp/{name_prefix}expert_response_{model_name}.jsonl"
    batch_id = f"exp/{name_prefix}expert_batch_id_{model_name}"
else:
    batch_file_path = f"exp/{name_prefix}expert_request.jsonl"
    batch_result = f"exp/{name_prefix}expert_response.jsonl"
    batch_id = f"exp/{name_prefix}expert_batch_id"
    model_name = "gpt-4o"

data = []
with open(file_path, 'r') as f:
    for line in f:
        data.append(json.loads(line))

df = pd.DataFrame(data)
df['techniques'] = df['techniques'].apply(lambda x: x if isinstance(x, list) else [])
all_labels = sorted({label for techniques in df['techniques'] for label in techniques})
all_labels_re = re.compile('|'.join(all_labels))

descriptions = {
    "appeal_to_fear": "Appeal to Fear: Uses fear to influence the audience, often exaggerating risks or consequences.",
    "bandwagon": "Bandwagon (Appeal to People): Encourages people to follow the crowd by suggesting everyone else is doing it.",
    "cherry_picking": "Cherry Picking: Selectively presents data or facts that support a viewpoint while ignoring contradictory evidence.",
    "cliche": "Thought-Terminating Cliché: Uses common phrases to shut down critical thinking and discussion.",
    "euphoria": "Euphoria: Creates a sense of triumph or happiness to boost morale and mobilize support.",
    "fud": "FUD (Fear, Uncertainty, Doubt): Sows fear, uncertainty, and doubt to manipulate perception and discourage opposition.",
    "glittering_generalities": "Glittering Generalities: Uses emotionally appealing but vague concepts like 'freedom' and 'justice' without specifics.",
    "loaded_language": "Loaded Language: Employs strongly emotional or charged words to sway opinion.",
    "straw_man": "Straw Man: Misrepresents an opponent’s argument to make it easier to attack.",
    "whataboutism": "Whataboutism: Deflects criticism by pointing out alleged hypocrisy instead of addressing the issue."
}

stage = 2 if Path(batch_id).exists() else 0
if stage < 1:
    with open(batch_file_path, 'w') as f:
        for i, row in df.iterrows():
            request = {
                "custom_id": row["id"],
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": f"You are an expert in the {target_label} manipulation technique. {descriptions[target_label]} List phrases that provide evidence of {target_label} in the given sentence if they are present. Given the evidence, make a judgement whether the sentence contains the {target_label} manipulation."},
                        {"role": "user", "content": row["content"]},
                    ],
                    "max_tokens": 1000,
                    "response_format": {            
                        "type": "json_schema",
                        "json_schema": {
                            "name": "techniques",
                            "strict": True,
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "evidence_phrases": {"type": "string"},
                                    target_label: {"type": "boolean"},
                                },
                                "required": ["evidence_phrases", target_label],
                                "additionalProperties": False,
                            },
                        },
                    },
                },
            }
            print(json.dumps(request, ensure_ascii=False), file=f)


if stage < 2:
    batch_input_file = client.files.create(
        file=open(batch_file_path, "rb"),
        purpose="batch"
    )

    batch = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": batch_file_path
        }
    )
    print(batch.id)
    Path(batch_id).write_text(batch.to_json())


if stage < 3:
    batch = json.loads(Path(batch_id).read_text())

    batch = client.batches.retrieve(batch['id'])
    output_file_id = batch.output_file_id
    while not output_file_id:
        print(batch.to_json())
        sys.exit(0) # try again later
        time.sleep(300)
        batch = client.batches.retrieve(batch.id)
        output_file_id = batch.output_file_id

    output_file = client.files.content(output_file_id)
    Path(batch_result).write_text(output_file.text)
    print(batch_result)

if stage < 4:
    fp = 0
    fn = 0
    tp = 0
    tn = 0
    count = 0

    for source, request, response in zip(data, Path(batch_file_path).open(), Path(batch_result).open()):
        content = json.loads(response)['response']['body']['choices'][0]['message']['content']
        try:
            content = json.loads(content)
        except:
            print(content)
            continue
        source_evidence = [source['content'][s:e] for (s, e), t in zip(source.get('trigger_words', []) or [], source['techniques'] or []) if t == target_label]
        print(content, source_evidence)
        reference = source['techniques'] and target_label in source['techniques']
        hypothesis = content[target_label]

        # manipulative if any of these match
        if reference and hypothesis:
            tp += 1
        elif reference and not hypothesis:
            fn += 1
        elif not reference and hypothesis:
            fp += 1
        else:
            tn += 1

        count += 1

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    f1 = 2*precision*recall/(precision+recall)
    print(f'{target_label} {precision=:0.2f} {recall=:0.2f} {accuracy=:0.2f} {f1=:0.2f} {tp=} {tn=} {fp=} {fn=} {count=}')
