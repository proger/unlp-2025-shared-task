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
if sys.argv[1:2] == ["mini"]:
    batch_file_path = "exp/structured_request_mini.jsonl"
    batch_result = "exp/structured_response_mini.jsonl"
    batch_id = "exp/structured_batch_id_mini"
    model_name = "gpt-4o-mini-2024-07-18"
else:
    batch_file_path = "exp/structured_request.jsonl"
    batch_result = "exp/structured_response.jsonl"
    batch_id = "exp/structured_batch_id"
    model_name = "gpt-4o"

data = []
with open(file_path, 'r') as f:
    for line in f:
        data.append(json.loads(line))

df = pd.DataFrame(data)
df['techniques'] = df['techniques'].apply(lambda x: x if isinstance(x, list) else [])
all_labels = sorted({label for techniques in df['techniques'] for label in techniques})
all_labels_re = re.compile('|'.join(all_labels))

stage = 2 if Path(batch_id).exists() else 0
if stage < 1:
    with open(batch_file_path, 'w') as f:
        for i, row in df.iterrows():
            user_prompt = f"The following text may use manipulative techniques. Your task is to identify the techniques used, if any, from this list: {', '.join(all_labels)}. Read the message and flag the presence of each technique.\nMessage:\n{row['content']}\n"
            request = {
                "custom_id": row["id"],
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": user_prompt},
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
                                    "reasoning": {"type": "string"},
                                } | {label: {"type": "boolean"} for label in all_labels},
                                "required": ["reasoning"] + all_labels,
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
            "description": "nightly eval job"
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
        time.sleep(300)
        batch = client.batches.retrieve(batch.id)
        output_file_id = batch.output_file_id

    output_file = client.files.content(output_file_id)
    Path(batch_result).write_text(output_file.text)
    print(batch_result)

if stage < 4:
    tp = {label: 0 for label in all_labels}
    tn = {label: 0 for label in all_labels}
    fp = {label: 0 for label in all_labels}
    fn = {label: 0 for label in all_labels}

    manip_fp = 0
    manip_fn = 0
    manip_tp = 0
    manip_tn = 0

    for source, request, response in zip(data, Path(batch_file_path).open(), Path(batch_result).open()):
        content = json.loads(response)['response']['body']['choices'][0]['message']['content']
        content = json.loads(content)
        reference = {label: True for label in source['techniques'] or []}
        hypothesis = {label: content[label] for label in all_labels if content[label]}

        remaining_labels = list(all_labels)
        for label in reference:
            if label in hypothesis:
                tp[label] += 1
                remaining_labels.remove(label)
            else:
                fn[label] += 1
                remaining_labels.remove(label)

        for label in hypothesis:
            if label not in reference:
                fp[label] += 1
                remaining_labels.remove(label)

        for label in remaining_labels:
            tn[label] += 1

        # manipulative if any of these match
        if reference and hypothesis:
            manip_tp += 1
        elif reference and not hypothesis:
            manip_fn += 1
        elif not reference and hypothesis:
            manip_fp += 1
        else:
            manip_tn += 1

    f1 = [] 
    for label in all_labels:
        precision = tp[label] / (tp[label] + fp[label])
        recall = tp[label] / (tp[label] + fn[label])
        accuracy = (tp[label] + tn[label]) / (tp[label] + tn[label] + fp[label] + fn[label])
        print(f'{label=} {precision=:0.2f} {recall=:0.2f} {accuracy=:0.2f}')
        f1.append(2*precision*recall/(precision+recall))
    f1 = np.mean(f1)
    print(f'macro {f1=:.2f} over {all_labels}')

    precision = manip_tp / (manip_tp + manip_fp)
    recall = manip_tp / (manip_tp + manip_fn)
    accuracy = (manip_tp + manip_tn) / (manip_tp + manip_tn + manip_fp + manip_fn)
    f1 = 2*precision*recall/(precision+recall)
    print(f'any {precision=:0.2f} {recall=:0.2f} {accuracy=:0.2f} {f1=:0.2f}')
