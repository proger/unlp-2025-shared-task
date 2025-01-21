import re
import json
import pandas as pd
import numpy as np
from openai import OpenAI
from pathlib import Path
client = OpenAI()

file_path = "data/train.jsonl"  # Path to your JSONL file
batch_file_path = "exp/batch_request.jsonl"
batch_result = "exp/batch_response.jsonl"
batch_id = "exp/batch_id"

data = []
with open(file_path, 'r') as f:
    for line in f:
        data.append(json.loads(line))

df = pd.DataFrame(data)
df['techniques'] = df['techniques'].apply(lambda x: x if isinstance(x, list) else [])
all_labels = sorted({label for techniques in df['techniques'] for label in techniques})
all_labels_re = re.compile('|'.join(all_labels))
negative_labels = ["does not", "doesn't", "don't"]
negative_labels_re = re.compile('|'.join(negative_labels))

stage = 3
if stage < 1:
    with open(batch_file_path, 'w') as f:
        for i, row in df.iterrows():
            user_prompt = f"The following text may use manipulative techniques. Your task is to identify the techniques used, if any, from this list: {', '.join(all_labels)}. Read the message and return the techniques as a comma-separated list.\nMessage:\n{row['content']}\n"
            request = {
                "custom_id": f"request-{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": user_prompt}
                    ],
                    "max_tokens": 1000
                }
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
    if output_file_id := batch.output_file_id:
        output_file = client.files.content(output_file_id)
        Path(batch_result).write_text(output_file.text)
        print(batch_result)
    else:
        print(batch.to_json())

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
        hypothesis = list(set(all_labels_re.findall(content)))
        source['response:4o.v1'] = hypothesis
        negative = negative_labels_re.findall(content)
        #if len(all_labels) == len(hypothesis) and negative:
        #    hypothesis = []
        if negative:
            hypothesis = []
        # pathologies:
        # does not use any of (lists some). none.
        # "euphoria" ... "does not employ any other"
        source['response:4o.v1'] = hypothesis
        reference = source['techniques'] or []
        if reference != hypothesis and len(content) > len(', '.join(hypothesis)):
            print(source['techniques'], source['response:4o.v1'], content)

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
    print(f'label-average {f1=:.2f} over {all_labels}')

    precision = manip_tp / (manip_tp + manip_fp)
    recall = manip_tp / (manip_tp + manip_fn)
    accuracy = (manip_tp + manip_tn) / (manip_tp + manip_tn + manip_fp + manip_fn)
    f1 = 2*precision*recall/(precision+recall)
    print(f'total manipulative {precision=:0.2f} {recall=:0.2f} {accuracy=:0.2f} {f1=:0.2f}')
