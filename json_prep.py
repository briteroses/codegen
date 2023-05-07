'''
Files to modify for intent:
conala_nl.txt
cmd_{split}.oracle_man.full.json (train / test / dev)
dev_retriever.json
fid.cmd_{split}.codet5.t10.json (train / test / dev)
train_retriever_sup_unsup.json (specifically only the supervised part)
'''

import json

# Load train, test and dev data
with open('data/conala/output_train.json', 'r') as f:
    train_data = [json.loads(line) for line in f]
    
with open('data/conala/output_test.json', 'r') as f:
    test_data = [json.loads(line) for line in f]

with open('data/conala/output_dev.json', 'r') as f:
    dev_data = [json.loads(line) for line in f]
    
split_types = ['train', 'test', 'dev']
data_list = [train_data, test_data, dev_data]

# Create a dictionary that maps question_id to gpt35_response
train_gpt35_response_dict = {entry['question_id']: entry['gpt35_response'] for entry in train_data}
test_gpt35_response_dict = {entry['question_id']: entry['gpt35_response'] for entry in test_data}
dev_gpt35_response_dict = {entry['question_id']: entry['gpt35_response'] for entry in dev_data}
dict_list = [train_gpt35_response_dict, test_gpt35_response_dict, dev_gpt35_response_dict]

combined_gpt35_response_dict = {**train_gpt35_response_dict, **test_gpt35_response_dict, **dev_gpt35_response_dict}

# 1. Process conala_nl.txt and conala_nl.id
with open('data/conala/conala_nl.txt', 'r') as nl_file, open('data/conala/conala_nl.id', 'r') as id_file, open('data/conala/conala_nl_modified.txt', 'w') as out_file:
    for nl_line, id_line in zip(nl_file, id_file):
        question_id = id_line.strip()
        gpt35_response = combined_gpt35_response_dict.get(question_id)
        if gpt35_response:
            out_file.write(gpt35_response + '\n')
        else:
            out_file.write(nl_line)

# 2. Process cmd_train.oracle_man.full.json
for split in range(3):
    split_type = split_types[split]
    with open(f'data/conala/cmd_{split_type}.oracle_man.full.json', 'r') as f:
        cmd_train_data = json.load(f)

    for entry in cmd_train_data:
        question_id = entry['question_id']
        gpt35_response = dict_list[split].get(question_id)
        if gpt35_response:
            entry['nl'] = gpt35_response

    with open(f'data/conala/cmd_{split_type}.oracle_man.full_modified.json', 'w') as f:
        json.dump(cmd_train_data, f, indent=2)

# 3. Process train_retriever_sup_unsup.json
with open('data/conala/train_retriever_sup_unsup.json', 'r') as f:
    train_retriever_data = json.load(f)

for i, entry in enumerate(train_retriever_data[573085:], start=573085):
    train_entry = train_data[i - 573085]
    if entry['text1'] == train_entry['nl']:
        entry['text1'] = train_entry['gpt35_response']

with open('data/conala/train_retriever_sup_unsup_modified.json', 'w') as f:
    json.dump(train_retriever_data, f, indent=2)

# 4. Process dev_retriever.json
with open('data/conala/dev_retriever.json', 'r') as f:
    dev_retriever_data = json.load(f)

current_text1 = None
dev_data_idx = 0

for entry in dev_retriever_data:
    if current_text1 != entry['text1']:
        current_text1 = entry['text1']
        if dev_data_idx < len(dev_data):
            gpt35_response = dev_data[dev_data_idx]['gpt35_response']
            dev_data_idx += 1
        else:
            gpt35_response = None

    if gpt35_response:
        entry['text1'] = gpt35_response

with open('data/conala/dev_retriever_modified.json', 'w') as f:
    json.dump(dev_retriever_data, f, indent=2)

# 5. Process fid.cmd_train.codet5.t10.json
for split in range(3):
    split_type = split_types[split]
    with open(f'data/conala/fid.cmd_{split_type}.codet5.t10.json', 'r') as f:
        fid_data = json.load(f)

    for fid_entry, orig_entry in zip(fid_data, data_list[split]):
        if fid_entry['id'] == orig_entry['question_id']:
            fid_entry['question'] = orig_entry['gpt35_response']

    with open(f'data/conala/fid.cmd_{split_type}.codet5.t10_modified.json', 'w') as f:
        json.dump(fid_data, f, indent=2)
