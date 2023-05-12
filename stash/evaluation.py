import os 
import sys 
import json

# Append the docprompting home directory so we can do the imports
sys.path.append(os.path.join(os.getcwd(), "../docprompting"))
from dataset_helper.conala.gen_metric import _bleu as conala_bleu

# If needed, pull the results from json file and save two files (y and y_pred)
# Where each line is a matching set of 
y_pred = "temp_files/test_pred.txt"
y_truth = "temp_files/test_target.txt"

save_file_name = "temp_files/pred_metrics.json"

metrics = tldr_metrics(y_truth, y_pred)

# save metrics json file
with open(save_file_name, "w") as f:
    json.dump(metrics, f, indent=2)
print(metrics)