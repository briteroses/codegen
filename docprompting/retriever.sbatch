#!/bin/bash
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=3:00:00
#SBATCH --partition=urop
#SBATCH --output=z.out
#SBATCH --error=z.err
#SBATCH --exclude=deep-chungus-[1-6,11]

python retriever/simcse/run_train.py     --num_layers 12     --model_name_or_path Salesforce/codet5-base     --sim_func cls_distance.cosine     --temp 0.05      --train_file data/conala-modified/train_retriever_sup_unsup_modified.json     --eval_file data/conala-modified/dev_retriever_modified.json     --output_dir models/retriever/docprompting_enhanced_python_doc_retriever     --eval_src_file data/conala-modified/conala_nl_modified.txt     --eval_tgt_file data/conala/python_manual_firstpara.tok.txt     --eval_root_folder data/conala-modified     --eval_oracle_file data/conala-modified/cmd_dev.oracle_man.full_modified.json     --run_name docprompting_enhanced_python_doc_retriever     --num_train_epochs 10     --per_device_train_batch_size 512     --learning_rate 1e-5     --max_seq_length 32     --evaluation_strategy steps     --metric_for_best_model recall@10     --load_best_model_at_end     --eval_steps 125     --overwrite_output_dir     --do_train     --eval_form retrieval