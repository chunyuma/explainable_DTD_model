## set working directory
work_folder=$(pwd)

## step1: download graph from neo4j database
echo 'running step1: download graph from neo4j database'
python  ${work_folder}/scripts/download_data_from_neo4j.py --output_folder ${work_folder}/data


## step2: generate training data
echo 'running step2: generate training data'
dataset_folder=${work_folder}/data
python ${work_folder}/scripts/generate_tp_tn_pairs.py --use_input_training_edges --tp ${dataset_folder}/training_data/semmed_tp.txt ${dataset_folder}/training_data/mychem_tp.txt ${dataset_folder}/training_data/ndf_tp.txt ${dataset_folder}/training_data/repoDB_tp.txt --tn ${dataset_folder}/training_data/semmed_tn.txt ${dataset_folder}/training_data/mychem_tn.txt ${dataset_folder}/training_data/ndf_tn.txt ${dataset_folder}/training_data/repoDB_tn.txt --graph ${dataset_folder}/graph_edges.txt --tncutoff "2" --tpcutoff "8" --output ${dataset_folder}

# # step3: train GNN model
echo 'running step3: train GNN model'
/usr/bin/time -v python ${work_folder}/scripts/run_GNN_model.py --run_model "GAT" --data_path ${work_folder}/data --use_known_embedding --run_mode 1 --num_epochs 600 --learning_ratio 0.001 --use_gpu --num_head 1 --init_emb_size 512 --emb_size 128 --num_layers 2 --batch_size 500 --mrr_hk_n 500 --patience 10 --factor 0.1 --print_every 20 --train_val_test_size "[0.8, 0.1, 0.1]" --dropout_p 0 --output_folder ${work_folder}/results/
CUDA_VISIBLE_DEVICES=1 python scripts/run_GNN_model.py --run_model "GAT" --data_path data --use_known_embedding --run_mode 1 --num_epochs 600 --learning_ratio 0.001 --use_gpu --num_head 1 --init_emb_size 100 --emb_size 128 --num_layers 2 --batch_size 500 --mrr_hk_n 500 --patience 10 --factor 0.1 --print_every 20 --train_val_test_size "[0.8, 0.1, 0.1]" --dropout_p 0 --output_folder results/
