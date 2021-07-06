## step1: download graph from neo4j database
## make sure if the version of neo4j-driver is >=1.7.0, <=1.7.6
echo 'running step1: download graph from neo4j database'
<<<<<<< HEAD
neo4j_bolt=bolt://localhost:7687  neo4j_username=neo4j neo4j_password=zhihan python  /home/zhihan/explainable_DTD_model/scripts/download_data_from_neo4j.py --output_folder /home/zhihan/explainable_DTD_model/data
=======
python  ~/work/explainable_DTD_model/scripts/download_data_from_neo4j.py --output_folder ~/work/explainable_DTD_model/data
>>>>>>> fdf2760b9ab749492decafecc37f2c597160dd65


## step2: generate training data
echo 'running step2: generate training data'
dataset_folder=~/work/explainable_DTD_model/data
python ~/work/explainable_DTD_model/scripts/generate_tp_tn_pairs.py --use_input_training_edges --tp ${dataset_folder}/training_data/semmed_tp.txt ${dataset_folder}/training_data/mychem_tp.txt ${dataset_folder}/training_data/ndf_tp.txt --tn ${dataset_folder}/training_data/semmed_tn.txt ${dataset_folder}/training_data/mychem_tn.txt ${dataset_folder}/training_data/ndf_tn.txt --graph ${dataset_folder}/graph_edges.txt --tncutoff "2" --tpcutoff "8" --output ${dataset_folder}

# # step3: train GNN model
echo 'running step3: train GNN model'
<<<<<<< HEAD
/usr/bin/time -v 

CUDA_VISIBLE_DEVICES=1 python /home/zhihan/explainable_DTD_model/scripts/run_GNN_model.py --data_path /home/zhihan/explainable_DTD_model/data --run_mode 1 --num_epochs 100 --learning_ratio 0.001 --use_gpu --num_head 1 --init_emb_size 100 --emb_size 100 --num_layers 2 --batch_size 500 --patience 10 --factor 0.1 --print_every 100 --train_val_test_size "[0.8, 0.1, 0.1]" --dropout_p 0 --output_folder /home/zhihan/explainable_DTD_model/results/ --use_known_embedding
=======
/usr/bin/time -v python ~/work/explainable_DTD_model/scripts/run_GNN_model.py --data_path ~/work/explainable_DTD_model/data --run_mode 1 --num_epochs 200 --learning_ratio 0.001 --use_gpu --num_head 1 --init_emb_size 100 --emb_size 128 --num_layers 2 --batch_size 500 --patience 10 --factor 0.1 --print_every 20 --train_val_test_size "[0.8, 0.1, 0.1]" --dropout_p 0 --output_folder ~/work/explainable_DTD_model/results/
>>>>>>> fdf2760b9ab749492decafecc37f2c597160dd65
