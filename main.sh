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
CUDA_VISIBLE_DEVICES=1 python scripts/run_GNN_model.py --run_model "gat" --num_samples [1000,1000] --data_path data --use_known_embedding --run_mode 1 --num_epochs 200 --learning_ratio 0.001 --use_gpu --num_head 1 --init_emb_size 100 --emb_size 128 --num_layers 2 --train_N 20 --non_train_N 500 --patience 10 --factor 0.1 --batch_size 100 --print_every 80 --train_val_test_size "[0.8, 0.1, 0.1]" --dropout_p 0 --output_folder results/  

model = GAT(100, 128, 3, num_layers=2, dropout_p=0, num_head = 1, use_gpu=True, use_multiple_gpu=False)
model.load_state_dict(torch.load("/home/zhihan/explainable_DTD_model/results/GAT_batchsize500_initemb100_embeddingsize128_layers2_numhead1_lr0.001_epoch00001_patience10_factor0.1_all_neighbors/GAT_batchsize500_initemb100_embeddingsize128_layers2_numhead1_lr0.001_epoch00001_patience10_factor0.1_all_neighbors.pt")['model_state_dict'])

RUN apt update -y &&  apt install -y build-essential ccache gfortran libssl-dev zlib1g-dev python3-dev libcurl4-openssl-dev libtbb-dev libboost-regex-dev libboost-program-options-dev libboost-system-dev libboost-filesystem-dev libboost-serialization-dev libboost-python-dev
RUN wget https://dl.google.com/go/go1.17.linux-amd64.tar.gz && rm -rf /usr/local/go && tar -C /usr/local -xzf go1.17.linux-amd64.tar.gz && export PATH=$PATH:/usr/local/go/bin && go version

RUN export PATH=/usr/local/cuda-11.0/bin/:$PATH && export CPATH=/usr/local/cuda-11.0/include:$CPATH && export LD_LIBRARY_PATH=/usr/local/cuda-11.0/lib64:$LD_LIBRARY_PATH && export DYLD_LIBRARY_PATH=/usr/local/cuda-11.0/lib:$DYLD_LIBRARY_PATH
RUN pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.1+cu110.html && pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.1+cu110.html && pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.7.1+cu110.html && pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.7.1+cu110.html && pip install torch-geometric -f https://pytorch-geometric.com/whl/torch-1.7.1+cu110.html

