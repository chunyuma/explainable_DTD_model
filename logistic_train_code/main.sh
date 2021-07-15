## set working directory
work_folder=$(pwd)

# # ## generate G.json, id_map.json, class_map.json for running graphsage
# ## Note to know how to generate the input file (eg. graph_edges.txt and graph_nodes_label_remove_name.txt), please refer to step 0 (0_generate_graph_data_and_model_training_data.sh)
# time python ${work_folder}/scripts/graphsage_data_generation.py --graph ${work_folder}/data/graph_edges.txt --node_class ${work_folder}/data/graph_nodes_label.txt --feature_dim 1 --validation_percent 0.3 --output ${work_folder}/data/graphsage_input
# ## generate walks.txt for running graphsage (Note: based on the run on the machine with 1TB ram and 256 threads, this step with the setting below needs to run 3-4 days. Using too many threads might run out of memory)
# time python ${work_folder}/scripts/generate_random_walk.py --Gjson ${work_folder}/data/graphsage_input/data-G.json --walk_length 100 --number_of_walks 10 --batch_size 200000 --process 80 --output ${work_folder}/data/graphsage_input

# ## This bash script is to run graphsage

# ## set graphsage folder
# graphsage_folder=${work_folder}/graphsage

# ## set your working path
# work_path=${work_folder}

# ## set python path (Please use python 2.7 to run graphsage as graphsage was written by python2.7)
# ppath=~/miniconda3/envs/graphsage_p2.7env/bin/python

# ## set model name
# model='graphsage_mean'
# ## model option:
# #graphsage_mean -- GraphSage with mean-based aggregator
# #graphsage_seq -- GraphSage with LSTM-based aggregator
# #graphsage_maxpool -- GraphSage with max-pooling aggregator
# #graphsage_meanpool -- GraphSage with mean-pooling aggregator
# #gcn -- GraphSage with GCN-based aggregator
# #n2v -- an implementation of DeepWalk

# ## set data input folder and training data prefix
# train_prefix=${work_folder}/data/graphsage_input/data #note: here 'data' is the training data prefix

# ## other parameters
# model_size='big' #Can be big or small
# learning_rate=0.001 #test 0.01 and 0.001, 'initial learning rate'
# epochs=10 #test 5 and 10, 'number of epochs to train'
# samples_1=96 #suggest 15-25, based on the paper, bigger is better
# samples_2=96 #script only allows to set K=2, the same as samples_1
# dim_1=256 #Size of output dim (final is 2x this)
# dim_2=256
# max_total_steps=500 #Maximum total number of iterations
# validate_iter=5000 #how often to run a validation minibatch
# identity_dim=50 #Set to positive value to use identity embedding features of that dimension. Default 0
# batch_size=512 #minibatch size
# max_degree=96 #maximum node degree

# ## run graphsage unsupervised model
# $ppath -m graphsage.unsupervised_train --train_prefix ${train_prefix} --model_size ${model_size} --learning_rate ${learning_rate} --epochs ${epochs} --samples_1 ${samples_1} --samples_2 ${samples_2} --dim_1 ${dim_1} --dim_2 ${dim_2} --model ${model} --max_total_steps ${max_total_steps} --validate_iter ${validate_iter} --identity_dim ${identity_dim} --batch_size ${batch_size} --max_degree ${max_degree}


# ## Transform the GraphSage results to .emb format which is the node2vec output format
# python ${work_folder}/scripts/transform_format.py --input ${work_folder}/unsup-graphsage_input/graphsage_mean_big_0.001000 --output ${work_folder}/data/graphsage_node_vec.emb

# Train logistic model
echo 'running step3: train logistic model'
/usr/bin/time -v python ${work_folder}/scripts/run_logistic_model.py --data_path ${work_folder}/data --run_mode 1 --pair_emb 'concatenate' --output_folder ${work_folder}/results/
