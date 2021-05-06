import os
import argparse
import json
import time
import pickle

import fasttext
from transformers import AutoModel, AutoTokenizer

import torch
import numpy as np
from sklearn.decomposition import PCA


def get_bert_embedding(texts, tokenizer, model, device):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    inputs = inputs.to(device)
    with torch.no_grad():
        embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
    return embeddings.detach().to("cpu").numpy()


def load_data(args):
    id2index = {}
    texts = []

    with open(args.data_dir, "r") as f:
        line = f.readline()
        line = f.readline()
        index = 0
        while line:
            item = line.split("\t")
            if len(item) < 4:
                print(line)
                print(index)
            n_id = item[0]
            category = item[1]
            name = item[2]
            des = item[3].rstrip("\n") if item[3] != "\n" else " "
            text = category + " " + n_id + " " + name + " " + des
            
            texts.append(text)
            id2index[n_id] = index

            index += 1
            line = f.readline()
    
    return id2index, texts




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasttext", action="store_true", help="Whether use the Fasttext to get embedding or not", default=False)
    parser.add_argument("--bert", action="store_true", help="Whether use BERT to get embedding or not", default=False)
    parser.add_argument("--use_gpu", action="store_true", help="Whether use GPU or not", default=False)
    parser.add_argument("--fasttext_dir", type=str, help="Directory of the Fasttext model", default=False)
    parser.add_argument("--data_dir", type=str, help="Directory of the data", default=False)
    parser.add_argument("--batch_size", type=int, help="Batch size of bert embedding calculation", default=0)
    args = parser.parse_args()

    batch_size = args.batch_size if args.batch_size else 10

    assert (args.fasttext ^ args.bert), "Please choose one model from [fasttext, bert]" 

    print(f"Start Loading data from {args.data_dir}")
    id2index, texts = load_data(args)
    index2id = {value:key for key, value in id2index.items()}

    if not os.path.exists("data/text_embedding/"):
        os.makedirs("data/text_embedding/")

    

    if args.fasttext:
        print(f"Loading fasttext model from: {args.fasttext_dir}")
        model = fasttext.load_model(args.fasttext_dir)

        print(f"Calculating Fasttext embedding")
        start_time = time.time()

        id2embedding = {}
        for index, text in enumerate(texts):
            if index > 0 and (index % 100000) == 0:
                print(f"Finished: {index} in {time.time() - start_time}")
                start_time = time.time()
            embedding = model.get_sentence_vector(text) 
            n_id = index2id[index]
            id2embedding[n_id] = embedding
        
        print(f"Saving Fasttext embedding")
        with open("data/text_embedding/text_embedding_fasttext.pkl", 'wb') as f:
            pickle.dump(id2embedding, f, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        # writing files
        with open("data/text_embedding/id2index.json", "w") as f:
            json.dump(id2index, f)
        with open("data/text_embedding/index2id.json", "w") as f:
            json.dump(index2id, f)

        device = "cuda" if args.use_gpu else "cpu"

        tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-roberta-base")
        model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-roberta-base")
        model.to(device)

        ori_embedding = np.zeros([len(texts), 768])

        print(f"Calculating BERT embedding on {device} with batch size: {batch_size}")

        start_time = time.time()

        for i in range(len(texts) // batch_size):
            if (i * batch_size) % 10000 == 0:
                print(f"Finished: {i * batch_size} in {time.time() - start_time}")
                start_time = time.time()
            batch_text = texts[i*batch_size:(i+1)*batch_size]
            batch_embeddings = get_bert_embedding(batch_text, tokenizer, model, device)
            ori_embedding[i*batch_size:(i+1)*batch_size] = batch_embeddings
        
        if (i+1)*batch_size < len(texts):
            batch_text = texts[(i+1)*batch_size:]
            batch_embeddings = get_bert_embedding(batch_text, tokenizer, model, device)
            ori_embedding[(i+1)*batch_size:] = batch_embeddings
        
        np.save("data/text_embedding/bert_embdding.npy", ori_embedding)
        
        print("Fitting new embedding with PCA")

        pca = PCA(n_components=100)
        pca_embedding = pca.fit_transform(ori_embedding)

        print("Generating and saving data")

        id2embedding = {}
        for n_id in id2index.keys():
            id2embedding[n_id] = pca_embedding[id2index[n_id]]

        with open("data/text_embedding/text_embedding_bert.pkl", 'wb') as f:
            pickle.dump(id2embedding, f, protocol=pickle.HIGHEST_PROTOCOL)
        


    

