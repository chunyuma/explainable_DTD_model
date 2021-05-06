import sys, os, re
import pandas as pd
from neo4j import GraphDatabase
import argparse

parser = argparse.ArgumentParser(description="Download data from neo4j database", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-o", "--output_folder", type=str, help="The path of output folder", default="~/explainable_DTD_model/mydata")
args = parser.parse_args()

neo4j_bolt = os.getenv('neo4j_bolt')
neo4j_username = os.getenv('neo4j_username')
neo4j_password = os.getenv('neo4j_password')

output_path = args.output_folder

## Connect to neo4j database
driver = GraphDatabase.driver(neo4j_bolt, auth=(neo4j_username, neo4j_password))
session = driver.session()

## Pull a dataframe of all of the graph edges
# query = "match (disease) where (disease.category='biolink:Disease' or disease.category='biolink:PhenotypicFeature' or disease.category='biolink:DiseaseOrPhenotypicFeature') with collect(distinct disease.id) as disease_ids match (drug) where (drug.category='biolink:Drug' or drug.category='biolink:ChemicalSubstance') with collect(distinct drug.id) as drug_ids, disease_ids as disease_ids match (m1)<-[]-(m2) where m1<>m2 and not (m1.id in drug_ids and m2.id in disease_ids) and not (m1.id in disease_ids and m2.id in drug_ids) with distinct m1 as node1, m2 as node2 return node1.id as target, node2.id as source"

# res = session.run(query)
# KG_alledges = pd.DataFrame(res.data())
# KG_alledges.to_csv(output_path + '/graph_edges.txt', sep='\t', index=None)

## Pulls a dataframe of all of the graph nodes with category label
query = "match (n) with distinct n.id as id, n.category as category, n.name as name, n.description as des return id, category, name, des"
res = session.run(query)
KG_allnodes_label = pd.DataFrame(res.data())
print(f"Total number of nodes: {len(KG_allnodes_label)}")
for i in range(len(KG_allnodes_label)):
    if KG_allnodes_label.loc[i, "des"]:
        KG_allnodes_label.loc[i, "des"] = " ".join(KG_allnodes_label.loc[i, "des"].replace("\n", " ").split())

KG_allnodes_label.to_csv(output_path + '/graph_nodes_label.txt', sep='\t', index=None)

## Pulls a dataframe of all of the graph drug-associated nodes
# query = f"match (n) where (n.category='biolink:Drug') or (n.category='biolink:ChemicalSubstance') with distinct n.id as id, n.name as name return id, name"
# res = session.run(query)
# drugs = pd.DataFrame(res.data())
# drugs.to_csv(output_path + '/drugs.txt', sep='\t', index=None)

## Pulls a dataframe of all of the graph disease and phenotype nodes
# query = "match (n) where (n.category='biolink:PhenotypicFeature') or (n.category='biolink:Disease') or (n.category='biolink:DiseaseOrPhenotypicFeature') with distinct n.id as id, n.name as name return id, name"
# res = session.run(query)
# diseases = pd.DataFrame(res.data())
# diseases.to_csv(output_path + '/diseases.txt', sep='\t', index=None)




