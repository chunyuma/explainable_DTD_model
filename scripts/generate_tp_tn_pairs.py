import sys
import os
import pandas as pd
from neo4j import GraphDatabase
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--tp", type=str, nargs='*', help="The filenames or paths to the true positive txts", default=['semmed_tp.txt','mychem_tp.txt','mychem_tp_umls.txt','NDF_TP.txt'])
parser.add_argument("--tn", type=str, nargs='*', help="The filenames or paths to the true negative txts", default=['semmed_tn.txt','mychem_tn.txt','mychem_tn_umls.txt','NDF_TN.txt'])
parser.add_argument("--graph", type=str, help="The filename or path of graph edge file", default="graph_edges.txt")
parser.add_argument("--use_input_training_edges", action="store_true", help="Use the training edges from Mychem, SemMedDB and NDF", default=False)
parser.add_argument("--use_graph_edges", action="store_true", help="Use the existing edges in graph as training data", default=False)
parser.add_argument("--consider_pmids", action="store_true", help="Use the existing edges in graph and consider pmids if they have", default=False)
parser.add_argument("--tncutoff", type=int, help="A positive integer for the true negative cutoff of SemMedDB hot counts to include in analysis", default=2)
parser.add_argument("--tpcutoff", type=int, help="A positive integer for the true positive cutoff of SemMedDB hot counts to include in analysis", default=12)
parser.add_argument("--output", type=str, help="The path of output folder.", default="./")
args = parser.parse_args()

graph_edge = pd.read_csv(args.graph, sep='\t', header=0)
all_nodes = set()
all_nodes.update(set(graph_edge.source))
all_nodes.update(set(graph_edge.target))

id_list_class_dict = dict()
ambiguous_pairs = dict()
temp_dict = dict()

neo4j_bolt = os.getenv('neo4j_bolt')
neo4j_username = os.getenv('neo4j_username')
neo4j_password = os.getenv('neo4j_password')
                

## Connect to neo4j database
driver = GraphDatabase.driver(neo4j_bolt, auth=(neo4j_username, neo4j_password))
session = driver.session()


## add existing 'treat' edge between drug and disease node pair in canonicalized kg
query = f"match (disease) where (disease.category='biolink:Disease' or disease.category='biolink:PhenotypicFeature' or disease.category='biolink:DiseaseOrPhenotypicFeature') with collect(distinct disease.id) as disease_ids match (drug) where (drug.category='biolink:Drug' or drug.category='biolink:ChemicalSubstance' or drug.category='biolink:Metabolite') with collect(distinct drug.id) as drug_ids, disease_ids as disease_ids match (m1)<-[r]-(m2) where m1<>m2 and (m1.id in disease_ids and m2.id in drug_ids and (r.predicate='biolink:treats' or r.predicate='biolink:disrupts' or r.predicate='biolink:prevents')) with distinct m1 as node1, m2 as node2 return node2.id as source, node1.id as target"
res = session.run(query)
temp = pd.DataFrame(res.data())
for row in range(len(temp)):
    source_curie = temp['source'][row]
    target_curie = temp['target'][row]
    ## remove the curies not in the downloaded edges
    if not (source_curie in all_nodes and target_curie in all_nodes):
        continue
    
    if (source_curie, target_curie) not in temp_dict:
        temp_dict[source_curie, target_curie] = 1
    else:
        if temp_dict[source_curie, target_curie] != 1:
            ambiguous_pairs[source_curie, target_curie] = 'ambiguous'
            del temp_dict[source_curie, target_curie]

## add existing 'not treat' edge between drug and disease node pair in canonicalized kg
query = f"match (disease) where (disease.category='biolink:Disease' or disease.category='biolink:PhenotypicFeature' or disease.category='biolink:DiseaseOrPhenotypicFeature') with collect(distinct disease.id) as disease_ids match (drug) where (drug.category='biolink:Drug' or drug.category='biolink:ChemicalSubstance' or drug.category='biolink:Metabolite') with collect(distinct drug.id) as drug_ids, disease_ids as disease_ids match (m1)<-[r]-(m2) where m1<>m2 and (m1.id in disease_ids and m2.id in drug_ids and (r.predicate='biolink:causes' or r.predicate='biolink:predisposes' or r.predicate='biolink:contraindicated_for' or r.predicate='biolink:produces' or r.predicate='biolink:causes_adverse_event')) with distinct m1 as node1, m2 as node2 return node2.id as source, node1.id as target"
res = session.run(query)
temp = pd.DataFrame(res.data())
for row in range(len(temp)):
    source_curie = temp['source'][row]
    target_curie = temp['target'][row]
    ## remove the curies not in the downloaded edges
    if not (source_curie in all_nodes and target_curie in all_nodes):
        continue
    
    if (source_curie, target_curie) not in temp_dict:
        temp_dict[source_curie, target_curie] = 0
    else:
        if temp_dict[source_curie, target_curie] != 0:
            ambiguous_pairs[source_curie, target_curie] = 'ambiguous'
            del temp_dict[source_curie, target_curie]



if args.use_input_training_edges:

    TP_list = []
    TN_list = []

    # generate list of true positive and true negative data frames
    for i in range(len(args.tp)):
        temp = pd.read_csv(args.tp[i], sep="\t", index_col=None)
        temp = temp.drop_duplicates().reset_index().drop(columns=['index'])
        select_rows = list(all_nodes.intersection(set(temp['source'])))
        temp = temp.set_index('source').loc[select_rows,:].reset_index()
        select_rows = list(all_nodes.intersection(set(temp['target'])))
        temp = temp.set_index('target').loc[select_rows,:].reset_index()
        if temp.shape[0]!=0:
            TP_list += [temp]
    for i in range(len(args.tn)):
        temp = pd.read_csv(args.tn[i], sep="\t", index_col=None)
        temp = temp.drop_duplicates().reset_index().drop(columns=['index'])
        select_rows = list(all_nodes.intersection(set(temp['source'])))
        temp = temp.set_index('source').loc[select_rows,:].reset_index()
        select_rows = list(all_nodes.intersection(set(temp['target'])))
        temp = temp.set_index('target').loc[select_rows,:].reset_index()
        if temp.shape[0]!=0:
            TN_list += [temp]

    # Generate true negative training set by concatinating source-target pair vectors
    for TN in TN_list:
        for row in range(len(TN)):
            if 'count' in list(TN):
                if int(TN['count'][row]) < args.tncutoff:
                    continue

            source_curie = TN['source'][row]
            target_curie = TN['target'][row]
            ## remove the curies not in the downloaded edges
            if not (source_curie in all_nodes and target_curie in all_nodes):
                continue

            if (source_curie, target_curie) not in id_list_class_dict:
                if (source_curie, target_curie) not in ambiguous_pairs:
                    id_list_class_dict[source_curie, target_curie] = 0
            else:
                if id_list_class_dict[source_curie, target_curie] != 0:
                    if (source_curie, target_curie) not in ambiguous_pairs:
                        ambiguous_pairs[source_curie, target_curie] = 'ambiguous'
                    del id_list_class_dict[source_curie, target_curie]


    # Generate true positive training set by concatinating source-target pair vectors
    for TP in TP_list:
        for row in range(len(TP)):
            if 'count' in list(TP):
                if int(TP['count'][row]) < args.tpcutoff:
                    continue

            source_curie = TP['source'][row]
            target_curie = TP['target'][row]
            ## remove the curies not in the downloaded edges
            if not (source_curie in all_nodes and target_curie in all_nodes):
                continue
            
            if (source_curie, target_curie) not in id_list_class_dict:
                if (source_curie, target_curie) not in ambiguous_pairs:
                    id_list_class_dict[source_curie, target_curie] = 1
            else:
                if id_list_class_dict[source_curie, target_curie] != 1:
                    if (source_curie, target_curie) not in ambiguous_pairs:
                        ambiguous_pairs[source_curie, target_curie] = 'ambiguous'
                    del id_list_class_dict[source_curie, target_curie]

                    
if args.use_graph_edges:

    if args.consider_pmids:
    
        ## add existing 'treat' edge between drug and disease node pair in canonicalized kg (has pmids)
        query = f"match (disease) where (disease.category='biolink:Disease' or disease.category='biolink:PhenotypicFeature' or disease.category='biolink:DiseaseOrPhenotypicFeature') with collect(distinct disease.id) as disease_ids match (drug) where (drug.category='biolink:Drug' or drug.category='biolink:ChemicalSubstance' or drug.category='biolink:Metabolite') with collect(distinct drug.id) as drug_ids, disease_ids as disease_ids match (m1)<-[r]-(m2) where m1<>m2 and (m1.id in disease_ids and m2.id in drug_ids and (r.predicate='biolink:treats' or r.predicate='biolink:disrupts' or r.predicate='biolink:prevents') and exists(r.publications) and size(r.publications)>={args.tpcutoff}) with distinct m1 as node1, m2 as node2 return node2.id as source, node1.id as target"
        res = session.run(query)
        temp = pd.DataFrame(res.data())
        for row in range(len(temp)):
            source_curie = temp['source'][row]
            target_curie = temp['target'][row]
            ## remove the curies not in the downloaded edges
            if not (source_curie in all_nodes and target_curie in all_nodes):
                continue

            if (source_curie, target_curie) not in id_list_class_dict:
                if (source_curie, target_curie) not in ambiguous_pairs:
                    id_list_class_dict[source_curie, target_curie] = 1
            else:
                if id_list_class_dict[source_curie, target_curie] != 1:
                    if (source_curie, target_curie) not in ambiguous_pairs:
                        ambiguous_pairs[source_curie, target_curie] = 'ambiguous'
                    del id_list_class_dict[source_curie, target_curie]


        ## add existing 'treat' edge between drug and disease node pair in canonicalized kg (has no pmid)
        query = f"match (disease) where (disease.category='biolink:Disease' or disease.category='biolink:PhenotypicFeature' or disease.category='biolink:DiseaseOrPhenotypicFeature') with collect(distinct disease.id) as disease_ids match (drug) where (drug.category='biolink:Drug' or drug.category='biolink:ChemicalSubstance' or drug.category='biolink:Metabolite') with collect(distinct drug.id) as drug_ids, disease_ids as disease_ids match (m1)<-[r]-(m2) where m1<>m2 and (m1.id in disease_ids and m2.id in drug_ids and (r.predicate='biolink:treats' or r.predicate='biolink:disrupts' or r.predicate='biolink:prevents') and not exists(r.publications)) with distinct m1 as node1, m2 as node2 return node2.id as source, node1.id as target"
        res = session.run(query)
        temp = pd.DataFrame(res.data())
        for row in range(len(temp)):
            source_curie = temp['source'][row]
            target_curie = temp['target'][row]
            ## remove the curies not in the downloaded edges
            if not (source_curie in all_nodes and target_curie in all_nodes):
                continue
            
            if (source_curie, target_curie) not in id_list_class_dict:
                if (source_curie, target_curie) not in ambiguous_pairs:
                    id_list_class_dict[source_curie, target_curie] = 1
            else:
                if id_list_class_dict[source_curie, target_curie] != 1:
                    if (source_curie, target_curie) not in ambiguous_pairs:
                        ambiguous_pairs[source_curie, target_curie] = 'ambiguous'
                    del id_list_class_dict[source_curie, target_curie]
                    
                    
    else:
        
        ## add existing 'treat' edge between drug and disease node pair in canonicalized kg
        query = f"match (disease) where (disease.category='biolink:Disease' or disease.category='biolink:PhenotypicFeature' or disease.category='biolink:DiseaseOrPhenotypicFeature') with collect(distinct disease.id) as disease_ids match (drug) where (drug.category='biolink:Drug' or drug.category='biolink:ChemicalSubstance' or drug.category='biolink:Metabolite') with collect(distinct drug.id) as drug_ids, disease_ids as disease_ids match (m1)<-[r]-(m2) where m1<>m2 and (m1.id in disease_ids and m2.id in drug_ids and (r.predicate='biolink:treats' or r.predicate='biolink:disrupts' or r.predicate='biolink:prevents')) with distinct m1 as node1, m2 as node2 return node2.id as source, node1.id as target"
        res = session.run(query)
        temp = pd.DataFrame(res.data())
        for row in range(len(temp)):
            source_curie = temp['source'][row]
            target_curie = temp['target'][row]
            ## remove the curies not in the downloaded edges
            if not (source_curie in all_nodes and target_curie in all_nodes):
                continue
            
            # if (source_curie, target_curie) not in id_list_class_dict:
            #     id_list_class_dict[source_curie, target_curie] = 1
            # else:
            #     if id_list_class_dict[source_curie, target_curie] != 1:
            #         del id_list_class_dict[source_curie, target_curie]

            id_list_class_dict[source_curie, target_curie] = 1
                    
    if args.consider_pmids:
    
        ## add existing 'not treat' edge between drug and disease node pair in canonicalized kg (has pmids)
        query = f"match (disease) where (disease.category='biolink:Disease' or disease.category='biolink:PhenotypicFeature' or disease.category='biolink:DiseaseOrPhenotypicFeature') with collect(distinct disease.id) as disease_ids match (drug) where (drug.category='biolink:Drug' or drug.category='biolink:ChemicalSubstance' or drug.category='biolink:Metabolite') with collect(distinct drug.id) as drug_ids, disease_ids as disease_ids match (m1)<-[r]-(m2) where m1<>m2 and (m1.id in disease_ids and m2.id in drug_ids and (r.predicate='biolink:causes' or r.predicate='biolink:predisposes' or r.predicate='biolink:contraindicated_for' or r.predicate='biolink:produces' or r.predicate='biolink:causes_adverse_event') and exists(r.publications) and size(r.publications)>={args.tncutoff}) with distinct m1 as node1, m2 as node2 return node2.id as source, node1.id as target"
        res = session.run(query)
        temp = pd.DataFrame(res.data())
        for row in range(len(temp)):
            source_curie = temp['source'][row]
            target_curie = temp['target'][row]
            ## remove the curies not in the downloaded edges
            if not (source_curie in all_nodes and target_curie in all_nodes):
                continue

            if (source_curie, target_curie) not in id_list_class_dict:
                if (source_curie, target_curie) not in ambiguous_pairs:
                    id_list_class_dict[source_curie, target_curie] = 0
            else:
                if id_list_class_dict[source_curie, target_curie] != 0:
                    if (source_curie, target_curie) not in ambiguous_pairs:
                        ambiguous_pairs[source_curie, target_curie] = 'ambiguous'
                    del id_list_class_dict[source_curie, target_curie]

        ## add existing 'not treat' edge between drug and disease node pair in canonicalized kg (has no pmid)
        query = f"match (disease) where (disease.category='biolink:Disease' or disease.category='biolink:PhenotypicFeature' or disease.category='biolink:DiseaseOrPhenotypicFeature') with collect(distinct disease.id) as disease_ids match (drug) where (drug.category='biolink:Drug' or drug.category='biolink:ChemicalSubstance' or drug.category='biolink:Metabolite') with collect(distinct drug.id) as drug_ids, disease_ids as disease_ids match (m1)<-[r]-(m2) where m1<>m2 and (m1.id in disease_ids and m2.id in drug_ids and (r.predicate='biolink:causes' or r.predicate='biolink:predisposes' or r.predicate='biolink:contraindicated_for' or r.predicate='biolink:produces' or r.predicate='biolink:causes_adverse_event') and not exists(r.publications)) with distinct m1 as node1, m2 as node2 return node2.id as source, node1.id as target"
        res = session.run(query)
        temp = pd.DataFrame(res.data())
        for row in range(len(temp)):
            source_curie = temp['source'][row]
            target_curie = temp['target'][row]
            ## remove the curies not in the downloaded edges
            if not (source_curie in all_nodes and target_curie in all_nodes):
                continue

            if (source_curie, target_curie) not in id_list_class_dict:
                if (source_curie, target_curie) not in ambiguous_pairs:
                    id_list_class_dict[source_curie, target_curie] = 0
            else:
                if id_list_class_dict[source_curie, target_curie] != 0:
                    if (source_curie, target_curie) not in ambiguous_pairs:
                        ambiguous_pairs[source_curie, target_curie] = 'ambiguous'
                    del id_list_class_dict[source_curie, target_curie]

    else:

        ## add existing 'not treat' edge between drug and disease node pair in canonicalized kg
        query = f"match (disease) where (disease.category='biolink:Disease' or disease.category='biolink:PhenotypicFeature' or disease.category='biolink:DiseaseOrPhenotypicFeature') with collect(distinct disease.id) as disease_ids match (drug) where (drug.category='biolink:Drug' or drug.category='biolink:ChemicalSubstance' or drug.category='biolink:Metabolite') with collect(distinct drug.id) as drug_ids, disease_ids as disease_ids match (m1)<-[r]-(m2) where m1<>m2 and (m1.id in disease_ids and m2.id in drug_ids and (r.predicate='biolink:causes' or r.predicate='biolink:predisposes' or r.predicate='biolink:contraindicated_for' or r.predicate='biolink:produces' or r.predicate='biolink:causes_adverse_event')) with distinct m1 as node1, m2 as node2 return node2.id as source, node1.id as target"
        res = session.run(query)
        temp = pd.DataFrame(res.data())
        for row in range(len(temp)):
            source_curie = temp['source'][row]
            target_curie = temp['target'][row]
            ## remove the curies not in the downloaded edges
            if not (source_curie in all_nodes and target_curie in all_nodes):
                continue
            
            # if (source_curie, target_curie) not in id_list_class_dict:
            #     id_list_class_dict[source_curie, target_curie] = 0
            # else:
            #     if id_list_class_dict[source_curie, target_curie] != 0:
            #         del id_list_class_dict[source_curie, target_curie]
            id_list_class_dict[source_curie, target_curie] = 0

        
output_TP = []
output_TN = []
for key, value in id_list_class_dict.items():
    if value == 1:
        output_TP += [key]
    else:
        output_TN += [key]

pd.DataFrame(output_TP, columns=['source', 'target']).to_csv(args.output + "/tp_pairs.txt", sep='\t', index=None)
pd.DataFrame(output_TN, columns=['source', 'target']).to_csv(args.output + "/tn_pairs.txt", sep='\t', index=None)


####################### generate all potential known tp data for MRR and Hit@K evaluation #####################

# id_list_class_dict = dict()
# TP_list = []

# for i in range(len(args.tp)):
#     temp = pd.read_csv(args.tp[i], sep="\t", index_col=None)
#     temp = temp.drop_duplicates().reset_index().drop(columns=['index'])
#     select_rows = list(all_nodes.intersection(set(temp['source'])))
#     temp = temp.set_index('source').loc[select_rows,:].reset_index()
#     select_rows = list(all_nodes.intersection(set(temp['target'])))
#     temp = temp.set_index('target').loc[select_rows,:].reset_index()
#     if temp.shape[0]!=0:
#         TP_list += [temp]
        
# # Generate true positive training set by concatinating source-target pair vectors
# for TP in TP_list:
#     for row in range(len(TP)):
#         source_curie = TP['source'][row]
#         target_curie = TP['target'][row]
#         ## remove the curies not in the downloaded edges
#         if not (source_curie in all_nodes and target_curie in all_nodes):
#             continue
        
#         if (source_curie, target_curie) not in id_list_class_dict:
#             id_list_class_dict[source_curie, target_curie] = 1
#         else:
#             if id_list_class_dict[source_curie, target_curie] != 1:
#                 del id_list_class_dict[source_curie, target_curie]
                

# neo4j_bolt = os.getenv('neo4j_bolt')
# neo4j_username = os.getenv('neo4j_username')
# neo4j_password = os.getenv('neo4j_password')


# ## Connect to neo4j database
# driver = GraphDatabase.driver(neo4j_bolt, auth=(neo4j_username, neo4j_password))
# session = driver.session()

# ## add existing 'treat' edge between drug and disease node pair in canonicalized kg
# query = f"match (disease) where (disease.category='biolink:Disease' or disease.category='biolink:PhenotypicFeature' or disease.category='biolink:DiseaseOrPhenotypicFeature') with collect(distinct disease.id) as disease_ids match (drug) where (drug.category='biolink:Drug' or drug.category='biolink:ChemicalSubstance' or drug.category='biolink:Metabolite') with collect(distinct drug.id) as drug_ids, disease_ids as disease_ids match (m1)<-[r]-(m2) where m1<>m2 and (m1.id in disease_ids and m2.id in drug_ids and (r.predicate='biolink:treats' or r.predicate='biolink:disrupts' or r.predicate='biolink:prevents')) with distinct m1 as node1, m2 as node2 return node2.id as source, node1.id as target"
# res = session.run(query)
# temp = pd.DataFrame(res.data())
# for row in range(len(temp)):
#     source_curie = temp['source'][row]
#     target_curie = temp['target'][row]
#     ## remove the curies not in the downloaded edges
#     if not (source_curie in all_nodes and target_curie in all_nodes):
#         continue
    
#     if (source_curie, target_curie) not in id_list_class_dict:
#         id_list_class_dict[source_curie, target_curie] = 1
#     else:
#         if id_list_class_dict[source_curie, target_curie] != 1:
#             del id_list_class_dict[source_curie, target_curie]
            

# all_output_TP = [key for key, value in id_list_class_dict.items()]
# pd.DataFrame(all_output_TP, columns=['source', 'target']).to_csv(args.output + "/all_known_tps.txt", sep='\t', index=None)
