import sys, os
import pandas as pd
from neo4j import GraphDatabase
import argparse

parser = argparse.ArgumentParser(description="Download data from neo4j database", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-kg", "--kg_graph", type=str, help="Which KG graph to download (Only KG1 and KG2C are available, default is 'kg2c')", default="kg2c")
parser.add_argument("-pcfg", "--path_RTXconfig", type=str, help="The path of RTXConfiguration.py", default="~/RTX/code/RTXConfiguration.py")
parser.add_argument("-o", "--output_folder", type=str, help="The path of output folder", default="~/explainable_DTD_model/mydata")
args = parser.parse_args()

sys.path.append(os.path.dirname(args.path_RTXconfig))
from RTXConfiguration import RTXConfiguration

output_path = args.output_folder

## Connect to neo4j database
if args.kg_graph.upper()=='KG1':
    rtxc = RTXConfiguration()
    driver = GraphDatabase.driver("bolt://arax.ncats.io:7687", auth=(rtxc.neo4j_username, rtxc.neo4j_password))
    session = driver.session()

    ## Pull a dataframe of all of the graph edges excluding
    # all edges directly connecting 'drug' and 'disease'
    query = "match (m1)<-[]-(m2) where m1<>m2 and not (m1.category='chemical_substance' and (m2.category='disease' or m2.category='phenotypic_feature')) and not (m2.category='chemical_substance' and (m1.category='disease' or m1.category='phenotypic_feature')) with distinct m1 as node1, m2 as node2 return node1.id as target, node2.id as source"
    res = session.run(query)
    KG_alledges = pd.DataFrame(res.data())
    KG_alledges.to_csv(output_path + '/graph_edges.txt', sep='\t', index=None)

    # Pulls a dataframe of all of the graph nodes with category label
    query = "match (n) with distinct n.id as id, n.name as name, n.category as category return id, name, category"
    res = session.run(query)
    KG_allnodes_label = pd.DataFrame(res.data())
    KG_allnodes_label = KG_allnodes_label.iloc[:, [0, 2]]
    KG_allnodes_label.to_csv(output_path + '/graph_nodes_label.txt', sep='\t', index=None)

    ## Pulls a dataframe of all of the graph drug-associated nodes
    query = f"match (n:chemical_substance) with distinct n.id as id, n.name as name return id, name"
    res = session.run(query)
    drugs = pd.DataFrame(res.data())
    drugs.to_csv(output_path + '/drugs.txt', sep='\t', index=None)

    ## Pulls a dataframe of all of the graph disease and phenotype nodes
    query = "match (n:phenotypic_feature) with distinct n.id as id, n.name as name return id, name union match (n:disease) with distinct n.id as id, n.name as name return id, name"
    res = session.run(query)
    diseases = pd.DataFrame(res.data())
    diseases.to_csv(output_path + '/diseases.txt', sep='\t', index=None)

elif args.kg_graph.upper()=='KG2C':
    rtxc = RTXConfiguration()
    driver = GraphDatabase.driver("bolt://kg2canonicalized2.rtx.ai:7687", auth=(rtxc.neo4j_username, rtxc.neo4j_password))
    session = driver.session()

    ## Pull a dataframe of all of the graph edges excluding
    # 1. the nodes which are both 'drug' and 'disease'
    # 2. the nodes with type including 'drug' and 'disease' != preferred_type
    # 3. all edges directly connecting 'drug' and 'disease'
    query = "match (n) where (((n.preferred_type<>'disease' and n.preferred_type<>'phenotypic_feature' and n.preferred_type<>'disease_or_phenotypic_feature') and ('disease' in n.types or 'phenotypic_feature' in n.types or 'disease_or_phenotypic_feature' in n.types)) or ((n.preferred_type<>'drug' and n.preferred_type<>'chemical_substance') and ('drug' in n.types or 'chemical_substance' in n.types))) or (('disease' in n.types or 'phenotypic_feature' in n.types or 'disease_or_phenotypic_feature' in n.types) and ('drug' in n.types or 'chemical_substance' in n.types)) with COLLECT(DISTINCT n.id) as exclude_id match (m1)<-[]-(m2) where m1<>m2 and not m1.id in exclude_id and not m2.id in exclude_id and not ((m1.preferred_type='disease' or m1.preferred_type='phenotypic_feature' or m1.preferred_type='disease_or_phenotypic_feature') and (m2.preferred_type='drug' or m2.preferred_type='chemical_substance')) and not ((m1.preferred_type='drug' or m1.preferred_type='chemical_substance') and (m2.preferred_type='disease' or m2.preferred_type='phenotypic_feature' or m2.preferred_type='disease_or_phenotypic_feature')) with distinct m1 as node1, m2 as node2 return node1.id as target, node2.id as source"
    res = session.run(query)
    KG_alledges = pd.DataFrame(res.data())
    KG_alledges.to_csv(output_path + '/graph_edges.txt', sep='\t', index=None)

    ## Pulls a dataframe of all of the graph nodes with category label
    query = "match (n) with distinct n.id as id, n.name as name, n.preferred_type as category return id, name, category"
    res = session.run(query)
    KG_allnodes_label = pd.DataFrame(res.data())
    KG_allnodes_label = KG_allnodes_label.iloc[:, [0, 2]]
    KG_allnodes_label.to_csv(output_path + '/graph_nodes_label.txt', sep='\t', index=None)

    ## Pulls a dataframe of all of the graph drug-associated nodes
    query = f"match (n:chemical_substance) with distinct n.id as id, n.name as name return id, name union match (n:drug) with distinct n.id as id, n.name as name return id, name"
    res = session.run(query)
    drugs = pd.DataFrame(res.data())
    drugs.to_csv(output_path + '/drugs.txt', sep='\t', index=None)

    ## Pulls a dataframe of all of the graph disease and phenotype nodes
    query = "match (n:phenotypic_feature) with distinct n.id as id, n.name as name return id, name union match (n:disease) with distinct n.id as id, n.name as name return id, name union match (n:disease_or_phenotypic_feature) with distinct n.id as id, n.name as name return id, name"
    res = session.run(query)
    diseases = pd.DataFrame(res.data())
    diseases.to_csv(output_path + '/diseases.txt', sep='\t', index=None)

else:
    print('Error: "--kg_graph" only allows kg1 or kg2c')
    exit(0)




