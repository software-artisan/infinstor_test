import mlflow
import os
import infinstor_mlflow_plugin
import boto3
from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository
import pandas as pd
import pickle
import json
import sys
from concurrent_plugin import concurrent_core
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification

print('sentiment_analysis: Entered', flush=True)
df = concurrent_core.list(None)

print('Column Names:', flush=True)
cn = df.columns.values.tolist()
print(str(cn))

print('------------------------------ Obtained input info ----------------', flush=True)

for ind, row in df.iterrows():
    print("Input row=" + str(row), flush=True)

print('------------------------------ Finished dump of input info ----------------', flush=True)

lp = concurrent_core.get_local_paths(df)

print('Location paths=' + str(lp))

print('------------------------------ Begin Loading Huggingface ner model ------------------', flush=True)
try:
    tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/roberta-large-ner-english")
    model = AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/roberta-large-ner-english")
except Exception as err:
    print('Caught ' + str(err) + ' while loading ner model')
print('------------------------------ After Loading Huggingface ner model ------------------', flush=True)

print('------------------------------ Begin Creating Huggingface ner pipeline ------------------', flush=True)
ner = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy="simple")
print('------------------------------ After Creating Huggingface ner pipeline ------------------', flush=True)

def do_ner_fnx(row):
    print("do_ner_fnx: Entered. row=" + str(row))
    s = ner(row['text'])
    orgs = []
    persons = []
    misc = []
    for entry in s:
        print("do_ner_fnx: Entry=" + str(entry))
        if entry['entity_group'] == 'ORG':
            orgs.append(entry['word'])
        elif entry['entity_group'] == 'PER':
            persons.append(entry['word'])
        elif entry['entity_group'] == 'MISC':
            misc.append(entry['word'])
    print("do_ner_fnx: Exit. returning orgs=" + str(orgs) + ", persons=" + str(persons) + ", misc=" + str(misc)) 
    return [{'orgs': orgs, 'persons': persons, 'misc': misc}]

print('------------------------------ Before Inference ------------------', flush=True)
consolidated_pd = pd.DataFrame()
for one_local_path in lp:
    print('Begin processing file ' + str(one_local_path), flush=True)
    df1 = pd.read_pickle(one_local_path)
    if consolidated_pd.empty:
        consolidated_pd = df1
    else:
        consolidated_pd = pd.DataFrame(df1)

consolidated_pd.reset_index()
consolidated_pd[['ner']] = consolidated_pd.apply(do_ner_fnx, axis=1, result_type='expand')
consolidated_pd.reset_index()
#for index, row in consolidated_pd.iterrows():
#    print("'" + row['text'] + "' sentiment=" + row['label'] + ", score=" + str(row['score']) + ", ner=" + str(row['ner']))

tfname = "/tmp/output.pickle"
if os.path.exists(tfname):
    os.remove(tfname)
consolidated_pd.to_pickle(tfname)
concurrent_core.concurrent_log_artifact(tfname, "")
print('Finished logging artifacts file')

print('------------------------------ After Inference. End ------------------', flush=True)

os._exit(os.EX_OK)
