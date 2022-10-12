import mlflow
import os
import infinstor_mlflow_plugin
import boto3
from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository
import tempfile
import pandas as pd
import pickle
import json
import sys
from concurrent_plugin import concurrent_core
from transformers import pipeline

print('sentiment_analysis: Entered', flush=True)
df = concurrent_core.list(None)
lp = concurrent_core.get_local_paths(df)
positives = 0
negatives = 0
for one_local_path in lp:
    print('Begin processing file ' + str(one_local_path), flush=True)
    try:
        with open(one_local_path, 'r') as f:
            jsn = json.load(f)
            print(json.dumps(jsn))
            for key, val in jsn.items():
                if key == 'positives':
                    positives = positives +  val
                elif key == 'negatives':
                    negatives = negatives +  val
    except Exception as ex:
        print('Caught ' + str(ex) + ' while processing ' + str(one_local_path), flush=True)

print('positives=' + str(positives) + ', negatives=' + str(negatives), flush=True)
fn = "/tmp/sentiment_summed.json"
if os.path.exists(fn):
    os.remove(fn)
with open(fn, 'w') as f:
    f.write(json.dumps({'positives': positives, 'negatives': negatives}))
concurrent_core.concurrent_log_artifact(fn, "")

os._exit(os.EX_OK)



df = concurrent_core.list(None, input_name='tweets')

print('Column Names:', flush=True)
cn = df.columns.values.tolist()
print(str(cn))

print('------------------------------ Obtained input info ----------------', flush=True)

for ind, row in df.iterrows():
    print("Input row=" + str(row), flush=True)

print('------------------------------ Finished dump of input info ----------------', flush=True)

lp = concurrent_core.get_local_paths(df)

print('Location paths=' + str(lp))

print('------------------------------ Begin Loading Huggingface Pipeline ------------------', flush=True)
nlp = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
print('------------------------------ After Loading Huggingface Pipeline ------------------', flush=True)

def do_nlp_fnx(row):
     s = nlp(row['text'])[0]
     return [s['label'], s['score']]

print('------------------------------ Before Inference ------------------', flush=True)
negatives = 0
positives = 0
for one_local_path in lp:
    print('Begin processing file ' + str(one_local_path), flush=True)
    jsonarray = pickle.load(open(one_local_path, 'rb'))
    # for i in jsonarray:
    #   print(json.dumps(i), flush=True)
    df1 = pd.DataFrame(jsonarray, columns=['text'])
    df1[['label', 'score']] = df1.apply(do_nlp_fnx, axis=1, result_type='expand')
    df1.reset_index()
    for index, row in df1.iterrows():
        # print("'" + row['text'] + "' sentiment=" + row['label'] + ", score=" + str(row['score']))
        if row['label'] == 'NEGATIVE' and row['score'] > 0.9:
            negatives = negatives + 1
        if row['label'] == 'POSITIVE' and row['score'] > 0.9:
            positives = positives + 1
    print('Finished processing file ' + str(one_local_path) + ': + ' + str(positives) + ', - ' + str(negatives), flush=True)
    # tf_fd, tfname = tempfile.mkstemp()
    # df1.to_pickle(tfname)
    # concurrent_core.concurrent_log_artifact(tfname, "result/" + os.path.basename(os.path.normpath(one_local_path)))
    # print('Finished logging artifacts file')

fn = '/tmp/sentiment_summary.json'
if os.path.exists(fn):
    os.remove(fn)
sentiment_summary = {'positives': positives, 'negatives': negatives}
with open(fn, 'w') as f:
    f.write(json.dumps(sentiment_summary))
concurrent_core.concurrent_log_artifact(fn, "")

print('------------------------------ After Inference. End ------------------', flush=True)

os._exit(os.EX_OK)

#print(str(sys.argv))

## tdir = tempfile.mkdtemp()
## print('model directory=' + str(tdir))
## ModelsArtifactRepository("models:/HFSentimentAnalysis/Production").download_artifacts(artifact_path="", dst_path=tdir)
## model = mlflow.pyfunc.load_model(tdir)
## print('model=' + str(model))

#inp = ['This is great weather', 'This is terrible weather']
#jsonarray = pickle.load(open('/home/jagane/Downloads/1565264790192365568', 'rb'))
#for i in jsonarray:
#  print(json.dumps(i))

#df = pd.DataFrame(jsonarray, columns=['text'])
#ii = model.predict(df)
#ii.reset_index()
#for index, row in ii.iterrows():
#  print("'" + row['text'] + "' sentiment=" + row['label'] + ", score=" + str(row['score']))
