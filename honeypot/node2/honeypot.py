import mlflow
import os
import infinstor_mlflow_plugin
import boto3
import tempfile
import pandas as pd
import json
import sys
from concurrent_plugin import concurrent_core
import pygeoip


print("Printing infin_boto3")
with open('/usr/local/lib/python3.7/site-packages/infinstor/infin_boto3.py') as infh:
    print(infh.read())
print("==============================================================================")

print('honeypot: Entered', flush=True)
df = concurrent_core.list(None)

print('Column Names:', flush=True)
cn = df.columns.values.tolist()
print(str(cn))

lp = concurrent_core.get_local_paths(df)
print('Local paths=' + str(lp))

summation = {}
for one_local_path in lp:
    print('Begin processing file ' + str(one_local_path), flush=True)
    try:
        with open(one_local_path, 'r') as f:
            jsn = json.load(f)
            print(json.dumps(jsn))
            for key, val in jsn:
                if key in summation:
                    summation['key'] = summation['key'] + val
                else:
                    summation['key'] = val
    except Exception as ex:
        print('Caught ' + str(ex) + ' while processing ' + str(one_local_path), flush=True)

print('summation=' + str(summation), flush=True)
fn = "/tmp/attack_by_country_summation.json"
if os.path.exists(fn):
    os.remove(fn)
with open(fn, 'w') as f:
    f.write(json.dumps(summation))
concurrent_core.concurrent_log_artifact(fn, "")

os._exit(os.EX_OK)
