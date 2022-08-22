#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
import mlflow
import shutil
from parallels_plugin import parallels_core


# In[5]:


df = parallels_core.list('isstage5-experiments', input_name='in1')


# In[ ]:


print(df.to_string())


# In[6]:


file_list = parallels_core.get_local_paths(df)
print(file_list)


# In[ ]:


df = df[df.apply(lambda row: '.infinstor' not in row['FileName'], axis=1)]


# In[ ]:


df['age'] = df.apply(lambda row: os.path.basename(os.path.dirname(row['FileName'])), axis=1)


# In[ ]:


age_list = df['age'].to_list()
file_list = parallels_core.get_local_paths(df)
print(file_list)


# In[ ]:


print(df.to_string())


# In[7]:


outdir = "/tmp/output2/"
os.mkdir(outdir)
for f, age in zip(file_list, age_list):
    print("Processing:", f, age)
    shutil.copy(f, outdir)
    if int(age) >= 60:
        heart_condition = "unhealthy"
    else:
        heart_condition = "healthy"
    print("heart_condition = ", heart_condition)
    parallels_core.parallels_log_artifact(f, "output", age=age, heart_condition=heart_condition)


# In[8]:


