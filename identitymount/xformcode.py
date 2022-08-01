#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
import mlflow
import shutil
from infinstor import infin_files_meta


# In[5]:


df = infin_files_meta.list_files_meta('isstage5-experiments', '', input_name='in1')


# In[ ]:


print(df.to_string())


# In[6]:


file_list = infin_files_meta.get_file_paths_local(df)
print(file_list)


# In[ ]:


df = df[df.apply(lambda row: '.infinstor' not in row['FileName'], axis=1)]
file_list = infin_files_meta.get_file_paths_local(df)
print(file_list)


# In[ ]:


print(df.to_string())


# In[7]:


outdir = "/tmp/output2/"
os.mkdir(outdir)
for f in file_list:
    shutil.copy(f, outdir)


# In[8]:


infin_files_meta.infin_log_artifacts(outdir, "output")


