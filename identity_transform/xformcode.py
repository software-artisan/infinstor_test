i#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
import mlflow
import shutil
from infinstor import infin_


# In[5]:


df = infin_files_meta.list_files_meta('place_holder', '', input_name='in1')


# In[6]:


file_list = infin_files_meta.get_file_paths_local(df)


# In[7]:


outdir = "/tmp/output2/"
os.mkdir(outdir)
for f in file_list:
    shutil.copy(f, outdir)


# In[8]:


infin_files_meta.infin_log_artifacts(outdir, "output")
