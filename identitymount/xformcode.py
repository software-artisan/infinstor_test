#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
import mlflow
import shutil
from parallels_plugin import parallels_files_meta


# In[5]:


df = parallels_files_meta.list_files_meta('isstage5-experiments', '', input_name='in1')


# In[ ]:


print(df.to_string())


# In[6]:


file_list = parallels_files_meta.get_file_paths_local(df)
print(file_list)


# In[ ]:


file_list = parallels_files_meta.get_file_paths_local(df)
print(file_list)


# In[ ]:


print(df.to_string())


# In[7]:


outdir = "/tmp/output2/"
os.mkdir(outdir)
for f in file_list:
    shutil.copy(f, outdir)


# In[8]:


parallels_files_meta.infin_log_artifacts(outdir, "output")


