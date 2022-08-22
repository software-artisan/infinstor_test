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


file_list = parallels_core.get_local_paths(df)
print(file_list)


# In[ ]:


print(df.to_string())


# In[7]:


outdir = "/tmp/output2/"
os.mkdir(outdir)
for f in file_list:
    shutil.copy(f, outdir)


# In[8]:


parallels_core.parallels_log_artifacts(outdir, "output")


