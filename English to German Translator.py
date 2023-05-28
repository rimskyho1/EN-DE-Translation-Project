#!/usr/bin/env python
# coding: utf-8

# In[1]:


import glob
import os

#Find the latest checkpoint folder created during PyTorch's model training process 
list_of_files = glob.glob(r'EN-DE PyTorch Translator\*')
latest_file = max(list_of_files, key=os.path.getctime)
print(latest_file)


# In[2]:


from transformers import pipeline

#Import the model and assign it to the translation pipeline
translator = pipeline("translation_en_to_de", model=latest_file)


# In[ ]:


#Program to take user input in English text and translating it into German text
while True:
    test_translation = input("Input English text to translate it into German: ")
    print(translator(test_translation))

