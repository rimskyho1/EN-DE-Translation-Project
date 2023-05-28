#!/usr/bin/env python
# coding: utf-8

# In[1]:


from datasets import load_dataset

ds = load_dataset("iwslt2017", "iwslt2017-en-de")


# In[2]:


len(ds['train'])


# In[3]:


len(ds['test'])


# In[5]:


ds['train'][0:10]


# In[6]:


from transformers import AutoTokenizer

#Loads the T5 tokenizer to process the pairs of English and German sentences
checkpoint = 't5-small'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


# In[7]:


source_lang = 'en'
target_lang = 'de'
prefix = 'translate English to German: '

#Define a function for preprocessing 
def preprocess_function(examples):
#Prefix the input with a prompt so that T5 knows it is for translation 
    inputs = [prefix + example[source_lang] for example in examples['translation']] 
#Separately tokenize both English and German, then truncate it so it isn't longer than the set max_length
    targets = [example[target_lang] for example in examples['translation']]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
    return model_inputs


# In[8]:


#Apply the preprocess function over the entire dataset using the map method of Datasets
tokenized_ds = ds.map(preprocess_function, batched=True)


# In[9]:


from transformers import DataCollatorForSeq2Seq
#Creating a batch of samples
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)


# In[10]:


import evaluate
#Loading the Sacrebleu metric for evaluation
metric = evaluate.load("sacrebleu")


# In[11]:


import numpy as np

#Function to strip and extract the predictions and labels
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

#Function to compute the metrics with Sacrebleu
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    
    return result


# In[12]:


from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)


# In[13]:


#Training the model with PyTorch, and exporting the model into the folder specified by output_dir
training_args = Seq2SeqTrainingArguments(
    output_dir="EN-DE PyTorch Translator",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=2,
    predict_with_generate=True,
    fp16=True,
    push_to_hub=False,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

