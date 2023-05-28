English to German Translation Project

This repository contains the following files and folders:
- EN-DE Model Trainer.py
- English to German Translator.py
- README.md
- requirements.txt

This is a Natural Language Processing project that loads a dataset of English and German translation pairs, preprocesses and tokenises it, utilising a pre-trained model and training it on the dataset, while using the SacreBLEU metric for evaluations. The dataset used is the iwslt2017-en-de dataset, which has 215K rows. This project uses the T5-Small (Text-To-Text Transfer Transformer) model as a pre-trained model.

Instructions:
To get started, run the file 'EN-DE Model Trainer.py', which contains the entire process from loading and preprocessing the dataset all the way to training and finetuning the model for it.

After the training is done, run 'English to German Translator.py' to load the model. You can then test out the model for yourself by typing any English sentences in the input field and pressing 'Enter', which subsequently generates a German translation based on what it has been trained on.