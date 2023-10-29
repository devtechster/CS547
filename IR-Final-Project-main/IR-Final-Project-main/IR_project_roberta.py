import pandas as pd
import numpy as np
import glob
import re
import emoji
import os
import datasets
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification,Trainer, TrainingArguments,RobertaTokenizerFast, RobertaForSequenceClassification,Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
from transformers.models.roberta.modeling_roberta import (
    RobertaClassificationHead,
    RobertaConfig,
    RobertaModel,
)
class CONFIG:
    model_name='roberta-large'


if __name__=='__main__':
    path = r"/home/chrachabathuni/Natural-Hazards-Twitter-Dataset/*.csv" # use your path
    all_files = glob.glob(path)
    all_files
    filenames=[]
    for p in all_files:
        filenames.append(p.split("/")[-1])

        
    model = RobertaForSequenceClassification.from_pretrained(CONFIG.model_name)
    tokenizer = RobertaTokenizerFast.from_pretrained(CONFIG.model_name, max_length = 512)
        
    df_dict = dict()
    for n,f in zip(filenames,all_files):
        df_dict[n] = pd.read_csv(f)
        if n=='2018Michael_Summary.csv':
                df_dict[n]['num']=df_dict[n]['Unnamed: 0']
                df_dict[n]['label']=df_dict[n]['sentiment']
                df_dict[n].drop(['sentiment','Unnamed: 0'],axis=1,inplace=True)
        df_dict[n]['filename']=[n]*len(df_dict[n])
    df_all=pd.concat(df_dict.values(), ignore_index=True) 
    required_df=df_all[['text','label','filename']]
    def clean_text(line):
        clean_text=re.sub(r'(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)','',line)
        clean_text = re.sub("[^a-z0-9]"," ", clean_text)
        clean_text = re.sub("@[A-Za-z0-9]+","",clean_text) #Remove @ sign
        clean_text = " ".join(clean_text.split())
        clean_text = ''.join(c for c in clean_text if c not in emoji.EMOJI_DATA) #Remove Emojis
        clean_text = clean_text.lower()
        return clean_text
    required_df['text']=required_df['text'].apply(lambda x: clean_text(x))
    train, test = train_test_split(df_all, test_size=0.7,stratify=required_df[['filename','label']])
    class MyClassificationDataset(Dataset):
        #'input_ids', 'attention_mask', 'label'
        def __init__(self, data):
            text, labels = data['text'].values,data['label'].values
            text = [str(i) for i in data['text'].values]
            self.examples = tokenizer(text,truncation=True,padding="max_length",
                                    return_tensors="pt")
            self.labels = torch.tensor(labels, dtype=torch.long)
            

        def __len__(self):
            return len(self.examples["input_ids"])

        def __getitem__(self, index):
            return {'input_ids': self.examples['input_ids'][index],
                    'attention_mask':self.examples['attention_mask'][index],
                    'label':self.labels[index]}
    train_dataset = MyClassificationDataset(train)
    test_dataset = MyClassificationDataset(test)
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    training_args = TrainingArguments(
        output_dir = '/home/chrachabathuni/model_IR',
        num_train_epochs=10,
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 16,    
        per_device_eval_batch_size= 8,
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        disable_tqdm = False, 
        load_best_model_at_end=True,
        warmup_steps=500,
        weight_decay=0.01,
        logging_steps = 8,
        fp16 = True,
        #logging_dir='/media/jlealtru/data_files/github/website_tutorials/logs',
        dataloader_num_workers = 8,
        save_total_limit = 1
        #run_name = 'roberta-classification'
    )
    trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
    )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer.train()
    predictions, labels, metrics = trainer.predict(test_dataset, metric_key_prefix="predict")
    print(metrics)

    test['prediction']=labels
    test.to_csv(CONFIG.model_name+"_IR_natural_disaster_tweets_predictions.csv")