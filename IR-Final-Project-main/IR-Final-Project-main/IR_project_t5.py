import pandas as pd
import numpy as np
import glob
import os
import datasets
from transformers import  TrainingArguments, TrainingArguments
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers.data.metrics.squad_metrics import compute_exact, compute_f1
from statistics import mean
from simpletransformers.t5 import T5Model
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer,AutoModelForSequenceClassification, TrainingArguments, Trainer,DataCollatorWithPadding
import sklearn
if __name__=='__main__':
    path = r"/home/chrachabathuni/Natural-Hazards-Twitter-Dataset/*.csv" # use your path
    all_files = glob.glob(path)
    all_files
    filenames=[]
    for p in all_files:
        filenames.append(p.split("/")[-1])


    df_dict = dict()
    for n,f in zip(filenames,all_files):
        df_dict[n] = pd.read_csv(f)
        if n=='2018Michael_Summary.csv':
            df_dict[n]['num']=df_dict[n]['Unnamed: 0']
            df_dict[n]['label']=df_dict[n]['sentiment']
            df_dict[n].drop(['sentiment','Unnamed: 0'],axis=1,inplace=True)
        df_dict[n]['filename']=[n]*len(df_dict[n])
        required_df=df_all[['text','label','filename']]
    df_all=pd.concat(df_dict.values(), ignore_index=True) 
    required_df=df_all[['text','label','filename']]

    def add_prefix(col,prefix='Binary Classification:'):
        for i in range(len(col)):
            col[i]=prefix+' '+col[i]
        return col
    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}
    required_df=required_df.replace({"label": id2label})
    required_df=required_df.rename(columns={"text": "input_text", "label": "target_text"})
    required_df['prefix']='Binary Classification'
    required_df.head()
    train_df, test_df = train_test_split(required_df, test_size=0.7,stratify=required_df[['filename','target_text']],random_state=0)

    model_args = {
    'output_'
    "max_seq_length": 196,
    "train_batch_size": 16,
    "eval_batch_size": 8,
    "num_train_epochs": 5,
    #"evaluate_during_training": True,
    "evaluate_during_training_steps": 15000,
    "evaluate_during_training_verbose": True,
    
    "use_multiprocessing": False,
    "fp16": False,

    "save_steps": -1,
    "save_eval_checkpoints": False,
    "save_model_every_epoch": False,

    "reprocess_input_data": True,
    "overwrite_output_dir": True,

    #"wandb_project": "T5 mixed tasks - Binary, Multi-Label, Regression",
    }
    model_id='t5-large'
    model = T5Model("t5", model_id, args=model_args)

    model.train_model(train_df,output_dir='/home/chrachabathuni/T5_for_IR/')#,acc=sklearn.metrics.accuracy_score,f1=sklearn.metrics.f1_score)
    to_predict = [
    prefix + ": " + str(input_text)
    for prefix, input_text in zip(test_df["prefix"].tolist(), test_df["input_text"].tolist())
    ]
    truth = test_df["target_text"].tolist()
    tasks = test_df["prefix"].tolist()

    # Get the model predictions
    preds = model.predict(to_predict)

    def f1(truths, preds):
        return mean([compute_f1(truth, pred) for truth, pred in zip(truths, preds)])
    print("The f1 score is:",f1(test_df['target_text'].values,preds))
    test_df['prediction']=preds
    test_df.to_csv('IR_'+model_id+'_natural_disasters_tweets.csv',index=False)
            