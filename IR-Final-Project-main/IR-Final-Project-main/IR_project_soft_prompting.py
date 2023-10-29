import math
import glob
from tqdm import tqdm
import numpy as np
from transformers import MT5ForConditionalGeneration, T5Tokenizer,T5ForConditionalGeneration
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import re
import emoji
if __name__=='__main__':
    class SoftEmbedding(nn.Module):
        def __init__(self, 
                    wte: nn.Embedding,
                    n_tokens: int = 10, 
                    random_range: float = 0.5,
                    initialize_from_vocab: bool = True):
            """appends learned embedding to 
            Args:
                wte (nn.Embedding): original transformer word embedding
                n_tokens (int, optional): number of tokens for task. Defaults to 10.
                random_range (float, optional): range to init embedding (if not initialize from vocab). Defaults to 0.5.
                initialize_from_vocab (bool, optional): initalizes from default vocab. Defaults to True.
            """
            super(SoftEmbedding, self).__init__()
            self.wte = wte
            self.n_tokens = n_tokens
            self.learned_embedding = nn.parameter.Parameter(self.initialize_embedding(wte,
                                                                                    n_tokens, 
                                                                                    random_range, 
                                                                                    initialize_from_vocab))
                
        def initialize_embedding(self, 
                                wte: nn.Embedding,
                                n_tokens: int = 10, 
                                random_range: float = 0.5, 
                                initialize_from_vocab: bool = True):
            """initializes learned embedding
            Args:
                same as __init__
            Returns:
                torch.float: initialized using original schemes
            """
            if initialize_from_vocab:
                return self.wte.weight[:n_tokens].clone().detach()
            return torch.FloatTensor(n_tokens, wte.weight.size(1)).uniform_(-random_range, random_range)
                
        def forward(self, tokens):
            """run forward pass
            Args:
                tokens (torch.long): input tokens before encoding
            Returns:
                torch.float: encoding of text concatenated with learned task specifc embedding
            """
            input_embedding = self.wte(tokens[:, self.n_tokens:])
            learned_embedding = self.learned_embedding.repeat(input_embedding.size(0), 1, 1)
            return torch.cat([learned_embedding, input_embedding], 1)
        


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
    df_all=pd.concat(df_dict.values(), ignore_index=True) 
    required_df=df_all[['text','label','filename']]

    # def add_prefix(col,prefix='Binary Classification:'):
    #     for i in range(len(col)):
    #         col[i]=prefix+' '+col[i]
    #     return col
    # id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    # label2id = {"NEGATIVE": 0, "POSITIVE": 1}
    # required_df=required_df.replace({"label": id2label})
    # required_df=required_df.rename(columns={"text": "input_text", "label": "target_text"})
    # required_df['prefix']='Binary Classification'
    required_df.head()
    train, test = train_test_split(required_df, test_size=0.7,stratify=required_df[['filename','target_text']],random_state=0,shuffle=False)
    title_train=train['text']
    label_train=train['label']
    title_test=test['text']
    label_test=test['label']
    def generate_data(batch_size, n_tokens, title_data, label_data):

        labels = [
            torch.tensor([[3]]),  # \x00
            torch.tensor([[4]]),  # \x01
            #torch.tensor([[5]]),  # \x02
        ]

        def yield_data(x_batch, y_batch, l_batch):
            x = torch.nn.utils.rnn.pad_sequence(x_batch, batch_first=True)
            y = torch.cat(y_batch, dim=0)
            m = (x > 0).to(torch.float32)
            decoder_input_ids = torch.full((x.size(0), n_tokens), 1)
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
                m = m.cuda()
                decoder_input_ids = decoder_input_ids.cuda()
            return x, y, m, decoder_input_ids, l_batch

        x_batch, y_batch, l_batch = [], [], []
        for x, y in zip(title_data, label_data):
            context = x
            inputs = tokenizer(context, return_tensors="pt")
            inputs['input_ids'] = torch.cat([torch.full((1, n_tokens), 1), inputs['input_ids']], 1)
            l_batch.append(y)
            y = labels[y]
            y = torch.cat([torch.full((1, n_tokens - 1), -100), y], 1)
            x_batch.append(inputs['input_ids'][0])
            y_batch.append(y)
            if len(x_batch) >= batch_size:
                yield yield_data(x_batch, y_batch, l_batch)
                x_batch, y_batch, l_batch = [], [], []

        if len(x_batch) > 0:
            yield yield_data(x_batch, y_batch, l_batch)
            x_batch, y_batch, l_batch = [], [], []
    model_id='google/flan-t5-xl'
    model = T5ForConditionalGeneration.from_pretrained(model_id)
    tokenizer = T5Tokenizer.from_pretrained(model_id)
    n_tokens = 100
    s_wte = SoftEmbedding(model.get_input_embeddings(), 
                        n_tokens=n_tokens, 
                        initialize_from_vocab=True)
    model.set_input_embeddings(s_wte)
    if torch.cuda.is_available():
        model = model.cuda()
    parameters = list(model.parameters())
    for x in parameters[1:]:
        x.requires_grad = False

    #Training and validation    
    batch_size = 2
    n_epoch = 5
    total_batch = math.ceil(len(title_train) / batch_size)
    dev_total_batch = math.ceil(len(title_test) / batch_size)
    use_ce_loss = False
    ce_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(s_wte.parameters(), lr=0.5)
    dev_acc=-99
    counter=3
    for epoch in range(n_epoch):
        print('epoch', epoch)

        all_true_labels = []
        all_pred_labels = []
        losses = []
        pbar = tqdm(enumerate(generate_data(batch_size, n_tokens, title_train, label_train)), total=total_batch)
        for i, (x, y, m, dii, true_labels) in pbar:
            all_true_labels += true_labels
            
            optimizer.zero_grad()
            outputs = model(input_ids=x, labels=y, attention_mask=m, decoder_input_ids=dii)
            pred_labels = outputs['logits'][:, -1, 3:5].argmax(-1).detach().cpu().numpy().tolist()
            all_pred_labels += pred_labels

            if use_ce_loss:
                logits = outputs['logits'][:, -1, 3:5]
                true_labels_tensor = torch.tensor(true_labels, dtype=torch.long).cuda()
                loss = ce_loss(logits, true_labels_tensor)
            else:
                loss = outputs.loss
            loss.backward()
            optimizer.step()
            loss_value = float(loss.detach().cpu().numpy().tolist()) / batch_size
            losses.append(loss_value)

            acc = accuracy_score(all_true_labels, all_pred_labels)
            pbar.set_description(f'train: loss={np.mean(losses):.4f}, acc={acc:.4f}')

        all_true_labels = []
        all_pred_labels = []
        losses = []
        with torch.no_grad():
            pbar = tqdm(enumerate(generate_data(batch_size, n_tokens, title_test, label_test)), total=dev_total_batch)
            for i, (x, y, m, dii, true_labels) in pbar:
                all_true_labels += true_labels
                outputs = model(input_ids=x, labels=y, attention_mask=m, decoder_input_ids=dii)
                loss = outputs.loss
                loss_value = float(loss.detach().cpu().numpy().tolist()) / batch_size
                losses.append(loss_value)
                pred_labels = outputs['logits'][:, -1, 3:5].argmax(-1).detach().cpu().numpy().tolist()
                all_pred_labels += pred_labels
                acc = accuracy_score(all_true_labels, all_pred_labels)
                pbar.set_description(f'dev: loss={np.mean(losses):.4f}, acc={acc:.4f}')
        if dev_acc<acc:
            dev_acc=acc
        else:
            counter+=1
        if counter==3:
            print("Early stop")
            break
            
    def predict(text):
        inputs = tokenizer(text, return_tensors='pt')
        inputs['input_ids'] = torch.cat([torch.full((1, n_tokens), 1), inputs['input_ids']], 1)

        decoder_input_ids = torch.full((1, n_tokens), 1)
        with torch.no_grad():
            outputs = model(input_ids=inputs['input_ids'].cuda(), decoder_input_ids=decoder_input_ids.cuda())
        logits = outputs['logits'][:, -1, 3:5]
        pred = logits.argmax(-1).detach().cpu().numpy()[0]
        # print(logits)
        return pred
    preds = []
    for i in tqdm(range(len(title_test))):
        preds.append(predict(title_test[i]))
        #rets.append((label_test[i], pred, title_test[i]))
    test['predictions']=preds
    test.to_csv("IR_soft_prompting_"+model_id.split('/')[1]+"_ndt.csv")