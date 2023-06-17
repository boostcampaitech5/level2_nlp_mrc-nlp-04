import json
import random
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from pprint import pprint
import os
import pickle
from typing import List, Optional, Tuple, Union

from sklearn.feature_extraction.text import TfidfVectorizer

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from datasets import Dataset, load_dataset, load_from_disk
from transformers import (
    AutoTokenizer,
    BertModel, BertPreTrainedModel,
    AdamW, get_linear_schedule_with_warmup,
    TrainingArguments,
)
from collections import deque
import GPUtil as GPU


def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    random.seed(random_seed)
    np.random.seed(random_seed)


set_seed(42)  # magic number :)


class dprDenseRetrieval:

    def __init__(self, args, tokenizer, p_encoder, q_encoder, num_sample):
        '''
        학습과 추론에 사용될 여러 셋업을 마쳐봅시다.
        '''
        if num_sample is None:
            num_sample = int(1e9)
        self.args = args
        train_dataset = load_from_disk("/opt/ml/input/data/train_dataset")
        self.train_dataset = train_dataset['train'][:num_sample]
        self.valid_dataset = train_dataset['validation']
        del train_dataset
        self.test_dataset = load_from_disk(
            "/opt/ml/input/data/test_dataset")['validation']
        self.tokenizer = tokenizer
        self.p_encoder = p_encoder
        self.q_encoder = q_encoder

        with open('/opt/ml/input/data/wikipedia_documents.json', "r", encoding="utf-8") as f:
            wiki = json.load(f)
        self.contexts = list(
            dict.fromkeys([v["text"] for v in wiki.values()])
        )
        self.contexts = self.contexts[:num_sample]

        self.prepare_in_batch_negative()

    def prepare_in_batch_negative(self):
        q_seqs = self.tokenizer(
            self.train_dataset['question'], padding="max_length",
            truncation=True, return_tensors='pt'
        )
        p_seqs = self.tokenizer(
            self.train_dataset['context'], padding="max_length",
            truncation=True, return_tensors='pt'
        )
        train_dataset = TensorDataset(
            p_seqs['input_ids'], p_seqs['attention_mask'], p_seqs['token_type_ids'],
            q_seqs['input_ids'], q_seqs['attention_mask'], q_seqs['token_type_ids']
        )
        self.train_dataloader = DataLoader(
            train_dataset, shuffle=True, batch_size=self.args.per_device_train_batch_size, drop_last=False)
        ###################################
        valid_p_seqs = self.tokenizer(
            self.valid_dataset['context'], padding="max_length",
            truncation=True, return_tensors='pt'
        )
        valid_dataset = TensorDataset(
            valid_p_seqs['input_ids'], valid_p_seqs['attention_mask'], valid_p_seqs['token_type_ids']
        )
        self.valid_dataloader = DataLoader(
            valid_dataset, batch_size=self.args.per_device_train_batch_size, drop_last=False)
        ###################################
        test_q_seqs = self.tokenizer(
            self.test_dataset['question'], padding="max_length",
            truncation=True, return_tensors='pt'
        )
        test_dataset = TensorDataset(
            test_q_seqs['input_ids'], test_q_seqs['attention_mask'], test_q_seqs['token_type_ids']
        )
        self.test_dataloader = DataLoader(
            test_dataset, batch_size=self.args.per_device_train_batch_size, drop_last=False)
        ###################################
        wiki_seqs = self.tokenizer(
            self.contexts, padding="max_length",
            truncation=True, return_tensors='pt'
        )
        wiki_dataset = TensorDataset(
            wiki_seqs['input_ids'], wiki_seqs['attention_mask'], wiki_seqs['token_type_ids']
        )
        self.wiki_dataloader = DataLoader(
            wiki_dataset, batch_size=self.args.per_device_train_batch_size, drop_last=False)


    def train(self, override: bool=False, num_pre_batch: int=0):
        '''
        p_encoder와 q_encoder를 학습합니다.
        저장된 stats_dict가 없으면 저장하고, 있으면 로드합니다.

        Args:
            override: True라면 무조건 학습을 다시 시작합니다.
            num_pre_batch: pre-batch negatives에 사용될 p_outputs의 숫자입니다.
        '''
        p_name = f'p_encoder_statesdict'
        q_name = f'q_encoder_statesdict'
        p_path = os.path.join('/opt/ml/output/models/dpr', p_name)
        q_path = os.path.join('/opt/ml/output/models/dpr', q_name)
        if os.path.isfile(p_path) and os.path.isfile(q_path) and (not override):
            self.p_encoder.load_state_dict(torch.load(p_path))
            self.p_encoder.to(self.args.device)
            self.q_encoder.load_state_dict(torch.load(q_path))
            self.q_encoder.to(self.args.device)
            print('encoder statedict loaded')
            return
        
        print('Training p_encoder&q_encoder starts')
        # Optimizer
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.p_encoder.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.p_encoder.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0},
            {'params': [p for n, p in self.q_encoder.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.q_encoder.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        t_total = len(
            self.train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total)

        # Start training!
        global_step = 0

        self.p_encoder.zero_grad()
        self.q_encoder.zero_grad()
        torch.cuda.empty_cache()

        train_iterator = tqdm(range(int(self.args.num_train_epochs)), desc="Epoch")
        for _ in train_iterator:
            p_deque = deque(maxlen=num_pre_batch+1) #자기 자신 포함이라 +1
            with tqdm(self.train_dataloader, desc='train', unit="batch") as tepoch:
                for batch in tepoch:
                    self.p_encoder.train()
                    self.q_encoder.train()

                    # target: position of positive samples = diagonal element
                    targets = torch.arange(
                        0, batch[0].size()[0]).long().to(self.args.device)

                    p_inputs = {
                        'input_ids': batch[0].to(self.args.device),
                        'attention_mask': batch[1].to(self.args.device),
                        'token_type_ids': batch[2].to(self.args.device)
                    }
                    q_inputs = {
                        'input_ids': batch[3].to(self.args.device),
                        'attention_mask': batch[4].to(self.args.device),
                        'token_type_ids': batch[5].to(self.args.device)
                    }

                    p_outputs = self.p_encoder(**p_inputs)
                    q_outputs = self.q_encoder(**q_inputs)
                    # GPU.showUtilization()
                    # print('\n', '@@ current memory')
                    # (batch_size, emb_dim)

                    ## pre-batch negatives 
                    p_deque.append(p_outputs.detach()) # max_len 선언됨
                    for i in range(len(p_deque)-1):
                        temp = p_deque.popleft()
                        p_outputs = torch.cat((p_outputs, temp), dim=0)
                        p_deque.append(temp)
                    ## pre-batch negatives 

                    # Calculate similarity score & loss
                    sim_scores = torch.matmul(q_outputs, torch.transpose(
                        p_outputs, 0, 1)).squeeze()
                    sim_scores = F.log_softmax(sim_scores, dim=1)
                    loss = F.nll_loss(sim_scores, targets)
                    tepoch.set_postfix(loss=f'{str(loss.item())}')

                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    self.p_encoder.zero_grad()
                    self.q_encoder.zero_grad()
                    global_step += 1

                    torch.cuda.empty_cache()
                    del p_inputs, q_inputs
        if not os.path.exists('/opt/ml/output/models/dpr'):
            os.makedirs('/opt/ml/output/models/dpr')
        torch.save(self.p_encoder.state_dict(), p_path)
        torch.save(self.q_encoder.state_dict(), q_path)
        print('encoder statedict saved')


    def valid_rate(self, doc_indices, topk):
        # todo
        # if name main 맨 밑의 방식으로 구현함.
        pass


    def build_faiss():
        # todo
        pass

    def get_p_embedding(self, override: bool=False):
        '''
        전체 위키피디아 5만 7천개 혹은 num_sample로 p_embs를 만듭니다.
        저장된 피클이 없다면 만든 뒤 저장하고, 있다면 로드합니다.
        '''
        emb_name = f"dpr_p_embedding.bin"
        emb_path = os.path.join('/opt/ml/output/models/dpr', emb_name)

        if os.path.isfile(emb_path) and (not override):
            with open(emb_path, "rb") as file:
                self.p_embs = pickle.load(file)
            print("Embedding pickle load.")
        else:
            with torch.no_grad():
                self.p_encoder.eval()

                self.p_embs = []
                for batch in tqdm(self.wiki_dataloader, desc='wiki_p_embs'):
                    p_inputs = {
                        'input_ids': batch[0].to(self.args.device),
                        'attention_mask': batch[1].to(self.args.device),
                        'token_type_ids': batch[2].to(self.args.device)
                    }
                    p_emb = self.p_encoder(**p_inputs).to('cpu')
                    self.p_embs.append(p_emb)
            self.p_embs = torch.cat(self.p_embs, dim=0)
            print('p_embs.shape is', self.p_embs.shape)
            with open(emb_path, 'wb') as file:
                pickle.dump(self.p_embs, file)
            print("Embedding pickle saved.")

    def get_testq_embedding(self, override: bool=False):
        '''
        test_dataset 600개로 q_embs를 만듭니다.
        저장된 피클이 없다면 만든 뒤 저장하고, 있다면 로드합니다.
        '''
        emb_name = f"dpr_testq_embedding.bin"
        emb_path = os.path.join('/opt/ml/output/models/dpr', emb_name)

        if os.path.isfile(emb_path) and (not override):
            with open(emb_path, "rb") as file:
                self.testq_embs = pickle.load(file)
            print("Embedding pickle load.")
        else:
            with torch.no_grad():
                self.q_encoder.eval()

                self.testq_embs = []
                for batch in tqdm(self.test_dataloader, desc='600_q_embs'):
                    q_inputs = {
                        'input_ids': batch[0].to(self.args.device),
                        'attention_mask': batch[1].to(self.args.device),
                        'token_type_ids': batch[2].to(self.args.device)
                    }
                    q_emb = self.q_encoder(**q_inputs).to('cpu')
                    self.testq_embs.append(q_emb)
            self.testq_embs = torch.cat(self.testq_embs, dim=0)
            print('q_embs.shape is', self.testq_embs.shape)
            with open(emb_path, 'wb') as file:
                pickle.dump(self.testq_embs, file)
            print("Embedding pickle saved.")

    def get_validq_embedding(self, override: bool=False):
        '''
        valid_dataset 240개로 q_embs를 만듭니다.
        저장된 피클이 없다면 만든 뒤 저장하고, 있다면 로드합니다.
        '''
        emb_name = f"dpr_validq_embedding.bin"
        emb_path = os.path.join('/opt/ml/output/models/dpr', emb_name)

        if os.path.isfile(emb_path) and (not override):
            with open(emb_path, "rb") as file:
                self.validq_embs = pickle.load(file)
            print("Embedding pickle load.")
        else:
            with torch.no_grad():
                self.q_encoder.eval()

                self.validq_embs = []
                for batch in tqdm(self.valid_dataloader, desc='240_q_embs'):
                    q_inputs = {
                        'input_ids': batch[0].to(self.args.device),
                        'attention_mask': batch[1].to(self.args.device),
                        'token_type_ids': batch[2].to(self.args.device)
                    }
                    q_emb = self.q_encoder(**q_inputs).to('cpu')
                    self.validq_embs.append(q_emb)
            self.validq_embs = torch.cat(self.validq_embs, dim=0)
            print('q_embs.shape is', self.validq_embs.shape)
            with open(emb_path, 'wb') as file:
                pickle.dump(self.validq_embs, file)
            print("Embedding pickle saved.")

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:
        """
        Arguments:
            query_or_dataset (Union[str, Dataset]):
                str이나 Dataset으로 이루어진 Query를 받습니다.
                str 형태인 하나의 query만 받으면 `get_relevant_doc`을 통해 유사도를 구합니다.
                Dataset 형태는 query를 포함한 HF.Dataset을 받습니다.
                이 경우 `get_relevant_doc_bulk`를 통해 유사도를 구합니다.
            topk (Optional[int], optional): Defaults to 1.
                상위 몇 개의 passage를 사용할 것인지 지정합니다.

        Returns:
            1개의 Query를 받는 경우  -> Tuple(List, List)
            다수의 Query를 받는 경우 -> pd.DataFrame: [description]

        Note:
            다수의 Query를 받는 경우,
                Ground Truth가 있는 Query (train/valid) -> 기존 Ground Truth Passage를 같이 반환합니다.
                Ground Truth가 없는 Query (test) -> Retrieval한 Passage만 반환합니다.
        """
        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print(f"Top-{i+1} passage with score {doc_scores[i]:4f}")
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):
            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            total = []
            doc_scores, doc_indices = self.get_relevant_doc_bulk(
                query_or_dataset["question"], k=topk
            )

            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="DPR retrieval: ")
            ):
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context": " ".join(
                        [self.contexts[pid] for pid in doc_indices[idx]]
                    ),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas

    def get_relevant_doc(self, query, k=1):
        with torch.no_grad():
            self.q_encoder.eval()
            q_seqs_val = self.tokenizer(
                [query], padding="max_length", truncation=True, return_tensors='pt').to(self.args.device)
            q_emb = self.q_encoder(**q_seqs_val).to('cpu')  # (num_query=1, emb_dim)
        result = torch.matmul(q_emb, torch.transpose(self.p_embs, 0, 1))
        sorted_result = torch.argsort(result, dim=1, descending=True).squeeze()
        doc_score = result.squeeze()[sorted_result][:k]
        doc_indices = sorted_result.tolist()[:k]
        print(doc_score, doc_indices)
        return doc_score, doc_indices
    
    def get_relevant_doc_bulk(self, query, k=1):
        if len(query)==240:
            print('current len is', len(query)) 
            q_embs = self.validq_embs
        elif len(query)==600:
            print('current len is', len(query)) 
            q_embs = self.testq_embs
        else :
            print('current len is', len(query))
            raise Exception

        result = torch.matmul(q_embs, torch.transpose(self.p_embs, 0, 1)) #(600, 5만)
        doc_scores = []
        doc_indices = []
        for i in range(result.shape[0]):
            sorted_result = torch.argsort(result[i, :], dim=-1, descending=True).squeeze()
            doc_scores.append(result[i, :][sorted_result].tolist()[:k])
            doc_indices.append(sorted_result.tolist()[:k])

        return doc_scores, doc_indices
    

class dprBertEncoder(BertPreTrainedModel):

    def __init__(self, config):
        super(dprBertEncoder, self).__init__(config)

        self.bert = BertModel(config)
        self.init_weights()

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        pooled_output = outputs[1]
        return pooled_output


if __name__ == '__main__':
    # 데이터셋과 모델은 아래와 같이 불러옵니다.
    train_dataset = load_from_disk("/opt/ml/input/data/train_dataset")['validation']


    # 메모리가 부족한 경우 일부만 사용하세요 !
    num_sample = None  # None or positive integer
    num_pre_batch = 0
    t_or_f = True
    topk = 10

    args = TrainingArguments(
        output_dir="dense_retireval",
        evaluation_strategy="epoch",
        learning_rate=1e-3,
        per_device_train_batch_size=6,
        per_device_eval_batch_size=6,
        num_train_epochs=3,
        weight_decay=0.01
    )
    model_checkpoint = 'klue/bert-base'

    # 혹시 위에서 사용한 encoder가 있다면 주석처리 후 진행해주세요 (CUDA ...)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
    p_encoder = dprBertEncoder.from_pretrained(model_checkpoint).to(args.device)
    q_encoder = dprBertEncoder.from_pretrained(model_checkpoint).to(args.device)


    # Retriever는 아래와 같이 사용할 수 있도록 코드를 짜봅시다.
    retriever = dprDenseRetrieval(args=args,
                            tokenizer=tokenizer, p_encoder=p_encoder, q_encoder=q_encoder,
                            num_sample=num_sample
                            )
    retriever.train(override=t_or_f, num_pre_batch=num_pre_batch)
    retriever.get_p_embedding(override=t_or_f)
    retriever.get_testq_embedding(override=t_or_f)
    retriever.get_validq_embedding(override=t_or_f)
    results = retriever.retrieve(query_or_dataset=train_dataset, topk=topk)

    print('Now print results')
    try:
        print('results = df')
        print(results.head())
    except:
        print('results = tuple(list, list)')
        print(results)

    try:
        results['rate'] = results.apply(lambda row: row['original_context'] in row['context'], axis=1)
        print(f'topk is {topk}, rate is {100*sum(results["rate"])/240}%')
    except:
        print('topk retrieval rate can\'t be printed. It is not train-valid set')