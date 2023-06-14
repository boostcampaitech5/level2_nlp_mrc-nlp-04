# 난수 고정
import json
import random
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from pprint import pprint

from sklearn.feature_extraction.text import TfidfVectorizer

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from datasets import load_dataset, load_from_disk
from transformers import (
    AutoTokenizer,
    BertModel, BertPreTrainedModel,
    AdamW, get_linear_schedule_with_warmup,
    TrainingArguments,
)


def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    random.seed(random_seed)
    np.random.seed(random_seed)


set_seed(42)  # magic number :)


class DenseRetrieval:

    def __init__(self, args, tokenizer, p_encoder, q_encoder, num_sample=1e10):
        '''
        학습과 추론에 사용될 여러 셋업을 마쳐봅시다.
        '''

        self.args = args
        train_dataset = load_from_disk("/opt/ml/input/data/train_dataset")
        self.train_dataset = train_dataset['train'][:num_sample]
        self.valid_dataset = train_dataset['validation'][:num_sample]
        del train_dataset
        self.test_dataset = load_from_disk(
            "/opt/ml/input/data/test_dataset")['validation'][:num_sample]
        self.tokenizer = tokenizer
        self.p_encoder = p_encoder
        self.q_encoder = q_encoder

        self.prepare_in_batch_negative()

    def prepare_in_batch_negative(self):
        tokenizer = self.tokenizer

        # 1. in-batch 만들어주기
        # 정작 in-batch 아니어서 삭제함

        # 2. (Question, Passage) 데이터셋 만들어주기
        q_seqs = tokenizer(
            self.train_dataset['question'], padding="max_length",
            truncation=True, return_tensors='pt'
        )
        p_seqs = tokenizer(
            self.train_dataset['context'], padding="max_length",
            truncation=True, return_tensors='pt'
        )
        train_dataset = TensorDataset(
            p_seqs['input_ids'], p_seqs['attention_mask'], p_seqs['token_type_ids'],
            q_seqs['input_ids'], q_seqs['attention_mask'], q_seqs['token_type_ids']
        )
        self.train_dataloader = DataLoader(
            train_dataset, shuffle=True, batch_size=self.args.per_device_train_batch_size, drop_last=False)
        ##
        valid_p_seqs = tokenizer(
            self.valid_dataset['context'], padding="max_length",
            truncation=True, return_tensors='pt'
        )
        valid_dataset = TensorDataset(
            valid_p_seqs['input_ids'], valid_p_seqs['attention_mask'], valid_p_seqs['token_type_ids']
        )
        self.valid_dataloader = DataLoader(
            valid_dataset, batch_size=self.args.per_device_train_batch_size, drop_last=False)
        ##
        test_q_seqs = tokenizer(
            self.test_dataset['question'], padding="max_length",
            truncation=True, return_tensors='pt'
        )
        test_dataset = TensorDataset(
            test_q_seqs['input_ids'], test_q_seqs['attention_mask'], test_q_seqs['token_type_ids']
        )
        self.test_dataloader = DataLoader(
            test_dataset, batch_size=self.args.per_device_train_batch_size, drop_last=False)

    def train(self):
        args = self.args

        # Optimizer
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.p_encoder.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in self.p_encoder.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0},
            {'params': [p for n, p in self.q_encoder.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in self.q_encoder.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=args.learning_rate, eps=args.adam_epsilon)
        t_total = len(
            self.train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

        # Start training!
        global_step = 0

        self.p_encoder.zero_grad()
        self.q_encoder.zero_grad()
        torch.cuda.empty_cache()

        train_iterator = tqdm(range(int(args.num_train_epochs)), desc="Epoch")
        for _ in train_iterator:
            with tqdm(self.train_dataloader, unit="batch") as tepoch:
                for batch in tepoch:
                    p_encoder.train()
                    q_encoder.train()

                    # target: position of positive samples = diagonal element
                    targets = torch.arange(
                        0, batch[0].size()[0]).long().to(args.device)

                    p_inputs = {
                        'input_ids': batch[0].to(args.device),
                        'attention_mask': batch[1].to(args.device),
                        'token_type_ids': batch[2].to(args.device)
                    }
                    q_inputs = {
                        'input_ids': batch[3].to(args.device),
                        'attention_mask': batch[4].to(args.device),
                        'token_type_ids': batch[5].to(args.device)
                    }

                    p_outputs = self.p_encoder(**p_inputs)
                    q_outputs = self.q_encoder(**q_inputs)
                    # (batch_size, emb_dim)

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

    def valid(self):
        # print, top-k-retrieval-rate
        pass

    def get_relevant_doc(self, query, k=1):
        args = self.args
        p_encoder = self.p_encoder
        q_encoder = self.q_encoder

        with torch.no_grad():
            p_encoder.eval()
            q_encoder.eval()

            q_seqs_val = self.tokenizer(
                [query], padding="max_length", truncation=True, return_tensors='pt').to(args.device)
            q_emb = q_encoder(**q_seqs_val).to('cpu')  # (num_query=1, emb_dim)

            p_embs = []
            for batch in self.test_dataloader:

                p_inputs = {
                    'input_ids': batch[0].to(args.device),
                    'attention_mask': batch[1].to(args.device),
                    'token_type_ids': batch[2].to(args.device)
                }
                p_emb = p_encoder(**p_inputs).to('cpu')
                p_embs.append(p_emb)

        p_embs = torch.cat(p_embs, dim=0)
        dot_prod_scores = torch.matmul(q_emb, torch.transpose(p_embs, 0, 1))
        rank = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze()
        return rank[:k]


class BertEncoder(BertPreTrainedModel):

    def __init__(self, config):
        super(BertEncoder, self).__init__(config)

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


# 데이터셋과 모델은 아래와 같이 불러옵니다.
train_dataset = load_from_disk("/opt/ml/input/data/train_dataset")['train']

# 메모리가 부족한 경우 일부만 사용하세요 !
num_sample = 300  # 1500

args = TrainingArguments(
    output_dir="dense_retireval",
    evaluation_strategy="epoch",
    learning_rate=3e-4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    weight_decay=0.01
)
model_checkpoint = 'klue/bert-base'

# 혹시 위에서 사용한 encoder가 있다면 주석처리 후 진행해주세요 (CUDA ...)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
p_encoder = BertEncoder.from_pretrained(model_checkpoint).to(args.device)
q_encoder = BertEncoder.from_pretrained(model_checkpoint).to(args.device)


# Retriever는 아래와 같이 사용할 수 있도록 코드를 짜봅시다.
retriever = DenseRetrieval(args=args,
                           tokenizer=tokenizer, p_encoder=p_encoder, q_encoder=q_encoder,
                           num_sample=num_sample
                           )
retriever.train()

query = '대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?'
results = retriever.get_relevant_doc(query=query, k=num_sample)


print(f"[Search Query] {query}\n")

indices = results.tolist()
for i, idx in enumerate(indices):
    print(f"Top-{i + 1}th Passage (Index {idx})")
    pprint(retriever.dataset['context'][idx][0:70])
    if idx == 0:
        print('@@@@@@@@@@@ Hit @@@@@@@@@@@')
        break
