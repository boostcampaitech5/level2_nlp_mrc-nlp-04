import torch
import os
import numpy as np
import pickle
import pandas as pd

from tqdm import tqdm, trange
from transformers import AdamW, get_linear_schedule_with_warmup
from datasets import DatasetDict, concatenate_datasets
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from typing import List
from CustomScheduler import CosineAnnealingWarmUpRestarts
from BM25Embedding import make_bm25_embedding

# TODO: Compute Metric 구현
class BiEncoderTrainer:
	"""
		q_encoder, p_encoder를 받아 DPR(Dense Phrases Retrieval)을 수행하는 Trainer

		Attributes:
			q_encoder: 질문을 인코딩하는 모델
			p_encoder: 문서를 인코딩하는 모델
			tokenizer: 토크나이저
			batch_size: 배치 사이즈
			epochs: 학습 에폭 수
			lr: learning rate
			gradient_accumulation_steps: gradient accumulation step
			train_datasets: 학습 데이터셋
			eval_datasets: 평가 데이터셋
			contexts_document: 문서 context
			q_max_length: 질문 최대 길이
			p_max_length: 문서 최대 길이
			neg_num: negative sample 개수
			warmup_rate: warmup 비율
	"""
	def __init__(self,
				 q_encoder, p_encoder, tokenizer, batch_size=16, epochs=3, lr=3e-5,
				 train_datasets=None, eval_datasets=None, contexts_document=None, q_max_length=50, p_max_length=300, neg_num=2, warmup_rate=0.1):

		self.scheduler = None
		self.t_total = None
		self.optimizer = None
		self.optimizer_grouped_parameters = None

		self.q_encoder = q_encoder
		self.p_encoder =  p_encoder
		self.tokenizer = tokenizer

		self.batch_size = batch_size
		self.epochs = epochs
		self.lr = lr

		self.train_datasets = train_datasets
		self.eval_datasets = eval_datasets
		self.contexts_document = contexts_document
		self.q_max_length = q_max_length
		self.p_max_length = p_max_length
		self.neg_num = neg_num
		self.warmup_rate = warmup_rate

		self.full_ds = concatenate_datasets([train_datasets.flatten_indices(),
											 eval_datasets.flatten_indices()])
		print("BM25 Embedding Loading... \n ** It takes a long time. **")
		if os.path.exists(os.path.join(os.getcwd(), "input", "data", "wiki_bm25_embedding.bin")):
			with open(os.path.join(os.getcwd(), "input", "data", "wiki_bm25_embedding.bin"), "rb") as f:
				self.bm25 = pd.DataFrame(pickle.load(f), index=self.full_ds['id'])
		else:
			print("BM25 Model is not exist. Make BM25 Embedding.")
			self.bm25 = make_bm25_embedding(DATA_DIR=os.path.join(os.getcwd(), "input", "data"),
										   tokenizer=self.tokenizer,
										   full_ds=self.full_ds,
										   contexts=self.contexts_document)
		self.p_encoder_path = os.path.join(os.getcwd(), "input", "code", "dpr",
										   self.p_encoder.config.name_or_path.replace("/", "_"), "p_encoder")
		self.q_encoder_path = os.path.join(os.getcwd(), "input", "code", "dpr",
										   self.q_encoder.config.name_or_path.replace("/", "_"), "q_encoder")

	def train(self):
		"""
			학습을 수행하는 Method
		"""

		train_sampler = RandomSampler(self.train_datasets)
		train_dataloader = DataLoader(self.train_datasets, sampler=train_sampler, batch_size=self.batch_size)

		no_decay = ['bias', 'LayerNorm.weight']
		self.optimizer_grouped_parameters = [
					{'params': [p for n, p in self.p_encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
					{'params': [p for n, p in self.p_encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
					{'params': [p for n, p in self.q_encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
					{'params': [p for n, p in self.q_encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
		]
		self.optimizer = AdamW(self.optimizer_grouped_parameters, lr=self.lr, eps=1e-8)
		# CosineAnnealWarmUp Restarts Scheduler 사용 시 주석 해제하고 사용하세요.
		# self.optimizer = AdamW(self.optimizer_grouped_parameters, lr=0, eps=1e-8)
		self.t_total = len(train_dataloader) * self.epochs
		self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
														 num_warmup_steps=int(self.t_total*self.warmup_rate),
														 num_training_steps=self.t_total)
		# CosineAnnealWarmUp Restarts Scheduler 사용 시 주석 해제하고 사용하세요.
		# self.scheduler = CosineAnnealingWarmUpRestarts(self.optimizer,
		# 											   T_0 = int(self.t_total*self.warmup_rate),
		# 											   T_mult=2,
		# 											   eta_max=self.lr,
		# 											   T_up=int(self.t_total*self.warmup_rate/4),
		# 											   gamma=0.5)

		# Training Step
		global_step = 0
		self.p_encoder.zero_grad()
		self.q_encoder.zero_grad()
		torch.cuda.empty_cache()

		train_iterator = trange(int(self.epochs), desc="Epoch")

		for epoch, _ in enumerate(train_iterator):
			epoch_iterator = tqdm(train_dataloader, desc="Iteration")
			correct_cnt = 0
			avg_loss = 0
			for step, batch in enumerate(epoch_iterator):
				hard_negs = []
				cur_batch_size = len(batch['context'])
				self.p_encoder.train()
				self.q_encoder.train()

				# BM25 기반 Hard Negative Sampling
				for i, (q_id, query) in enumerate(zip(batch['id'], batch['question'])):
					search_context = np.argsort(self.bm25.loc[[q_id], :])[0, ::-1]
					hard_neg = []
					for cont in search_context:
						if self.contexts_document[cont] != batch['context'][i]:
							hard_neg.append(self.contexts_document[cont])
							if len(hard_neg) == self.neg_num:
								hard_negs.append(hard_neg)
								break


				p_inputs = self.tokenizer(batch['context'],
										  padding='max_length',
										  truncation=True,
										  max_length=self.p_max_length,
										  return_tensors='pt').to("cuda:0")
				q_inputs = self.tokenizer(batch['question'],
										  padding='max_length',
										  truncation=True,
										  max_length=self.q_max_length,
										  return_tensors='pt').to("cuda:0")

				p_outputs = self.p_encoder(**p_inputs)
				q_outputs = self.q_encoder(**q_inputs)

				# Hard Negative Sampling Simirality Score 계산
				hard_negs_sim_score = torch.Tensor([]).to("cuda:0")
				for i, hard_neg in enumerate(hard_negs):
					hard_neg_inputs = self.tokenizer(hard_neg,
													 padding='max_length',
													 truncation=True,
													 max_length=self.p_max_length,
													 return_tensors='pt').to("cuda:0")

					hard_neg_outputs = self.p_encoder(**hard_neg_inputs)  # [neg_num, hidden_size]
					hard_neg_sim_score = torch.matmul(q_outputs[i],
													  torch.transpose(hard_neg_outputs, 0, 1))  # [neg_num]
					hard_negs_sim_score = torch.concat([hard_negs_sim_score, hard_neg_sim_score.view(1, -1)], dim=0)

				p_sim_score = torch.matmul(q_outputs, torch.transpose(p_outputs, 0, 1)) # [batch_size, batch_size]
				sim_score = torch.cat((p_sim_score, hard_negs_sim_score), dim=1)  # [batch_size, batch_size + self.neg_num]
				targets = torch.arange(0, cur_batch_size).to(sim_score.device)
				if torch.cuda.is_available():
					targets = targets.to("cuda:0")

				sim_score = torch.nn.functional.log_softmax(sim_score, dim=1)
				# Top-5 Accuracy 계산
				for target, pred in zip(targets.view(-1, 1), torch.argsort(sim_score,dim=1, descending=True)[:, :5]):
					if target in pred:
						correct_cnt += 1

				loss = torch.nn.functional.nll_loss(sim_score, targets)

				loss.backward()
				self.optimizer.step()
				self.scheduler.step()
				self.p_encoder.zero_grad()
				self.q_encoder.zero_grad()

				epoch_iterator.set_postfix_str(f"Loss = {loss.detach():.4f}, lr = {self.optimizer.param_groups[0]['lr']:.4e}")
				global_step += 1

				torch.cuda.empty_cache()

			train_iterator.set_postfix_str(f"Top-5 Accuracy = {correct_cnt / len(train_dataloader.dataset):.4f}")

			self.eval()

			if epoch != self.epochs-1:
				self.p_encoder.save_pretrained(self.p_encoder_path + "_epoch" + str(epoch))
				self.q_encoder.save_pretrained(self.q_encoder_path + "_epoch" + str(epoch))
			else:
				self.p_encoder.save_pretrained(self.p_encoder_path+"_final")
				self.q_encoder.save_pretrained(self.q_encoder_path+"_final")

	# TODO: Validation Dataset 기준 Evaluate
	def eval(self):
		correct_cnt = 0
		total_cnt = 0
		print("=" * 10, "Evaluation Start", "=" * 10)
		val_q_seqs = self.tokenizer(self.eval_datasets['question'],
									padding="max_length",
									truncation=True,
									return_tensors="pt",
									max_length=self.q_max_length)
		val_p_seqs = self.tokenizer(self.eval_datasets['context'],
									padding="max_length",
									truncation=True,
									return_tensors="pt",
									max_length=self.p_max_length)

		val_dataset = TensorDataset(val_p_seqs['input_ids'], val_p_seqs['attention_mask'], val_p_seqs['token_type_ids'],
									val_q_seqs['input_ids'], val_q_seqs['attention_mask'], val_q_seqs['token_type_ids'])

		val_sampler = RandomSampler(val_dataset)
		val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=self.batch_size)

		val_iterator = tqdm(val_dataloader, desc="Validation")

		for step, batch in enumerate(val_iterator):
			with torch.no_grad():
				self.p_encoder.eval()
				self.q_encoder.eval()

				if torch.cuda.is_available():
					batch = tuple(t.to("cuda:0") for t in batch)
				p_inputs = {"input_ids": batch[0],
							"attention_mask": batch[1],
							"token_type_ids": batch[2]}
				q_inputs = {"input_ids": batch[3],
							"attention_mask": batch[4],
							"token_type_ids": batch[5]}

				p_outputs = self.p_encoder(**p_inputs)
				q_outputs = self.q_encoder(**q_inputs)

				sim_score = torch.matmul(q_outputs, torch.transpose(p_outputs, 0, 1))
				targets = torch.arange(0, batch[0].size(0)).long()

				if torch.cuda.is_available():
					targets = targets.to("cuda:0")

				sim_score = torch.nn.functional.log_softmax(sim_score, dim=1)

				for target, pred in zip(targets.view(-1, 1), torch.argsort(sim_score, dim=1, descending=True)[:, :5]):
					if target in pred:
						correct_cnt += 1

				total_cnt += batch[0].size(0)
				torch.cuda.empty_cache()

				del p_inputs, q_inputs, p_outputs, q_outputs, sim_score, targets
		print(f"============= Validation Top-5 Accuracy = {correct_cnt / total_cnt:.4f} =================")


	def predict(self, passage_dataset: DatasetDict, query: List[str], k: int=5):
		"""
			주어진 Passages내에서, query와 가장 유사한 Document를 k개 반환합니다.

			Params:
				passage_dataset: 질의를 검색할 Passage Dataset
				query: 질의할 Query List
				k: Top-k

			Return:
				pred_score: Query-Document간의 Top-k Score
				rank: Query-Document간의 Top-k Rank
				pred_corpus: Query-Passage간의 Top-k Document
		"""
		corpus, p_embs = self.__setup_for_eval__(passage_dataset)
		with torch.no_grad():
			self.q_encoder.eval()

			q_seqs = self.tokenizer(query,
									padding="max_length",
									truncation=True,
									return_tensors="pt",
									max_length=self.q_max_length).to("cuda:0")
			q_emb = self.q_encoder(**q_seqs).to('cpu')

		pred_score = torch.matmul(q_emb, torch.transpose(p_embs, 0, 1))
		rank = torch.argsort(pred_score, dim=1, descending=True).squeeze()
		pred_corpus = []
		for i in range(rank.size(0)):
			pred_corpus.append([corpus[idx] for idx in rank[i, :k]])
		return pred_score.gather(1, rank[:, :k]).tolist(), rank[:, :k].tolist(), pred_corpus

	def __setup_for_eval__(self, passage_dataset):
		p_embs = []
		corpus = list(set([example for example in passage_dataset['context']])) # set -> list로 다시 만든 이유는 잘 모르겠지만, 아마 context에 중복되는 문장이 존재해서 제거하려고 트릭을 사용한듯.
		with torch.no_grad():
			self.p_encoder.eval()

			for phrases in corpus:
				phrases = self.tokenizer(phrases,
										 padding="max_length",
										 truncation=True,
										 return_tensors="pt",
										 max_length=self.p_max_length).to("cuda:0")
				phrases_emb = self.p_encoder(**phrases).to('cpu').numpy()
				p_embs.append(phrases_emb)
		p_embs = torch.Tensor(p_embs).squeeze()

		return corpus, p_embs