import torch

from tqdm import tqdm, trange
from transformers import AdamW, get_linear_schedule_with_warmup
from datasets import DatasetDict
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from typing import List

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
			train_datasets: 학습 데이터셋
			eval_datasets: 평가 데이터셋
			q_max_length: 질문 최대 길이
			p_max_length: 문서 최대 길이
			warmup_rate: warmup 비율
	"""
	def __init__(self,
				 q_encoder, p_encoder, tokenizer, batch_size=32, epochs=3, lr=2e-5,
				 train_datasets=None, eval_datasets=None, q_max_length=50, p_max_length=300, warmup_rate=0.1):

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
		self.q_max_length = q_max_length
		self.p_max_length = p_max_length
		self.warmup_rate = warmup_rate


	def train(self):
		"""
			학습을 수행하는 Method
		"""
		q_seqs = self.tokenizer(self.train_datasets['question'],
							   padding="max_length",
							   truncation=True,
							   return_tensors="pt",
							   max_length=self.q_max_length)
		p_seqs = self.tokenizer(self.train_datasets['context'],
							   padding="max_length",
							   truncation=True,
							   return_tensors="pt",
							   max_length=self.p_max_length)

		train_dataset = TensorDataset(p_seqs['input_ids'], p_seqs['attention_mask'], p_seqs['token_type_ids'],
									  q_seqs['input_ids'], q_seqs['attention_mask'], q_seqs['token_type_ids'])

		train_sampler = RandomSampler(train_dataset)
		train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=self.batch_size)

		no_decay = ['bias', 'LayerNorm.weight']
		self.optimizer_grouped_parameters = [
					{'params': [p for n, p in self.p_encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
					{'params': [p for n, p in self.p_encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
					{'params': [p for n, p in self.q_encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
					{'params': [p for n, p in self.q_encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
		]
		self.optimizer = AdamW(self.optimizer_grouped_parameters, lr=self.lr, eps=1e-8)
		self.t_total = len(train_dataloader) * self.epochs
		self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
														 num_warmup_steps=int(self.t_total*self.warmup_rate),
														 num_training_steps=self.t_total)

		# Training Step
		global_step = 0
		self.p_encoder.zero_grad()
		self.q_encoder.zero_grad()
		torch.cuda.empty_cache()

		train_iterator = trange(int(self.epochs), desc="Epoch")

		for _ in train_iterator:
			epoch_iterator = tqdm(train_dataloader, desc="Iteration")

			for step, batch in enumerate(epoch_iterator):
				self.p_encoder.train()
				self.q_encoder.train()

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

				loss = torch.nn.functional.nll_loss(sim_score, targets)
				epoch_iterator.set_postfix_str(f"Loss = {loss.detach():.4f}, lr = {self.optimizer.param_groups[0]['lr']:.4e}")

				loss.backward()
				self.optimizer.step()
				self.scheduler.step()
				self.q_encoder.zero_grad()
				self.p_encoder.zero_grad()

				global_step += 1

				torch.cuda.empty_cache()

	# TODO: Validation Dataset 기준 Evaluate
	def eval(self, valid_dataset, query):
		pass
		# _, p_embs = self.__setup_for_eval__(valid_dataset)
		# with torch.no_grad():
		# 	self.q_encoder.eval()
		#
		# 	q_seqs = self.tokenizer([query],
		# 							padding="max_length",
		# 							truncation=True,
		# 							return_tensors="pt",
		# 							max_length=self.max_length).to("cuda:0")
		# 	q_emb = self.q_encoder(**q_seqs).to('cpu')
		#
		# dot_prod_score = torch.matmul(q_emb, torch.transpose(p_embs, 0, 1))
		# rank = torch.argsort(dot_prod_score, dim=1, descending=True).squeeze()


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