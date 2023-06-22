from transformers import BertPreTrainedModel, BertModel, RobertaPreTrainedModel, RobertaModel


class BertEncoder(BertPreTrainedModel):
	def __init__(self, config):
		super().__init__(config)
		self.bert = BertModel(config)
		self.init_weights()

	def forward(self,
				input_ids,
				attention_mask=None,
				token_type_ids=None):
		out = self.bert(input_ids,
						attention_mask=attention_mask,
						token_type_ids=token_type_ids)
		pooled_output = out[1]
		return pooled_output

class RoBertaEncoder(RobertaPreTrainedModel):
	def __init__(self, config):
		super().__init__(config)
		self.roberta = RobertaModel(config)
		self.init_weights()

	def forward(self,
				input_ids,
				attention_mask=None,
				token_type_ids=None):
		out = self.roberta(input_ids,
						attention_mask=attention_mask,
						token_type_ids=token_type_ids)
		pooled_output = out[1]
		return pooled_output