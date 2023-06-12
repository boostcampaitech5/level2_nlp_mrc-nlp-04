from torch import nn
from transformers import (
	AutoConfig,
	AutoModelForQuestionAnswering,
)

class BaseQAModel(nn.Module):
	def __init__(self, model_name, config=None):
		super().__init__()
		self.config = config if config else AutoConfig.from_pretrained(model_name)
		self.config.update({"output_hidden_states": True})
		self.plm = AutoModelForQuestionAnswering.from_pretrained(model_name, config=self.config)

	def forward(self, **kwargs):
		return self.plm(**kwargs)