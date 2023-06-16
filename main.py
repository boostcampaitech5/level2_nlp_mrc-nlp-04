# from inference import main
from train import train
import argparse

from transformers import HfArgumentParser, TrainingArguments
from arguments import DataTrainingArguments, ModelArguments, CustomArguments

if __name__ == "__main__" :
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, CustomArguments))
    model_args, data_args, training_args, custom_args = parser.parse_args_into_dataclasses()
    
    if custom_args.mode == "train" :
        print("----- Train -----") 
        train(model_args, data_args, training_args)  

    if custom_args.mode == "inference" :
        print("----- Inference -----")
    #   # main()
    
      