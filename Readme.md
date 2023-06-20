## 실행 방법
디렉토리 변경  
`cd /opt/ml/level2_nlp_mrc-nlp-04`

train.py의 학습  
기본 model_name_or_path을 사용 후 모델을 output_dir에 저장합니다.  
`python main.py --mode train --output_dir ../output/models/train_dataset --do_train`  

train.py의 학습과 MRC 모델의 valid 평가  
기본 model_name_or_path을 사용 후 모델과 예측을 output_dir에 저장합니다.  
예측이 모델과 섞이기 때문에 사용을 비권장합니다.  
`python main.py --mode train --output_dir ../output/models/train_dataset --do_train --do_eval`

train.py의 MRC 모델의 valid 평가  
명시된 model_name_or_path을 사용 후 예측을 output_dir에 저장합니다.  
예측이 모델과 별도의 폴더에 저장되기 때문에 이 명령어를 사용하는 것을 권장합니다.  
`python main.py --mode train --output_dir ../output/prediction_valid_dataset --model_name_or_path ../output/models/train_dataset --do_eval`

inference.py의 ODQA 모델의 valid 평가  
명시된 dataset과 명시된 model_name_or_path를 사용 후 예측을 output_dir에 저장합니다.  
기본 사용되는 retriever model은 bm25입니다. 변경하시려면 --inference_mode {bm25, base, dpr}을 입력하세요.  
`python main.py --mode inference --output_dir ../output/prediction_valid_dataset/ --model_name_or_path ../output/models/train_dataset/ --dataset_name ../input/data/train_dataset/ --do_eval`

inference.py의 ODQA 모델의 test 평가  
명시된 dataset과 명시된 model_name_or_path를 사용 후 예측을 output_dir에 저장합니다.  
기본 사용되는 retriever model은 bm25입니다. 변경하시려면 --inference_mode {bm25, base, dpr}을 입력하세요.  
`python main.py --mode inference --output_dir ../output/prediction_test_dataset/ --model_name_or_path ../output/models/train_dataset/ --dataset_name ../input/data/test_dataset/  --do_predict`

output_dir을 overwrite하고 싶다면 다음 코드를 추가하세요.  
`--overwrite_output_dir `  