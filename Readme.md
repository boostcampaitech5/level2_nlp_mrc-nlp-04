# 🗓️ 개발 기간

23.06.07 - 23.06.22(총 16일)

# 📄 프로젝트 소개

- 지문이 주어지지 않고, 사전에 구축되어 있는 문서 내에서 질문에 대한 정확한 답변을 하는 모델을 만드는 것을 목표로 한다.
- 따라서, 질문에 대한 정답이 포함되어 있는 문서를 찾기 위한 Retrieval Model과 찾은 문서 내에서 질문에 대한 답변을 찾기 위한 Reader Model로 2-Stage로 문제를 해결한다.

# 💽 사용 데이터셋

- Train Data : 3,952개(train) / 240개(validation)
Test Data : 240개(public) / 360개(private) 로 데이터가 구성되어 있다.
- Train Data는 `id(질문의 고유 id)`, `question(질문)`, `context(답변이 포함된 문서)`, `answers(답변에 대한 정보)`, `document_id(문서의 고유id)`, `title(문서의 제목)` 컬럼으로 구성되어 있고, 
Test Data는 `id`, `question` 만 공개되어 있다.

# 📋평가 지표

- **Exact Match(EM)** : 모델의 예측과 실제 정답이 정확하게 일치하는 비율에 대한 점수이다. 특수문자 등을 제외하고 정확하게 일치하면 1점, 아니라면 0점을 부여함으로써 해당 지표를 측정한다.
- **F1 Score** : EM과는 다르게 부분 점수를 측정한다. 정확히 같은 위치가 아니더라도 겹치는 단어가 있다면 부분 점수를 받을 수 있다.

# 👨‍👨‍👧‍👧 멤버 구성 및 역할

| [곽민석](https://github.com/kms7530) | [이인균](https://github.com/lig96) | [임하림](https://github.com/halimx2) | [최휘민](https://github.com/ChoiHwimin) | [황윤기](https://github.com/dbsrlskfdk) |
| --- | --- | --- | --- | --- |
| <img src="https://avatars.githubusercontent.com/u/6489395" width="140px" height="140px" title="Minseok Kwak" /> | <img src="https://avatars.githubusercontent.com/u/126560547" width="140px" height="140px" title="Ingyun Lee" /> | <img src="https://ca.slack-edge.com/T03KVA8PQDC-U04RK3E8L3D-ebbce77c3928-512" width="140px" height="140px" title="ChoiHwimin" /> | <img src="https://avatars.githubusercontent.com/u/102031218?v=4" width="140px" height="140px" title="이름" /> | <img src="https://avatars.githubusercontent.com/u/4418651?v=4" width="140px" height="140px" title="yungi" /> |
- **곽민석**
    - Reader 모델 데이터 증강, Elastic Search 구현 및 적용
- **이인균**
    - Retriever 모델(DPR) 구현 및 개선, Reader 모델 개선
- **임하림**
    - Retriever 모델(BM25) 구현, Context 전처리
- **최휘민**
    - Reader 모델 개선, Question Generation 구현,  Ensemble 구현
- **황윤기**
    - Retriever 모델(DPR) 구현, Re-Rank 구현

# ⚒️ 기능 및 사용 모델

## Retrieval 모델

- 수 만개에 달하는 위키피디아 문서 중에서 주어진 질문에 대한 정답이 적혀있는 문서를 탐색한다.
- 글의 임베딩 정보를 활용하는 `DPR` 모델과 글의 표면적인 단어의 일치 여부를 활용하는 `BM25` 모델을 이용한다. `BM-25` 로 추려진 답변을 `DPR` 모델을 이용해 Re-Rank 한다.

## Reader 모델

- Retrieval 모델을 통해 선택된 문서의 내용 속에서 질문에 대한 답변을 찾는다.
- `klue/roberta-large` 모델을 하이퍼파라미터 튜닝하여 사용한다.

# 🏗️ 프로젝트 구조

```bash
├── Trainer_DPR_Example.ipynb
├── arguments.py       
├── dpr
│   ├── __init__.py
│   ├── cls_Encoder.py 
│   └── trainer_DPR.py 
├── dpr_retrieval.py   
├── inference.py       
├── main.py            
├── preprocessing
│   └── dataset_preprocessing.py
├── reader.py
├── retrieval.py
├── train.py
├── trainer_qa.py
├── utils.py
├── utils_qa.py
├── Readme.md
└── github_utils
    └── markdown_to_notion.py
```

# 🔗 링크

- [Warp-up report](assets/docs/NLP_04_Wrap-Up_Report_MRC.pdf)
