# korean-qa-paraphrase
Datasets and Training resources for paraphrasing queries in Korean question and answer datasets.

## [ Dataset ](https://huggingface.co/datasets/ohgnues/korean-qa-paraphrase)

Check about 930k Korean QA paraphrase data at this [link](https://huggingface.co/datasets/ohgnues/korean-qa-paraphrase)

* Aihub 금융, 법률 문서 기계독해 데이터
* Aihub 기술과학 문서 기계독해 데이터
* Aihub 뉴스 기사 기계독해 데이터
* Aihub 도서자료 기계독해 데이터
* Aihub 행정 문서 대상 기계독해 데이터


| source | question-1 | question-2 |
|:---:|:---:|:---:|
| 금융, 법률 문서 기계독해 데이터 | 베이비 부머 가운데 2011년에 성년이 되지 않은 자녀를 둔 비율은 얼마나 됐지 | 2011년도에 전체 베이비 부머 대비 얼마나 되는 비중이 미성년 자녀가 있어 |
| 행정 문서 대상 기계독해 데이터 | 음식물류폐기물로 인한 무엇을 알리는 것이 음식문화 발전을 위한 표어공모전의 목적이야 | 음식문화 개선을 위한 표어공모전은 음식물류폐기물로 인한 무엇을 알리는 데 목적이 있니 |


## Train

The training code is based on T5, and the evaluation metric used is ROUGE