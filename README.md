# CapRetrieval

This repository contains the dataset and evaluation script for **CapRetrieval**, introduced in the EMNLP 2025 Finding paper: [[Dense Retrievers Can Fail on Simple Queries: Revealing The Granularity Dilemma of Embeddings](https://arxiv.org/abs/2506.08592)].

The dataset is also available at [Huggingface](https://huggingface.co/datasets/lxucs/CapRetrieval); the English version is available at [CapRetrievalEn](https://huggingface.co/datasets/lxucs/CapRetrievalEn).

### Dataset

**CapRetrieval** evaluates the fine-grained embedding matching, tailored towards a practical image search scenario in Chinese via dense passage retrieval:
- Candidate passages are image captions, and queries are short phrases of entities or events reflected in captions.
- Overall, the dataset comprises seemingly simple queries and captions; however, text encoders are shown limitations resolving these cases.
- Evaluation results call for attention on embedding training strategies with different **granularity**.

CapRetrieval is in Chinese. The according English version is provided as **CapRetrievalEn**; queries and passages are translated automatically by GPT-4.1; all IDs and labels are kept the same as CapRetrieval. A few labels thus are not entirely accurate due to different language traits and expressions, but most labels should remain consistent.

#### Format

CapRetrieval follows the same retrieval task format as in [MTEB](https://huggingface.co/spaces/mteb/leaderboard), with relevance labels in $[0,1,2]$ for each pair.
Note that unlike prior datasets, we annotate full labels for each query-passage pair (1.3 million pairs), minimizing false negatives for more accurate evaluation.

A small amount of queries do not have any relevant captions; they are excluded in computation of retrieval metrics (e.g. nDCG), but can be useful for other analysis, e.g. in classification setting.

### Evaluation Script

[run.py](run.py) is a general script to evaluate embedding retrieval of various encers.

Results and embeddings will be saved under a new `evaluation` directory.


## Environment

Install `pytorch` according to your local environment, then `pip install -r requirements.txt`


## Usage

See options by `python run.py --help`

The script by default automatically uses the most appropriate device; you can also set `device_map` explicitly. Embeddings will be cached and re-used.

<details>
  <summary>Current Options</summary>

```
options:
  -h, --help            show this help message and exit
  --dataset DATASET     Dataset name
  --lang {en,zh}        Dataset language (for BM25)
  --mode {dense,bm25}   Search mode
  --model MODEL         HF model name or path
  --device_map DEVICE_MAP
                        Set model device map explicitly
  --max_len MAX_LEN     Max seq length
  --pooling {cls,mean,last,use_sentence_transformer}
                        Encoder pooling style
  --disable_normalization
                        Disable embedding normalization
  --query_template QUERY_TEMPLATE
                        Prompt template for query
  --candidate_template CANDIDATE_TEMPLATE
                        Prompt template for candidate
  --padding_side {left,right}
                        Tokenizer padding side
  --threshold THRESHOLD
                        Use results under distance threshold for evaluation
  --topk TOPK           Use top k results for evaluation
  --batch_size BATCH_SIZE
                        Eval batch size
  --result_path RESULT_PATH
                        Compute metrics of existing results directly
```
</details>

<details>
  <summary>Output Example</summary>

```
Search: 100%|██████████| 404/404 [00:00<00:00, 5315.29it/s]
Metrics for dataset CapRetrieval:
Query evaluation: reciprocal_rank @top10 = 88.70
Query evaluation: average_precision @top10 = 82.91
Query evaluation: ndcg @top10 = 78.86
Query evaluation: hit @top10 = 92.08
Query evaluation: query_precision = 38.22
Query evaluation: query_recall = 68.71
Query evaluation: pair_precision = 38.22
Query evaluation: pair_recall = 32.97
Query evaluation: query_f1 = 49.12
Query evaluation: query_f2 = 59.25
Query evaluation: pair_f1 = 35.40
Query evaluation: pair_f2 = 33.90

Saved 404 query results to evaluation/results.CapRetrieval.bge-base-zh-v1.5.top10.json
Saved report to evaluation/report.CapRetrieval.bge-base-zh-v1.5.top10.json
```
</details>

### Usage Examples

Evaluate BM25:

- `python run.py --dataset CapRetrieval --topk 10 --mode bm25 --lang zh`
- `python run.py --dataset CapRetrievalEn --topk 10 --mode bm25 --lang en`

Evaluate BGE encoders using CLS pooling (default pooling):

- `python run.py --dataset CapRetrieval --topk 10 --model BAAI/bge-base-zh-v1.5`
- `python run.py --dataset CapRetrievalEn --topk 10 --model BAAI/bge-base-en-v1.5`

Evaluate GTE multilingual model using CLS pooling:

- `python run.py --dataset CapRetrieval --topk 10 --model Alibaba-NLP/gte-multilingual-base`
- `python run.py --dataset CapRetrievalEn --topk 10 --model Alibaba-NLP/gte-multilingual-base`

Evaluate Conan-v1 encoder using default SentenceTransformers setup:

- `python run.py --dataset CapRetrieval --topk 10 --model TencentBAC/Conan-embedding-v1 --pooling use_sentence_transformer`

Evaluate E5 encoders using mean pooling, with suggested prompt templates:

- `python run.py --dataset CapRetrieval --topk 10 --model intfloat/multilingual-e5-base --pooling mean --max_len 512 --query_template "query: {text}" --candidate_template "passage: {text}"`

Evaluate E5-Mistral-7B using last token pooling, with according prompt templates:
    
- `python run.py --dataset CapRetrieval --topk 10 --model intfloat/e5-mistral-7b-instruct --pooling last --query_template "Instruct: Given an image search query, retrieve relevant image captions\nQuery: {text}" --batch_size 8`

Evaluate GTE-Qwen encoders using last token pooling, with according prompt templates:
    
- `python run.py --dataset CapRetrieval --topk 10 --model Alibaba-NLP/gte-Qwen2-7B-instruct --pooling last --query_template "Instruct: Given an image search query, retrieve relevant image captions\nQuery: {text}" --batch_size 8`

Evaluate Qwen3 embedding models using last token pooling, with according prompt templates:

- `python run.py --dataset CapRetrieval --topk 10 --model Qwen/Qwen3-Embedding-8B --padding_side left --pooling last --query_template "Instruct: Given an image search query, retrieve relevant image captions\nQuery: {text}" --batch_size 8`


## Evaluation on CapRetrieval

| Type     | Model                   | nDCG@10   |
|----------|-------------------------|-----------|
| **BM25** | Basic BM25              | 66.54     |
|          |                         |           |
| **0.1B** | bge-base-zh-v1.5        | 78.86     |
|          | gte-multilingual-base   | 79.67     |
|          | multilingual-e5-base    | 76.33     |
| **0.3B** | bge-large-zh-v1.5       | 79.15     |
|          | multilingual-e5-large   | 81.01     |
|          | Conan-embedding-v1      | 77.04     |
| **0.6B** | Qwen3-Embedding-0.6B    | 81.04     |
| **>1B**  | gte-Qwen2-1.5B-instruct | 77.35     |
|          | gte-Qwen2-7B-instruct   | **86.55** |
|          | e5-mistral-7b-instruct  | 76.40     |
|          | Qwen3-Embedding-8B      | 84.61     |
|          |                         |           |
| Trained  | Out-of-Domain           | 87.23     |
|          | In-Domain               | 91.83     |


The trained models (based on `bge-base-zh-v1.5`) are trained with queries by our data generation strategies described in the paper. The in-domain model can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1l2pvELMQPKjhAasNGaY7d14jMK0iCRhj).


## Evaluation on CapRetrievalEn

| Type     | Model                   | nDCG@10   |
|----------|-------------------------|-----------|
| **BM25** | Basic BM25              | 69.56     |
|          |                         |           |
| **0.1B** | bge-base-en-v1.5        | 67.26     |
|          | gte-multilingual-base   | 75.77     |
|          | multilingual-e5-base    | 74.53     |
| **0.3B** | bge-large-en-v1.5       | 61.94     |
|          | multilingual-e5-large   | 77.40     |
| **0.6B** | Qwen3-Embedding-0.6B    | 77.80     |
| **>1B**  | gte-Qwen2-1.5B-instruct | 72.04     |
|          | gte-Qwen2-7B-instruct   | **83.38** |
|          | e5-mistral-7b-instruct  | 77.07     |
|          | Qwen3-Embedding-8B      | 78.38     |


## License Agreement

The dataset and trained models are licensed under Apache 2.0. 


## Citation

```bibtex
@inproceedings{xu-etal-2025-dense,
    title = "Dense Retrievers Can Fail on Simple Queries: Revealing The Granularity Dilemma of Embeddings",
    author = "Xu, Liyan and Su, Zhenlin and Yu, Mo and Li, Jiangnan and Meng, Fandong and Zhou, Jie",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2025",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics"
}
```
