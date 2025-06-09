# embedding-eval

This repository contains the dataset and evaluation script for `CapRetrieval`.

##### Dataset

`CapRetrieval` follows the same format as retrieval datasets in MTEB. Data will be uploaded to Huggingface soon.

##### Evaluation Script

[run.py](run.py) is a general script to evaluate embeddings of various encoders on different datasets.

Results and embeddings will be saved at a new `evaluation` directory.


### Environment

Install pytorch, then `pip install requirements.txt`

Usage: `python run.py --help`


### Usage

Evaluate BM25 (only for language zh for now):

`python run.py --dataset CapRetrieval --topk 10 --mode bm25 --lang zh`

Evaluate BGE encoders using CLS pooling (default pooling):

`python run.py --dataset CapRetrieval --topk 10 --model BAAI/bge-base-zh-v1.5`

Evaluate GTE multilingual model using CLS pooling:

`python run.py --dataset CapRetrieval --topk 10 --model Alibaba-NLP/gte-multilingual-base`

Evaluate Conan-v1 encoder using default SentenceTransformers setup:

`python run.py --dataset CapRetrieval --topk 10 --model TencentBAC/Conan-embedding-v1 --pooling use_sentence_transformer`

Evaluate E5 encoders using mean pooling, with suggested prompt templates:

`python run.py --dataset CapRetrieval --topk 10 --model intfloat/multilingual-e5-base --pooling mean --max_len 512 --query_template "query: {text}" --candidate_template "passage: {text}"`

Evaluate GTE-Qwen encoders using last token pooling, with according prompt templates:
    
`python run.py --dataset CapRetrieval --topk 10 --model Alibaba-NLP/gte-Qwen2-7B-instruct --pooling last --query_template "Instruct: Given an image search query, retrieve relevant image captions\nQuery: {text}" --batch_size 8`

Evaluate Qwen3 embedding models using last token pooling, with according prompt templates:

`python run.py --dataset CapRetrieval --topk 10 --model Qwen/Qwen3-Embedding-8B --padding_side left --pooling last --query_template "Instruct: Given an image search query, retrieve relevant image captions\nQuery: {text}" --batch_size 8`
