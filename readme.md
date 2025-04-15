# Medical Specialty Classification with LLMs

This repository contains the implementation of fine-tuning and evaluating three different Large Language Models (LLMs) for medical specialty classification. The models used are Google's Gemma-3-1B-IT, Meta's Llama 3, and DeepSeek-R1-Distill-Qwen-1.5B.

## Project Overview

The goal of this project is to classify medical case descriptions into appropriate medical specialties. We compare the performance of three different LLMs using three different approaches:
- Zero-shot learning
- Few-shot learning
- Fine-tuning

## Dataset

We use the [Medical Cases Classification Tutorial](https://huggingface.co/datasets/hpe-ai/medical-cases-classification-tutorial) dataset from Hugging Face, which contains medical case descriptions and their corresponding specialties. The dataset includes training, validation, and test splits.

## Models

1. **Google Gemma-3-1B-IT**: A 1 billion parameter model from Google's Gemma 3 family.
2. **Meta Llama 3 (8B)**: An 8 billion parameter model from Meta's Llama 3 family.
3. **DeepSeek-R1-Distill-Qwen-1.5B**: A 1.5 billion parameter distilled model from DeepSeek's R1 series.

## Implementation Details

The entire implementation is contained in a single Jupyter notebook: `main_all.ipynb`. The notebook includes:

1. **Data Loading and Preprocessing**:
   - Loading the dataset from Hugging Face
   - Preprocessing and formatting for different models
   - Creating few-shot examples

2. **Model Loading and Configuration**:
   - Loading models with appropriate quantization
   - Setting up LoRA configurations for fine-tuning

3. **Prompt Engineering**:
   - Custom prompt templates for each model
   - Zero-shot, few-shot, and fine-tuning prompts

4. **Fine-tuning Process**:
   - Parameter-efficient fine-tuning using LoRA
   - Training configuration and hyperparameters

5. **Evaluation**:
   - Metrics calculation (accuracy, F1 score, recall, precision)
   - Comparison between zero-shot, few-shot, and fine-tuned approaches

## Results

The notebook includes comprehensive evaluation results comparing:
- Performance across all three models
- Effectiveness of zero-shot, few-shot, and fine-tuning approaches
- Detailed metrics including accuracy, F1 score, recall, and precision

The results are presented in tables and visualized with charts for easy comparison.

## Requirements

The following libraries are required to run the notebook:
- transformers
- datasets
- peft
- trl
- torch
- pandas
- numpy
- matplotlib
- scikit-learn
- bitsandbytes
- accelerate

## Usage

1. Clone the repository
2. Install the required dependencies
3. Open and run the `main_all.ipynb` notebook
4. For models requiring authentication (like Gemma-3-1B-IT), you'll need to provide your Hugging Face token

## Note on Model Access

Some models used in this project (particularly Gemma-3-1B-IT and Llama 3) are gated models on Hugging Face. You'll need to:
1. Request access to these models on the Hugging Face Hub
2. Generate a Hugging Face token
3. Authenticate using the token in the notebook
