# GraphDB LangChain

A question-answering application built with LangChain and Neo4j that enables natural language queries over graph databases.

## Overview

This project demonstrates how to build an intelligent question-answering system that can understand natural language questions and translate them into Cypher queries to extract information from a Neo4j graph database. The system uses LangChain's GraphCypherQAChain to bridge the gap between natural language and graph database queries.

## Features

- Natural language to Cypher query translation
- Movie database with actors, directors, and genres
- Few-shot prompting for improved query accuracy
- Integration with Groq's Gemma2-9b-It model
- Support for complex graph queries and relationships

## Dataset

The project uses a movie dataset that includes:
- Movies with titles, release dates, and IMDB ratings
- Actors and their relationships to movies
- Directors and their relationships to movies
- Genres and their relationships to movies

## Dependencies

- neo4j==5.14
- langchain
- langchain_community
- langchain_openai
- langchain_groq
- python_dotenv

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables in a `.env` file:
```
NEO4J_URI=your_neo4j_uri
NEO4J_USERNAME=your_neo4j_username
NEO4J_PASSWORD=your_neo4j_password
GROQ_API_KEY=your_groq_api_key
```

3. Start your Neo4j database instance

4. Run the Jupyter notebooks to load data and test queries

## Usage

The project includes three main notebooks:

- `experiments.ipynb` - Basic setup and initial experiments
- `promptstatergies.ipynb` - Advanced prompting strategies with few-shot examples
- `promptingstartegies.ipynb` - Additional prompting experiments

## Example Queries

The system can handle various types of questions:

- "Who was the director of the movie Casino?"
- "Which actors played in the movie Casino?"
- "How many movies has Tom Hanks acted in?"
- "Which actors have worked in movies from both comedy and action genres?"
- "Find the actor with the highest number of movies in the database"

## Key Concepts

### Graph Databases
Graph databases store data as nodes (entities) and relationships (connections between entities). This structure is ideal for representing complex relationships like social networks, recommendation systems, and knowledge graphs.

### Cypher Query Language
Cypher is Neo4j's query language designed specifically for graph databases. It uses ASCII art syntax to describe patterns in the graph, making it intuitive to read and write.

### LangChain GraphCypherQAChain
A specialized chain that combines natural language understanding with graph query generation. It automatically translates user questions into Cypher queries and executes them against the graph database.

### Few-Shot Prompting
A technique that provides the language model with examples of input-output pairs to improve its performance on specific tasks. This helps the model understand the expected format and style of responses.

### Natural Language to Query Translation
The process of converting human-readable questions into structured database queries. This involves understanding the user's intent and mapping it to the appropriate database operations.

## Architecture

The system uses:
- Neo4j as the graph database
- LangChain's GraphCypherQAChain for query translation
- Groq's Gemma2-9b-It model for natural language processing
- Few-shot prompting to improve query accuracy

## Finetuning

The project includes a finetuning module (`2-finetuning/`) that demonstrates how to customize language models using Lamini:

### Features
- Custom dataset preparation for Q&A tasks
- Model finetuning with configurable hyperparameters
- Secure API key management using environment variables
- Support for various optimization strategies

### Setup for Finetuning

1. Install finetuning dependencies:
```bash
cd 2-finetuning
pip install -r requirements.txt
```

2. Set up your Lamini API key as an environment variable:
```bash
# Windows
set LAMINI_API_KEY=your_lamini_api_key_here

# Linux/Mac
export LAMINI_API_KEY=your_lamini_api_key_here
```

3. Run the finetuning script:
```bash
python finetune.py
```

### Security Note
The finetuning module uses environment variables to securely handle API keys, preventing accidental exposure in version control.

## Advanced ML Concepts

### Finetuning
Finetuning is the process of adapting a pre-trained language model to perform better on specific tasks or domains. Instead of training a model from scratch, finetuning takes an existing model (like GPT, BERT, or Llama) and continues training it on task-specific data with a lower learning rate.

**Benefits:**
- Faster training compared to training from scratch
- Better performance on specific tasks
- Requires less computational resources
- Leverages pre-existing knowledge

### LoRA (Low-Rank Adaptation)
LoRA is an efficient finetuning technique that reduces the number of trainable parameters by decomposing weight updates into low-rank matrices. Instead of updating all model parameters, LoRA only trains small adapter layers.

**Key Advantages:**
- Dramatically reduces memory requirements
- Faster training and inference
- Multiple task-specific adapters can be stored and switched
- Maintains model performance while being resource-efficient

**How it works:**
- Original weights remain frozen
- Low-rank matrices are added to specific layers
- Only these small matrices are trained
- During inference, adapters are merged with original weights

### QLoRA (Quantized LoRA)
QLoRA combines quantization with LoRA for even more efficient finetuning. It quantizes the base model to 4-bit precision while using LoRA adapters for training.

**Benefits:**
- Extremely memory-efficient (can run on consumer GPUs)
- Maintains model quality
- Enables finetuning of large models on limited hardware
- Uses 4-bit NormalFloat (NF4) quantization

**Technical Details:**
- Base model quantized to 4-bit
- LoRA adapters trained in 16-bit precision
- Double quantization to reduce memory overhead
- Paged optimizers for memory management

### Quantization
Quantization reduces the precision of model weights and activations to decrease memory usage and increase inference speed.

**Types of Quantization:**

1. **Post-Training Quantization (PTQ)**
   - Applied after model training
   - Converts FP32 weights to INT8/INT4
   - Minimal accuracy loss
   - No retraining required

2. **Quantization-Aware Training (QAT)**
   - Simulates quantization during training
   - Better accuracy preservation
   - Requires retraining
   - More complex implementation

3. **Dynamic Quantization**
   - Weights quantized, activations computed in FP32
   - Good balance of speed and accuracy
   - Runtime quantization of activations

4. **Static Quantization**
   - Both weights and activations quantized
   - Requires calibration data
   - Maximum performance gains

**Quantization Levels:**
- **FP32**: Full precision (32-bit floating point)
- **FP16**: Half precision (16-bit floating point)
- **INT8**: 8-bit integer
- **INT4**: 4-bit integer (most aggressive)

### Memory and Performance Trade-offs

| Method | Memory Usage | Training Speed | Inference Speed | Accuracy |
|--------|-------------|----------------|-----------------|----------|
| Full Finetuning | High | Slow | Medium | Best |
| LoRA | Medium | Fast | Fast | Good |
| QLoRA | Low | Medium | Fast | Good |
| Quantization | Low | N/A | Fastest | Good |

### When to Use Each Method

**Full Finetuning:**
- When you have abundant computational resources
- For domain-specific applications requiring maximum accuracy
- When working with smaller models

**LoRA:**
- When you need to adapt models for multiple tasks
- For efficient finetuning with good performance
- When memory is limited but you want to maintain quality

**QLoRA:**
- When working with very large models (7B+ parameters)
- For consumer hardware finetuning
- When memory is severely constrained

**Quantization:**
- For production deployment
- When inference speed is critical
- For edge device deployment
- When storage space is limited


