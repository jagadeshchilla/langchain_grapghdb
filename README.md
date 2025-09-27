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

## Files

- `requirements.txt` - Python dependencies
- `1-Q&A WITH GRAPHDB/` - Jupyter notebooks with experiments and examples
- `.env` - Environment variables (create this file)
