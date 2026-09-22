# FirstAidAI

An AI-based first-aid assistance system that combines text classification, Retrieval-Augmented Generation (RAG), ChromaDB, and Gemini to provide category-aware first-aid guidance based on a trusted first-aid reference source.

## Overview

FirstAidAI is designed to help users obtain relevant first-aid information by entering questions in natural language.

The system first classifies the user's question into one of 11 first-aid categories using a trained DeBERTa-v3 model. It then retrieves relevant information from a ChromaDB vector database containing content derived from the St. John Ambulance Canada First Aid Reference Guide. Finally, Gemini generates a clear response using the retrieved information.

The system is implemented as a Flask web application.

## System Architecture

```text
User Question
      |
      v
DeBERTa-v3 Classifier
      |
      v
First-Aid Category + Confidence
      |
      v
ChromaDB Vector Database
      |
      v
Relevant First-Aid Information
      |
      v
Gemini
      |
      v
Final First-Aid Response
      |
      v
Flask Web Interface