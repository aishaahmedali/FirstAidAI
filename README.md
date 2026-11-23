# FirstAidAI
### AI-Powered First Aid Assistance Using Retrieval-Augmented Generation (RAG)

FirstAidAI is an AI-powered first-aid assistance platform developed as a Software Engineering academic project at Near East University.

The system combines text classification, semantic search, a vector database, and a generative language model to provide relevant first-aid information based on a trusted reference source.

## 1. Project Overview

Users submit first-aid questions through a web interface. The system processes each question through the following pipeline:

1. **Question Classification:** A fine-tuned DeBERTa-v3 model classifies the question into one of 11 first-aid categories.
2. **Information Retrieval:** ChromaDB retrieves relevant passages from the first-aid reference document using semantic similarity.
3. **Response Generation:** Google's Gemini model generates a response using the retrieved information.
4. **Web Interface:** A Flask application displays the generated response to the user.

## 2. System Architecture

```text
             User Question
                   |
                   v
          DeBERTa-v3 Classifier
                   |
                   v
          Predicted First-Aid Category
                   |
                   v
          Semantic Information Retrieval
                   |
                   v
             ChromaDB
                   |
                   v
          Retrieved Reference Passages
                   |
                   v
                 Gemini
                   |
                   v
          Generated First-Aid Response
                   |
                   v
            Flask Web Interface
