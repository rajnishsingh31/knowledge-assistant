# Retrieval-Augmented Generation (RAG) concepts:

## Chunking: 
Splitting large documents into smaller, manageable text segments for optimal embedding and search.

## Embeddings:
High-dimensional vector representations capturing the semantic meaning of text chunks.

## Vector DB:
Specialized database designed to store, index, and query vector embeddings efficiently.

## Retriever:
Component that searches the Vector DB to extract the most relevant chunks for a user query.

## Prompt construction:
Assembling retrieved contexts and the user's prompt into a cohesive LLM instruction.

## Citations:
Linking generated answers back to source chunks to ensure accuracy and auditability.

## Lost in the Middle:
The LLM performance drop when relevant context is buried in long input prompts.