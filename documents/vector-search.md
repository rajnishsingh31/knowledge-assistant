# vector search concepts:

Vector search retrieves semantically similar content by comparing embedding vectors.

## Cosine similarity: 
Measures the directional alignment (angle) between two vectors, ignoring magnitude.

## ANN (Approximate Nearest Neighbor):
Fast similarity search algorithms (e.g., HNSW) trading minimal accuracy for high speed.

## Exact search:
Computes distances against every vector (k-NN) for guaranteed 100% precision, but scales poorly.

## Metadata filtering: 
Restricts vector searches using structured attributes (e.g., date, category) pre- or post-query.

## Hybrid retrieval: 
Combines dense vector search (semantic) with sparse keyword search (lexical) for optimal relevance.

## RRF (Reciprocal Rank Fusion): 
An algorithm that merges and reranks search results from multiple distinct retrieval methods.