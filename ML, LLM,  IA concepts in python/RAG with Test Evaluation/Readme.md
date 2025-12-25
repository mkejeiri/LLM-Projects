# RAG Evaluations

---

## 1- Curate a Test Set

Example questions set with the right context identified and reference answers provided.

---

## 2- Measure Retrieval

- **MRR** (Mean Reciprocal Rank)  
- **nDCG** (Normalized Discounted Cumulative Gain)  
- **Recall@K** and **Precision@K**

---

## 3- Measure Answers

e.g. Use **LLM-as-a-judge** to score provided answers against criteria like:
- accuracy
- completeness
- relevance

---

## Metric Definitions

- **MRR**  
  Average inverse rank of first hit; **1** if the first chunk always has relevant context.

- **nDCG**  
  Did relevant chunks get ranked higher up?

- **Recall@K**  
  Proportion of tests where relevant context was in the top **K** chunks.

- If you have multiple keywords to look for, **keyword coverage** is a similar recall metric.

- **Precision@K**  
  Proportion of the top **K** chunks that are relevant.
