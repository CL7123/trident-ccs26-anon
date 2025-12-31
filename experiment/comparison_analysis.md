# FAISS HNSW vs TridentSearcher [CN]

## 1. [CN] ([CN] ef_search parameters)

### SIFTSMALL dataset
| [CN] | ef_search | Recall@10 | MRR@10 | query[CN](ms) | QPS |
|------|-----------|-----------|---------|--------------|-----|
| FAISS HNSW | 32 | 1.0000 | - | 0.009 | 114,661 |
| TridentSearcher | 32 | - | 0.9500 | 4.60 | 217 |

### LAION dataset
| [CN] | ef_search | Recall@10 | MRR@10 | query[CN](ms) | QPS |
|------|-----------|-----------|---------|--------------|-----|
| FAISS HNSW | 32 | 0.9876 | - | 0.121 | 8,269 |
| TridentSearcher | 32 | - | 0.9900 | 7.15 | 140 |

### TRIPCLICK dataset
| [CN] | ef_search | Recall@10 | MRR@10 | query[CN](ms) | QPS |
|------|-----------|-----------|---------|--------------|-----|
| FAISS HNSW | 36 | 0.4921 | - | 0.545 | 2,156 |
| TridentSearcher | 36 | - | 0.9333 | 20.26 | 49 |

### NFCORPUS dataset
| [CN] | ef_search | Recall@10 | MRR@10 | query[CN](ms) | QPS |
|------|-----------|-----------|---------|--------------|-----|
| FAISS HNSW | 32 | 0.1071 | - | 0.017 | 59,629 |
| TridentSearcher | 32 | - | 0.2687 | 4.22 | 237 |

## 2. [CN]

### 2.1 [CN]
- **TRIPCLICK**: TridentSearcher [CN] MRR@10 (0.9333) [CN] FAISS [CN] Recall@10 (0.4921)
- **NFCORPUS**: TridentSearcher [CN] MRR@10 (0.2687) [CN] FAISS [CN] Recall@10 (0.1071)
- **LAION**: [CN] (TridentSearcher: 0.99 vs FAISS: 0.9876)
- **SIFTSMALL**: FAISS [CN] (1.0 vs 0.95)

### 2.2 query[CN]
- FAISS HNSW [CN]query[CN] **10-500[CN]**
- TridentSearcher [CN]query[CN] 2.65-59.58ms [CN]
- FAISS HNSW [CN]query[CN] 0.009-1.8ms [CN]

### 2.3 dataset[CN]
- [CN]dataset（TRIPCLICK, NFCORPUS）[CN]，TridentSearcher [CN]
- [CN]vectordataset（SIFTSMALL, LAION）[CN]，[CN]

## 3. [CN]

### 3.1 [CN]
1. **[CN]**: TridentSearcher [CN]，[CN]path
2. **MRR vs Recall**: MRR [CN]result[CN]，[CN] Recall [CN]result[CN]
3. **indexbuild[CN]**: [CN]indexbuild[CN]parameters

### 3.2 [CN]
- TridentSearcher [CN]query[CN]：
  - [CN]
  - [CN]dataset[CN]
  - [CN]（[CN]）

## 4. [CN]

1. **TridentSearcher [CN]**：
   - [CN]
   - [CN]
   - [CN]dataset[CN] MRR

2. **FAISS HNSW [CN]**：
   - [CN]query[CN]
   - [CN]vector[CN]
   - [CN]

3. **[CN]**：
   - [CN]，[CN] FAISS HNSW
   - [CN]，[CN] TridentSearcher
   - [CN]，TridentSearcher [CN]