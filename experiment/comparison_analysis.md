# FAISS HNSW vs TridentSearcher performance[CN]analysis

## 1. recall[CN] (usage[CN] ef_search parameters)

### SIFTSMALL dataset
| method | ef_search | Recall@10 | MRR@10 | querytime(ms) | QPS |
|------|-----------|-----------|---------|--------------|-----|
| FAISS HNSW | 32 | 1.0000 | - | 0.009 | 114,661 |
| TridentSearcher | 32 | - | 0.9500 | 4.60 | 217 |

### LAION dataset
| method | ef_search | Recall@10 | MRR@10 | querytime(ms) | QPS |
|------|-----------|-----------|---------|--------------|-----|
| FAISS HNSW | 32 | 0.9876 | - | 0.121 | 8,269 |
| TridentSearcher | 32 | - | 0.9900 | 7.15 | 140 |

### TRIPCLICK dataset
| method | ef_search | Recall@10 | MRR@10 | querytime(ms) | QPS |
|------|-----------|-----------|---------|--------------|-----|
| FAISS HNSW | 36 | 0.4921 | - | 0.545 | 2,156 |
| TridentSearcher | 36 | - | 0.9333 | 20.26 | 49 |

### NFCORPUS dataset
| method | ef_search | Recall@10 | MRR@10 | querytime(ms) | QPS |
|------|-----------|-----------|---------|--------------|-----|
| FAISS HNSW | 32 | 0.1071 | - | 0.017 | 59,629 |
| TridentSearcher | 32 | - | 0.2687 | 4.22 | 237 |

## 2. [CN]key[CN]

### 2.1 recalldiff
- **TRIPCLICK**: TridentSearcher [CN] MRR@10 (0.9333) [CN] FAISS [CN] Recall@10 (0.4921)
- **NFCORPUS**: TridentSearcher [CN] MRR@10 (0.2687) [CN] FAISS [CN] Recall@10 (0.1071)
- **LAION**: [CN] (TridentSearcher: 0.99 vs FAISS: 0.9876)
- **SIFTSMALL**: FAISS [CN] (1.0 vs 0.95)

### 2.2 queryperformancediff
- FAISS HNSW [CN]queryvelocity[CN] **10-500[CN]**
- TridentSearcher [CN]querytime[CN] 2.65-59.58ms [CN]
- FAISS HNSW [CN]querytime[CN] 0.009-1.8ms [CN]

### 2.3 datasetattribute[CN]
- [CN]semanticssearchdataset(TRIPCLICK, NFCORPUS)on,TridentSearcher [CN]
- [CN]vectordataset(SIFTSMALL, LAION)on,[CN]

## 3. [CN]analysis

### 3.1 [CN]diff[CN]
1. **greedysearchstrategy**: TridentSearcher [CN]layerusage[CN]greedysearch,[CN]path
2. **MRR vs Recall**: MRR [CN]correctresult[CN]ranking,[CN] Recall [CN]correctresulttotal
3. **indexbuilddiff**: [CN]usage[CN]indexbuildstrategy[CN]parameters

### 3.2 performancetrade-off
- TridentSearcher [CN]queryvelocity[CN]:
  - [CN]distributedenvironment[CN]
  - [CN]dataseton[CN]searchquality
  - [CN]network[CN]overhead([CN]distributedscenein)

## 4. [CN]

1. **TridentSearcher [CN]**:
   - [CN]semanticssearchtaskon[CN]
   - distributed[CN]
   - [CN]dataseton[CN] MRR

2. **FAISS HNSW [CN]**:
   - [CN]querythroughput
   - [CN]vector[CN]taskon[CN]stable
   - [CN]late

3. **usagesuggestion**:
   - [CN]late[CN]scene,usage FAISS HNSW
   - [CN]distributedenvironment[CN]searchquality[CN]scene,[CN] TridentSearcher
   - [CN]semanticssearchtask,TridentSearcher [CN]yes[CN]select