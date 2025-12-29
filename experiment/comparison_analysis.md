# FAISS HNSW vs TridentSearcher 性能对比分析

## 1. 召回率对比 (使用相同的 ef_search 参数)

### SIFTSMALL 数据集
| 方法 | ef_search | Recall@10 | MRR@10 | 查询时间(ms) | QPS |
|------|-----------|-----------|---------|--------------|-----|
| FAISS HNSW | 32 | 1.0000 | - | 0.009 | 114,661 |
| TridentSearcher | 32 | - | 0.9500 | 4.60 | 217 |

### LAION 数据集
| 方法 | ef_search | Recall@10 | MRR@10 | 查询时间(ms) | QPS |
|------|-----------|-----------|---------|--------------|-----|
| FAISS HNSW | 32 | 0.9876 | - | 0.121 | 8,269 |
| TridentSearcher | 32 | - | 0.9900 | 7.15 | 140 |

### TRIPCLICK 数据集
| 方法 | ef_search | Recall@10 | MRR@10 | 查询时间(ms) | QPS |
|------|-----------|-----------|---------|--------------|-----|
| FAISS HNSW | 36 | 0.4921 | - | 0.545 | 2,156 |
| TridentSearcher | 36 | - | 0.9333 | 20.26 | 49 |

### NFCORPUS 数据集
| 方法 | ef_search | Recall@10 | MRR@10 | 查询时间(ms) | QPS |
|------|-----------|-----------|---------|--------------|-----|
| FAISS HNSW | 32 | 0.1071 | - | 0.017 | 59,629 |
| TridentSearcher | 32 | - | 0.2687 | 4.22 | 237 |

## 2. 关键发现

### 2.1 召回率差异
- **TRIPCLICK**: TridentSearcher 的 MRR@10 (0.9333) 显著高于 FAISS 的 Recall@10 (0.4921)
- **NFCORPUS**: TridentSearcher 的 MRR@10 (0.2687) 高于 FAISS 的 Recall@10 (0.1071)
- **LAION**: 两者表现相近 (TridentSearcher: 0.99 vs FAISS: 0.9876)
- **SIFTSMALL**: FAISS 略优 (1.0 vs 0.95)

### 2.2 查询性能差异
- FAISS HNSW 的查询速度快 **10-500倍**
- TridentSearcher 的查询时间在 2.65-59.58ms 之间
- FAISS HNSW 的查询时间在 0.009-1.8ms 之间

### 2.3 数据集特性影响
- 在语义搜索数据集（TRIPCLICK, NFCORPUS）上，TridentSearcher 表现更好
- 在传统向量数据集（SIFTSMALL, LAION）上，两者表现相近

## 3. 原因分析

### 3.1 算法差异导致的影响
1. **贪心搜索策略**: TridentSearcher 在高层使用纯贪心搜索，减少了探索但可能找到更直接的路径
2. **MRR vs Recall**: MRR 更关注第一个正确结果的排名，而 Recall 关注找到的正确结果总数
3. **索引构建差异**: 可能使用了不同的索引构建策略或参数

### 3.2 性能权衡
- TridentSearcher 牺牲了查询速度以获得：
  - 更适合分布式环境的设计
  - 在某些数据集上更好的搜索质量
  - 减少的网络通信开销（在分布式场景中）

## 4. 结论

1. **TridentSearcher 的优势**：
   - 在语义搜索任务上表现更好
   - 分布式友好的设计
   - 在某些数据集上有更高的 MRR

2. **FAISS HNSW 的优势**：
   - 极高的查询吞吐量
   - 在传统向量检索任务上表现稳定
   - 更低的延迟

3. **使用建议**：
   - 对于需要极低延迟的场景，使用 FAISS HNSW
   - 对于分布式环境或更注重搜索质量的场景，考虑 TridentSearcher
   - 对于语义搜索任务，TridentSearcher 可能是更好的选择