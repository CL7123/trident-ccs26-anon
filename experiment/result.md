# Trident Search Results
**Test Time**: 2025-07-26 17:00:51
**Dataset**: tripclick

## Configuration Information
- Vector Dimension: 768
- Document Count: 1,523,87
- HNSW Parameters: M=128, efConstruction=160
- Default efSearch: 36
- Query Count: 100
- k: 10

## Data Information
- Query Vectors: (1175, 768)
- Ground Truth: (1175, 10)

## Test Results
| efSearch | MRR@10 | Average Latency (ms) |
|----------|-----------|---------------|
| 18 | 0.9333 | 11.71 |
| 36 | 0.9333 | 20.26 |
| 72 | 0.9333 | 34.97 |
| 144 | 0.9333 | 59.58 |

## Summary
- Test Completion Time: 2025-07-26 17:01:23
- Dataset: tripclick
- Index Location: /home/anonymous/Test-Trident/dataset/tripclick


**Test Time**: 2025-07-26 17:24:58
**Dataset**: laion

## Configuration Information
- Vector Dimension: 512
- Document Count: 100,000
- HNSW Parameters: M=64, efConstruction=80
- Default efSearch: 32
- Query Count: 100
- k: 10

## Data Information
- Query Vectors: (1000, 512)
- Ground Truth: (1000, 10)

## Test Results
| efSearch | MRR@10 | Average Latency (ms) |
|----------|-----------|---------------|
| 16 | 0.9800 | 5.15 |
| 32 | 0.9900 | 7.15 |
| 64 | 1.0000 | 11.58 |
| 128 | 1.0000 | 18.74 |

## Summary
- Test Completion Time: 2025-07-26 17:25:10
- Dataset: laion
- Index Location: /home/anonymous/Test-Trident/dataset/laion



**Test Time**: 2025-07-26 18:01:40
**Dataset**: siftsmall

## Configuration Information
- Vector Dimension: 128
- Document Count: 10,000
- HNSW Parameters: M=64, efConstruction=80
- Default efSearch: 32
- Query Count: 100
- k: 10

## Data Information
- Query Vectors: (100, 128)
- Ground Truth: (100, 100)

## Test Results
| efSearch | MRR@10 | Average Latency (ms) |
|----------|-----------|---------------|
| 16 | 0.9100 | 3.31 |
| 32 | 0.9500 | 4.60 |
| 64 | 0.9800 | 6.51 |
| 128 | 0.9800 | 9.98 |

## Summary
- Test Completion Time: 2025-07-26 18:01:43
- Dataset: siftsmall
- Index Location: /home/anonymous/Test-Trident/dataset/siftsmall



# Trident Search Results
**Test Time**: 2025-07-26 20:22:22
**Dataset**: nfcorpus

## Configuration Information
- Vector Dimension: 768
- Document Count: 3,633
- HNSW Parameters: M=32, efConstruction=80
- Default efSearch: 32
- Query Count: 100
- k: 10

## Data Information
- Query Vectors: (323, 768)
- Ground Truth: (323, 100)

## Test Results
| efSearch | MRR@10 | Average Latency (ms) |
|----------|-----------|---------------|
| 16 | 0.2582 | 2.65 |
| 32 | 0.2687 | 4.22 |
| 64 | 0.2682 | 6.40 |
| 128 | 0.2882 | 10.56 |

## Summary
- Test Completion Time: 2025-07-26 20:22:25
- Dataset: nfcorpus
- Index Location: /home/anonymous/Test-Trident/dataset/nfcorpus 







---

# Test Result Report - laion

**Generation Time**: 2025-07-26 17:52:10
**Dataset**: laion
**Number of Queries**: 10

## Detailed Query Results

| Query No. | Node ID | Phase 1 (VDPF) | Phase 2 (e/f) | Phase 3 (Exchange) | Phase 4 (Reconstruction) | Total Time | Cosine Similarity |
|---------|--------|--------------|-------------|--------------|--------------|--------|-----------|
| 1 | 2414 | 6.47s | 1.21s | 4.85s | 3.06s | 15.81s | 1.000000 |
| 2 | 74125 | 6.98s | 1.20s | 5.37s | 3.40s | 17.11s | 1.000000 |
| 3 | 80693 | 7.23s | 1.15s | 5.05s | 3.38s | 17.15s | 1.000000 |
| 4 | 73351 | 7.19s | 1.14s | 5.11s | 3.19s | 16.75s | 1.000000 |
| 5 | 39973 | 7.19s | 1.20s | 5.29s | 3.20s | 17.00s | 1.000000 |
| 6 | 37952 | 6.70s | 1.17s | 5.35s | 3.34s | 16.75s | 1.000000 |
| 7 | 35116 | 6.74s | 1.22s | 5.33s | 3.44s | 16.81s | 1.000000 |
| 8 | 78475 | 7.14s | 1.20s | 5.43s | 3.49s | 17.55s | 1.000000 |
| 9 | 92101 | 7.06s | 1.16s | 5.32s | 3.01s | 16.84s | 1.000000 |
| 10 | 80623 | 7.35s | 1.24s | 5.59s | 3.11s | 17.37s | 1.000000 |

## Average Performance Statistics

- **Phase 1 (Multi-process VDPF Evaluation)**: 7.01 seconds
- **Phase 2 (e/f Calculation)**: 1.19 seconds
- **Phase 3 (Data Exchange)**: 5.27 seconds
- **Phase 4 (Reconstruction)**: 3.26 seconds
- **Server Internal Total**: 16.91 seconds
- **Average Cosine Similarity**: 1.000000

## Performance Analysis

### Time Distribution

- Phase 1 (VDPF Evaluation): 41.4%
- Phase 2 (e/f Calculation): 7.0%
- Phase 3 (Data Exchange): 31.1%
- Phase 4 (Reconstruction): 19.3%

### Throughput
- Average Query Time: 16.91 seconds
- Theoretical Throughput: 0.06 queries/second


laion Encrypted Query
=== Average Performance Statistics (10 Successful Queries) ===
  Phase 1 (Multi-process VDPF Evaluation): 7.01 seconds
  Phase 2 (e/f Calculation): 1.19 seconds
  Phase 3 (Data Exchange): 5.27 seconds
  Phase 4 (Reconstruction): 3.26 seconds
  Server Internal Total: 16.91 seconds
  Average Cosine Similarity: 1.000000


---

# Test Result Report - laion

**Generation Time**: 2025-07-26 17:58:38
**Dataset**: laion
**Number of Queries**: 10

## Detailed Query Results

| Query No. | Node ID | Phase 1 (VDPF) | Phase 2 (e/f) | Phase 3 (Exchange) | Phase 4 (Reconstruction) | Total Time | Cosine Similarity |
|---------|--------|--------------|-------------|--------------|--------------|--------|-----------|
| 1 | 38551 | 5.98s | 1.14s | 4.98s | 3.47s | 15.69s | 1.000000 |
| 2 | 66998 | 6.88s | 1.20s | 5.27s | 3.43s | 16.94s | 1.000000 |
| 3 | 26665 | 6.23s | 1.32s | 5.37s | 3.42s | 16.59s | 1.000000 |
| 4 | 84090 | 6.26s | 1.20s | 5.02s | 3.11s | 15.71s | 1.000000 |
| 5 | 11941 | 6.88s | 1.26s | 5.22s | 3.42s | 16.97s | 1.000000 |
| 6 | 4109 | 6.24s | 1.16s | 5.23s | 2.97s | 15.82s | 1.000000 |
| 7 | 90584 | 6.67s | 1.18s | 5.25s | 3.55s | 16.74s | 1.000000 |
| 8 | 2012 | 6.04s | 1.20s | 5.50s | 3.37s | 16.27s | 1.000000 |
| 9 | 73818 | 7.05s | 1.20s | 5.45s | 3.25s | 17.17s | 1.000000 |
| 10 | 62588 | 6.32s | 1.13s | 5.57s | 3.34s | 16.47s | 1.000000 |

## Average Performance Statistics

- **Phase 1 (Multi-process VDPF Evaluation)**: 6.45 seconds
- **Phase 2 (e/f Calculation)**: 1.20 seconds
- **Phase 3 (Data Exchange)**: 5.29 seconds
- **Phase 4 (Reconstruction)**: 3.33 seconds
- **Server Internal Total**: 16.44 seconds
- **Average Cosine Similarity**: 1.000000

## Performance Analysis

### Time Distribution

- Phase 1 (VDPF Evaluation): 39.3%
- Phase 2 (e/f Calculation): 7.3%
- Phase 3 (Data Exchange): 32.2%
- Phase 4 (Reconstruction): 20.3%

### Throughput
- Average Query Time: 16.44 seconds
- Theoretical Throughput: 0.06 queries/second


---

# Test Result Report - siftsmall

**Generation Time**: 2025-07-26 18:03:48
**Dataset**: siftsmall
**Number of Queries**: 10

## Detailed Query Results

| Query No. | Node ID | Phase 1 (VDPF) | Phase 2 (e/f) | Phase 3 (Exchange) | Phase 4 (Reconstruction) | Total Time | Cosine Similarity |
|---------|--------|--------------|-------------|--------------|--------------|--------|-----------|
| 1 | 1036 | 0.84s | 0.06s | 0.22s | 0.06s | 1.25s | 1.000000 |
| 2 | 9147 | 0.82s | 0.06s | 0.16s | 0.05s | 1.16s | 1.000000 |
| 3 | 5180 | 0.81s | 0.06s | 0.18s | 0.07s | 1.18s | 1.000000 |
| 4 | 2220 | 0.84s | 0.06s | 0.21s | 0.05s | 1.23s | 1.000000 |
| 5 | 9099 | 0.80s | 0.07s | 0.18s | 0.06s | 1.17s | 1.000000 |
| 6 | 6586 | 0.85s | 0.05s | 0.17s | 0.05s | 1.19s | 1.000000 |
| 7 | 2072 | 0.82s | 0.05s | 0.17s | 0.05s | 1.16s | 1.000000 |
| 8 | 2945 | 0.85s | 0.07s | 0.18s | 0.05s | 1.22s | 1.000000 |
| 9 | 4030 | 0.80s | 0.05s | 0.17s | 0.06s | 1.16s | 1.000000 |
| 10 | 2558 | 0.81s | 0.05s | 0.18s | 0.05s | 1.15s | 1.000000 |

## Average Performance Statistics

- **Phase 1 (Multi-process VDPF Evaluation)**: 0.82 seconds
- **Phase 2 (e/f Calculation)**: 0.06 seconds
- **Phase 3 (Data Exchange)**: 0.18 seconds
- **Phase 4 (Reconstruction)**: 0.06 seconds
- **Server Internal Total**: 1.19 seconds
- **Average Cosine Similarity**: 1.000000

## Performance Analysis

### Time Distribution

- Phase 1 (VDPF Evaluation): 69.4%
- Phase 2 (e/f Calculation): 5.0%
- Phase 3 (Data Exchange): 15.3%
- Phase 4 (Reconstruction): 4.6%

### Throughput
- Average Query Time: 1.19 seconds
- Theoretical Throughput: 0.84 queries/second


---

---

# Test Result Report - tripclick

**Generation Time**: 2025-07-26 20:19:17
**Dataset**: tripclick
**Number of Queries**: 10

## Detailed Query Results

| Query No. | Node ID | Phase 1 (VDPF) | Phase 2 (e/f) | Phase 3 (Exchange) | Phase 4 (Reconstruction) | Total Time | Cosine Similarity |
|---------|--------|--------------|-------------|--------------|--------------|--------|-----------|
| 1 | 109962 | 10.44s | 2.13s | 11.87s | 7.34s | 31.94s | 1.000000 |
| 2 | 145038 | 11.28s | 2.30s | 11.90s | 7.89s | 33.51s | 1.000000 |
| 3 | 136182 | 11.93s | 2.19s | 11.74s | 7.88s | 33.84s | 1.000000 |
| 4 | 112534 | 11.50s | 2.25s | 11.83s | 7.57s | 33.29s | 1.000000 |
| 5 | 110027 | 11.29s | 2.23s | 11.59s | 7.56s | 32.84s | 1.000000 |
| 6 | 113616 | 11.73s | 2.24s | 11.55s | 7.68s | 33.46s | 1.000000 |
| 7 | 83049 | 11.49s | 2.35s | 11.89s | 7.24s | 33.24s | 1.000000 |
| 8 | 88282 | 11.73s | 2.19s | 12.64s | 9.32s | 36.69s | 1.000000 |
| 9 | 123066 | 12.63s | 2.37s | 11.75s | 8.15s | 35.18s | 1.000000 |
| 10 | 48803 | 12.13s | 2.30s | 11.81s | 7.65s | 34.15s | 1.000000 |

## Average Performance Statistics

- **Phase 1 (Multi-process VDPF Evaluation)**: 11.62 seconds
- **Phase 2 (e/f Calculation)**: 2.26 seconds
- **Phase 3 (Data Exchange)**: 11.86 seconds
- **Phase 4 (Reconstruction)**: 7.83 seconds
- **Server Internal Total**: 33.81 seconds
- **Average Cosine Similarity**: 1.000000

## Performance Analysis

### Time Distribution

- Phase 1 (VDPF Evaluation): 34.4%
- Phase 2 (e/f Calculation): 6.7%
- Phase 3 (Data Exchange): 35.1%
- Phase 4 (Reconstruction): 23.2%

### Throughput
- Average Query Time: 33.81 seconds
- Theoretical Throughput: 0.03 queries/second

---

# Test Result Report - nfcorpus

**Generation Time**: 2025-07-26 20:39:42
**Dataset**: nfcorpus
**Number of Queries**: 10

## Detailed Query Results

| Query No. | Node ID | Phase 1 (VDPF) | Phase 2 (e/f) | Phase 3 (Exchange) | Phase 4 (Reconstruction) | Total Time | Cosine Similarity |
|---------|--------|--------------|-------------|--------------|--------------|--------|-----------|
| 1 | 1522 | 0.67s | 0.04s | 0.30s | 0.17s | 1.24s | 1.000000 |
| 2 | 2693 | 0.66s | 0.04s | 0.26s | 0.12s | 1.15s | 1.000000 |
| 3 | 33 | 0.63s | 0.04s | 0.25s | 0.12s | 1.10s | 1.000000 |
| 4 | 2970 | 0.62s | 0.04s | 0.24s | 0.14s | 1.11s | 1.000000 |
| 5 | 2961 | 0.58s | 0.05s | 0.25s | 0.13s | 1.08s | 1.000000 |
| 6 | 1543 | 0.58s | 0.04s | 0.25s | 0.13s | 1.07s | 1.000000 |
| 7 | 3570 | 0.57s | 0.04s | 0.26s | 0.16s | 1.10s | 1.000000 |
| 8 | 2993 | 0.61s | 0.04s | 0.24s | 0.14s | 1.10s | 1.000000 |
| 9 | 3589 | 0.60s | 0.04s | 0.27s | 0.13s | 1.11s | 1.000000 |
| 10 | 1399 | 0.58s | 0.04s | 0.22s | 0.17s | 1.08s | 1.000000 |

## Average Performance Statistics

- **Phase 1 (Multi-process VDPF Evaluation)**: 0.61 seconds
- **Phase 2 (e/f Calculation)**: 0.04 seconds
- **Phase 3 (Data Exchange)**: 0.25 seconds
- **Phase 4 (Reconstruction)**: 0.14 seconds
- **Server Internal Total**: 1.11 seconds
- **Average Cosine Similarity**: 1.000000

## Performance Analysis

### Time Distribution

- Phase 1 (VDPF Evaluation): 54.9%
- Phase 2 (e/f Calculation): 3.5%
- Phase 3 (Data Exchange): 22.7%
- Phase 4 (Reconstruction): 12.7%

### Throughput
- Average Query Time: 1.11 seconds
- Theoretical Throughput: 0.90 queries/second


---



---

# Test Result Report - siftsmall

**Generation Time**: 2025-07-27 13:00:54
**Dataset**: siftsmall
**Number of Queries**: 10

## Detailed Query Results

| Query No. | Node ID | Layer | Phase 1 (VDPF) | Phase 2 (e/f) | Phase 3 (Exchange) | Phase 4 (Reconstruction) | Total Time | Neighbor Match Rate |
|---------|--------|-----|--------------|-------------|--------------|--------------|--------|-----------|
| 1 | 5530 | 1 | 1.54s | 0.21s | 0.47s | 0.22s | 2.54s | 100.00% |
| 2 | 687 | 1 | 1.58s | 0.19s | 0.38s | 0.22s | 2.44s | 100.00% |
| 3 | 3996 | 0 | 1.70s | 0.21s | 0.37s | 0.21s | 2.56s | 100.00% |
| 4 | 4885 | 2 | 1.57s | 0.21s | 0.36s | 0.21s | 2.39s | 100.00% |
| 5 | 5379 | 1 | 1.63s | 0.20s | 0.35s | 0.20s | 2.45s | 100.00% |
| 6 | 6088 | 0 | 1.62s | 0.20s | 0.40s | 0.20s | 2.56s | 100.00% |
| 7 | 4044 | 1 | 1.65s | 0.20s | 0.36s | 0.22s | 2.49s | 100.00% |
| 8 | 1991 | 2 | 1.56s | 0.20s | 0.35s | 0.21s | 2.43s | 100.00% |
| 9 | 6068 | 0 | 1.60s | 0.19s | 0.35s | 0.21s | 2.42s | 100.00% |
| 10 | 9854 | 1 | 1.62s | 0.19s | 0.36s | 0.20s | 2.45s | 100.00% |

## Average Performance Statistics

- **Phase 1 (Multi-process VDPF Evaluation)**: 1.61 seconds
- **Phase 2 (e/f Calculation)**: 0.20 seconds
- **Phase 3 (Data Exchange)**: 0.37 seconds
- **Phase 4 (Reconstruction)**: 0.21 seconds
- **Server Internal Total**: 2.47 seconds
- **Average Neighbor Match Rate**: 100.00%

## Performance Analysis

### Time Distribution

- Phase 1 (VDPF Evaluation): 65.0%
- Phase 2 (e/f Calculation): 8.1%
- Phase 3 (Data Exchange): 15.1%
- Phase 4 (Reconstruction): 8.6%

### Throughput
- Average Query Time: 2.47 seconds
- Theoretical Throughput: 0.40 queries/second


---

# Test Result Report - laion

**Generation Time**: 2025-07-27 13:09:26
**Dataset**: laion
**Number of Queries**: 10

## Detailed Query Results

| Query No. | Node ID | Layer | Phase 1 (VDPF) | Phase 2 (e/f) | Phase 3 (Exchange) | Phase 4 (Reconstruction) | Total Time | Neighbor Match Rate |
|---------|--------|-----|--------------|-------------|--------------|--------------|--------|-----------|
| 1 | 49679 | 1 | 14.99s | 2.36s | 3.64s | 2.56s | 24.07s | 100.00% |
| 2 | 90937 | 2 | 15.88s | 2.40s | 4.01s | 2.61s | 25.10s | 100.00% |
| 3 | 47762 | 2 | 16.20s | 2.39s | 3.89s | 2.66s | 25.48s | 100.00% |
| 4 | 39040 | 0 | 16.53s | 2.35s | 3.76s | 2.53s | 25.72s | 100.00% |
| 5 | 40419 | 1 | 16.81s | 2.24s | 3.83s | 2.82s | 25.89s | 100.00% |
| 6 | 43395 | 2 | 16.52s | 2.33s | 3.93s | 2.50s | 25.62s | 100.00% |
| 7 | 14062 | 2 | 16.34s | 2.36s | 4.01s | 2.71s | 25.83s | 100.00% |
| 8 | 9688 | 2 | 15.53s | 2.30s | 3.90s | 2.59s | 24.58s | 100.00% |
| 9 | 1182 | 2 | 16.74s | 2.31s | 3.84s | 2.67s | 25.82s | 100.00% |
| 10 | 59281 | 0 | 17.27s | 2.27s | 3.74s | 2.52s | 26.13s | 100.00% |

## Average Performance Statistics

- **Phase 1 (Multi-process VDPF Evaluation)**: 16.28 seconds
- **Phase 2 (e/f Calculation)**: 2.33 seconds
- **Phase 3 (Data Exchange)**: 3.85 seconds
- **Phase 4 (Reconstruction)**: 2.62 seconds
- **Server Internal Total**: 25.43 seconds
- **Average Neighbor Match Rate**: 100.00%

## Performance Analysis

### Time Distribution

- Phase 1 (VDPF Evaluation): 64.0%
- Phase 2 (e/f Calculation): 9.2%
- Phase 3 (Data Exchange): 15.2%
- Phase 4 (Reconstruction): 10.3%

### Throughput
- Average Query Time: 25.43 seconds
- Theoretical Throughput: 0.04 queries/second


---

# Test Result Report - tripclick

**Generation Time**: 2025-07-27 13:25:12
**Dataset**: tripclick
**Number of Queries**: 10

## Detailed Query Results

| Query No. | Node ID | Layer | Phase 1 (VDPF) | Phase 2 (e/f) | Phase 3 (Exchange) | Phase 4 (Reconstruction) | Total Time | Neighbor Match Rate |
|---------|--------|-----|--------------|-------------|--------------|--------------|--------|-----------|
| 1 | 49561 | 1 | 27.08s | 4.51s | 14.74s | 10.13s | 57.01s | 100.00% |
| 2 | 105651 | 0 | 29.65s | 4.44s | 13.57s | 8.64s | 56.59s | 100.00% |
| 3 | 85369 | 1 | 30.21s | 4.45s | 13.39s | 8.98s | 58.23s | 100.00% |
| 4 | 123816 | 2 | 29.98s | 4.57s | 14.02s | 8.23s | 58.01s | 100.00% |
| 5 | 135194 | 1 | 30.16s | 4.59s | 13.21s | 7.77s | 56.48s | 100.00% |
| 6 | 71939 | 0 | 29.06s | 4.56s | 13.20s | 8.02s | 55.29s | 100.00% |
| 7 | 148133 | 0 | 30.33s | 4.67s | 12.77s | 7.93s | 56.01s | 100.00% |
| 8 | 24040 | 1 | 28.67s | 4.53s | 13.77s | 8.01s | 55.17s | 100.00% |
| 9 | 87391 | 2 | 28.80s | 4.66s | 13.05s | 8.15s | 55.46s | 100.00% |
| 10 | 80734 | 0 | 29.84s | 4.74s | 13.58s | 8.22s | 56.76s | 100.00% |

## Average Performance Statistics

- **Phase 1 (Multi-process VDPF Evaluation)**: 29.38 seconds
- **Phase 2 (e/f Calculation)**: 4.57 seconds
- **Phase 3 (Data Exchange)**: 13.53 seconds
- **Phase 4 (Reconstruction)**: 8.41 seconds
- **Server Internal Total**: 56.50 seconds
- **Average Neighbor Match Rate**: 100.00%

## Performance Analysis

### Time Distribution

- Phase 1 (VDPF Evaluation): 52.0%
- Phase 2 (e/f Calculation): 8.1%
- Phase 3 (Data Exchange): 23.9%
- Phase 4 (Reconstruction): 14.9%

### Throughput
- Average Query Time: 56.50 seconds
- Theoretical Throughput: 0.02 queries/second


---------

# Test Result Report - nfcorpus

**Generation Time**: 2025-07-27 13:32:57
**Dataset**: nfcorpus
**Number of Queries**: 10

## Detailed Query Results

| Query No. | Node ID | Layer | Phase 1 (VDPF) | Phase 2 (e/f) | Phase 3 (Exchange) | Phase 4 (Reconstruction) | Total Time | Neighbor Match Rate |
|---------|--------|-----|--------------|-------------|--------------|--------------|--------|-----------|
| 1 | 3532 | 0 | 0.85s | 0.06s | 0.14s | 0.03s | 1.14s | 100.00% |
| 2 | 517 | 2 | 0.79s | 0.06s | 0.12s | 0.03s | 1.08s | 100.00% |
| 3 | 2068 | 0 | 0.79s | 0.05s | 0.13s | 0.03s | 1.08s | 100.00% |
| 4 | 1298 | 1 | 0.85s | 0.06s | 0.13s | 0.03s | 1.13s | 100.00% |
| 5 | 83 | 1 | 0.77s | 0.07s | 0.13s | 0.03s | 1.07s | 100.00% |
| 6 | 2074 | 2 | 0.80s | 0.05s | 0.13s | 0.03s | 1.08s | 100.00% |
| 7 | 260 | 0 | 0.74s | 0.05s | 0.12s | 0.03s | 1.01s | 100.00% |
| 8 | 1213 | 2 | 0.76s | 0.05s | 0.13s | 0.02s | 1.03s | 100.00% |
| 9 | 2087 | 1 | 0.80s | 0.05s | 0.12s | 0.03s | 1.07s | 100.00% |
| 10 | 501 | 0 | 0.83s | 0.06s | 0.13s | 0.03s | 1.12s | 100.00% |

## Average Performance Statistics

- **Phase 1 (Multi-process VDPF Evaluation)**: 0.80 seconds
- **Phase 2 (e/f Calculation)**: 0.06 seconds
- **Phase 3 (Data Exchange)**: 0.13 seconds
- **Phase 4 (Reconstruction)**: 0.03 seconds
- **Server Internal Total**: 1.08 seconds
- **Average Neighbor Match Rate**: 100.00%

## Performance Analysis

### Time Distribution

- Phase 1 (VDPF Evaluation): 74.0%
- Phase 2 (e/f Calculation): 5.3%
- Phase 3 (Data Exchange): 11.8%
- Phase 4 (Reconstruction): 2.6%

### Throughput
- Average Query Time: 1.08 seconds
- Theoretical Throughput: 0.93 queries/second
