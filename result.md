# Trident Search Results
**Test time**: 2025-07-26 17:00:51
**Dataset**: tripclick

## Configuration Information
- Vector dimension: 768
- Number of documents: 1,523,87
- HNSW parameters: M=128, efConstruction=160
- Default efSearch: 36
- Number of queries: 100
- k: 10

## Data Information
- Query vectors: (1175, 768)
- Ground truth: (1175, 10)

## Test Results
| efSearch | MRR@10 | Average Latency (ms) |
|----------|-----------|---------------|
| 18 | 0.9333 | 11.71 |
| 36 | 0.9333 | 20.26 |
| 72 | 0.9333 | 34.97 |
| 144 | 0.9333 | 59.58 |

## Summary
- Test completion time: 2025-07-26 17:01:23
- Dataset: tripclick
- Index location: /home/anonymous/Test-Trident/dataset/tripclick


**Test time**: 2025-07-26 17:24:58
**Dataset**: laion

## Configuration Information
- Vector dimension: 512
- Number of documents: 100,000
- HNSW parameters: M=64, efConstruction=80
- Default efSearch: 32
- Number of queries: 100
- k: 10

## Data Information
- Query vectors: (1000, 512)
- Ground truth: (1000, 10)

## Test Results
| efSearch | MRR@10 | Average Latency (ms) |
|----------|-----------|---------------|
| 16 | 0.9800 | 5.15 |
| 32 | 0.9900 | 7.15 |
| 64 | 1.0000 | 11.58 |
| 128 | 1.0000 | 18.74 |

## Summary
- Test completion time: 2025-07-26 17:25:10
- Dataset: laion
- Index location: /home/anonymous/Test-Trident/dataset/laion



**Test time**: 2025-07-26 18:01:40
**Dataset**: siftsmall

## Configuration Information
- Vector dimension: 128
- Number of documents: 10,000
- HNSW parameters: M=64, efConstruction=80
- Default efSearch: 32
- Number of queries: 100
- k: 10

## Data Information
- Query vectors: (100, 128)
- Ground truth: (100, 100)

## Test Results
| efSearch | MRR@10 | Average Latency (ms) |
|----------|-----------|---------------|
| 16 | 0.9100 | 3.31 |
| 32 | 0.9500 | 4.60 |
| 64 | 0.9800 | 6.51 |
| 128 | 0.9800 | 9.98 |

## Summary
- Test completion time: 2025-07-26 18:01:43
- Dataset: siftsmall
- Index location: /home/anonymous/Test-Trident/dataset/siftsmall



# Trident Search Results
**Test time**: 2025-07-26 20:22:22
**Dataset**: nfcorpus

## Configuration Information
- Vector dimension: 768
- Number of documents: 3,633
- HNSW parameters: M=32, efConstruction=80
- Default efSearch: 32
- Number of queries: 100
- k: 10

## Data Information
- Query vectors: (323, 768)
- Ground truth: (323, 100)

## Test Results
| efSearch | MRR@10 | Average Latency (ms) |
|----------|-----------|---------------|
| 16 | 0.2582 | 2.65 |
| 32 | 0.2687 | 4.22 |
| 64 | 0.2682 | 6.40 |
| 128 | 0.2882 | 10.56 |

## Summary
- Test completion time: 2025-07-26 20:22:25
- Dataset: nfcorpus
- Index location: /home/anonymous/Test-Trident/dataset/nfcorpus







---

# Test Results Report - laion

**Generation time**: 2025-07-26 17:52:10
**Dataset**: laion
**Number of queries**: 10

## Detailed Query Results

| Query ID | Node ID | Phase 1 (VDPF) | Phase 2 (e/f) | Phase 3 (Exchange) | Phase 4 (Reconstruction) | Total Time | Cosine Similarity |
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
- **Phase 2 (e/f Computation)**: 1.19 seconds
- **Phase 3 (Data Exchange)**: 5.27 seconds
- **Phase 4 (Reconstruction)**: 3.26 seconds
- **Server Internal Total**: 16.91 seconds
- **Average Cosine Similarity**: 1.000000

## Performance Analysis

### Time Distribution

- Phase 1 (VDPF Evaluation): 41.4%
- Phase 2 (e/f Computation): 7.0%
- Phase 3 (Data Exchange): 31.1%
- Phase 4 (Reconstruction): 19.3%

### Throughput
- Average query time: 16.91 seconds
- Theoretical throughput: 0.06 queries/second


laion Encrypted Queries
=== Average Performance Statistics (10 successful queries) ===
  Phase 1 (Multi-process VDPF Evaluation): 7.01 seconds
  Phase 2 (e/f Computation): 1.19 seconds
  Phase 3 (Data Exchange): 5.27 seconds
  Phase 4 (Reconstruction): 3.26 seconds
  Server Internal Total: 16.91 seconds
  Average Cosine Similarity: 1.000000


---

# Test Results Report - laion

**Generation time**: 2025-07-26 17:58:38
**Dataset**: laion
**Number of queries**: 10

## Detailed Query Results

| Query ID | Node ID | Phase 1 (VDPF) | Phase 2 (e/f) | Phase 3 (Exchange) | Phase 4 (Reconstruction) | Total Time | Cosine Similarity |
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
- **Phase 2 (e/f Computation)**: 1.20 seconds
- **Phase 3 (Data Exchange)**: 5.29 seconds
- **Phase 4 (Reconstruction)**: 3.33 seconds
- **Server Internal Total**: 16.44 seconds
- **Average Cosine Similarity**: 1.000000

## Performance Analysis

### Time Distribution

- Phase 1 (VDPF Evaluation): 39.3%
- Phase 2 (e/f Computation): 7.3%
- Phase 3 (Data Exchange): 32.2%
- Phase 4 (Reconstruction): 20.3%

### Throughput
- Average query time: 16.44 seconds
- Theoretical throughput: 0.06 queries/second


---

# Test Results Report - siftsmall

**Generation time**: 2025-07-26 18:03:48
**Dataset**: siftsmall
**Number of queries**: 10

## Detailed Query Results

| Query ID | Node ID | Phase 1 (VDPF) | Phase 2 (e/f) | Phase 3 (Exchange) | Phase 4 (Reconstruction) | Total Time | Cosine Similarity |
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
- **Phase 2 (e/f Computation)**: 0.06 seconds
- **Phase 3 (Data Exchange)**: 0.18 seconds
- **Phase 4 (Reconstruction)**: 0.06 seconds
- **Server Internal Total**: 1.19 seconds
- **Average Cosine Similarity**: 1.000000

## Performance Analysis

### Time Distribution

- Phase 1 (VDPF Evaluation): 69.4%
- Phase 2 (e/f Computation): 5.0%
- Phase 3 (Data Exchange): 15.3%
- Phase 4 (Reconstruction): 4.6%

### Throughput
- Average query time: 1.19 seconds
- Theoretical throughput: 0.84 queries/second


---

---

# Test Results Report - tripclick

**Generation time**: 2025-07-26 20:19:17
**Dataset**: tripclick
**Number of queries**: 10

## Detailed Query Results

| Query ID | Node ID | Phase 1 (VDPF) | Phase 2 (e/f) | Phase 3 (Exchange) | Phase 4 (Reconstruction) | Total Time | Cosine Similarity |
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
- **Phase 2 (e/f Computation)**: 2.26 seconds
- **Phase 3 (Data Exchange)**: 11.86 seconds
- **Phase 4 (Reconstruction)**: 7.83 seconds
- **Server Internal Total**: 33.81 seconds
- **Average Cosine Similarity**: 1.000000

## Performance Analysis

### Time Distribution

- Phase 1 (VDPF Evaluation): 34.4%
- Phase 2 (e/f Computation): 6.7%
- Phase 3 (Data Exchange): 35.1%
- Phase 4 (Reconstruction): 23.2%

### Throughput
- Average query time: 33.81 seconds
- Theoretical throughput: 0.03 queries/second

---

# Test Results Report - nfcorpus

**Generation time**: 2025-07-26 20:39:42
**Dataset**: nfcorpus
**Number of queries**: 10

## Detailed Query Results

| Query ID | Node ID | Phase 1 (VDPF) | Phase 2 (e/f) | Phase 3 (Exchange) | Phase 4 (Reconstruction) | Total Time | Cosine Similarity |
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
- **Phase 2 (e/f Computation)**: 0.04 seconds
- **Phase 3 (Data Exchange)**: 0.25 seconds
- **Phase 4 (Reconstruction)**: 0.14 seconds
- **Server Internal Total**: 1.11 seconds
- **Average Cosine Similarity**: 1.000000

## Performance Analysis

### Time Distribution

- Phase 1 (VDPF Evaluation): 54.9%
- Phase 2 (e/f Computation): 3.5%
- Phase 3 (Data Exchange): 22.7%
- Phase 4 (Reconstruction): 12.7%

### Throughput
- Average query time: 1.11 seconds
- Theoretical throughput: 0.90 queries/second


---



---

# Test Results Report - siftsmall

**Generation time**: 2025-07-27 13:00:54
**Dataset**: siftsmall
**Number of queries**: 10

## Detailed Query Results

| Query ID | Node ID | Layer | Phase 1 (VDPF) | Phase 2 (e/f) | Phase 3 (Exchange) | Phase 4 (Reconstruction) | Total Time | Neighbor Match Rate |
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
- **Phase 2 (e/f Computation)**: 0.20 seconds
- **Phase 3 (Data Exchange)**: 0.37 seconds
- **Phase 4 (Reconstruction)**: 0.21 seconds
- **Server Internal Total**: 2.47 seconds
- **Average Neighbor Match Rate**: 100.00%

## Performance Analysis

### Time Distribution

- Phase 1 (VDPF Evaluation): 65.0%
- Phase 2 (e/f Computation): 8.1%
- Phase 3 (Data Exchange): 15.1%
- Phase 4 (Reconstruction): 8.6%

### Throughput
- Average query time: 2.47 seconds
- Theoretical throughput: 0.40 queries/second


---

# Test Results Report - laion

**Generation time**: 2025-07-27 13:09:26
**Dataset**: laion
**Number of queries**: 10

## Detailed Query Results

| Query ID | Node ID | Layer | Phase 1 (VDPF) | Phase 2 (e/f) | Phase 3 (Exchange) | Phase 4 (Reconstruction) | Total Time | Neighbor Match Rate |
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
- **Phase 2 (e/f Computation)**: 2.33 seconds
- **Phase 3 (Data Exchange)**: 3.85 seconds
- **Phase 4 (Reconstruction)**: 2.62 seconds
- **Server Internal Total**: 25.43 seconds
- **Average Neighbor Match Rate**: 100.00%

## Performance Analysis

### Time Distribution

- Phase 1 (VDPF Evaluation): 64.0%
- Phase 2 (e/f Computation): 9.2%
- Phase 3 (Data Exchange): 15.2%
- Phase 4 (Reconstruction): 10.3%

### Throughput
- Average query time: 25.43 seconds
- Theoretical throughput: 0.04 queries/second


---

# Test Results Report - tripclick

**Generation time**: 2025-07-27 13:25:12
**Dataset**: tripclick
**Number of queries**: 10

## Detailed Query Results

| Query ID | Node ID | Layer | Phase 1 (VDPF) | Phase 2 (e/f) | Phase 3 (Exchange) | Phase 4 (Reconstruction) | Total Time | Neighbor Match Rate |
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
- **Phase 2 (e/f Computation)**: 4.57 seconds
- **Phase 3 (Data Exchange)**: 13.53 seconds
- **Phase 4 (Reconstruction)**: 8.41 seconds
- **Server Internal Total**: 56.50 seconds
- **Average Neighbor Match Rate**: 100.00%

## Performance Analysis

### Time Distribution

- Phase 1 (VDPF Evaluation): 52.0%
- Phase 2 (e/f Computation): 8.1%
- Phase 3 (Data Exchange): 23.9%
- Phase 4 (Reconstruction): 14.9%

### Throughput
- Average query time: 56.50 seconds
- Theoretical throughput: 0.02 queries/second


---------

# Test Results Report - nfcorpus

**Generation time**: 2025-07-27 13:32:57
**Dataset**: nfcorpus
**Number of queries**: 10

## Detailed Query Results

| Query ID | Node ID | Layer | Phase 1 (VDPF) | Phase 2 (e/f) | Phase 3 (Exchange) | Phase 4 (Reconstruction) | Total Time | Neighbor Match Rate |
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
- **Phase 2 (e/f Computation)**: 0.06 seconds
- **Phase 3 (Data Exchange)**: 0.13 seconds
- **Phase 4 (Reconstruction)**: 0.03 seconds
- **Server Internal Total**: 1.08 seconds
- **Average Neighbor Match Rate**: 100.00%

## Performance Analysis

### Time Distribution

- Phase 1 (VDPF Evaluation): 74.0%
- Phase 2 (e/f Computation): 5.3%
- Phase 3 (Data Exchange): 11.8%
- Phase 4 (Reconstruction): 2.6%

### Throughput
- Average query time: 1.08 seconds
- Theoretical throughput: 0.93 queries/second


---

# Test Results Report - siftsmall

**Generation time**: 2025-07-29 13:33:15  
**Dataset**: siftsmall  
**Number of queries**: 10

## Detailed Query Results

| Query ID | Node ID | Phase 1 (VDPF) | Phase 2 (e/f) | Phase 3 (Exchange) | Phase 4 (Reconstruction) | Total Time | Cosine Similarity |
|---------|--------|--------------|-------------|--------------|--------------|--------|-----------|
| 1 | 5365 | 1.03s | 0.08s | 0.22s | 0.06s | 1.49s | 1.000000 |
| 2 | 1339 | 0.94s | 0.07s | 0.22s | 0.06s | 1.36s | 1.000000 |
| 3 | 1194 | 0.96s | 0.07s | 0.21s | 0.06s | 1.41s | 1.000000 |
| 4 | 9784 | 0.96s | 0.07s | 0.22s | 0.06s | 1.38s | 1.000000 |
| 5 | 516 | 0.96s | 0.07s | 0.22s | 0.06s | 1.38s | 1.000000 |
| 6 | 4456 | 0.99s | 0.08s | 0.22s | 0.06s | 1.45s | 1.000000 |
| 7 | 8890 | 1.00s | 0.07s | 0.22s | 0.06s | 1.42s | 1.000000 |
| 8 | 858 | 0.97s | 0.08s | 0.22s | 0.06s | 1.40s | 1.000000 |
| 9 | 8057 | 0.95s | 0.07s | 0.22s | 0.06s | 1.37s | 1.000000 |
| 10 | 9982 | 0.99s | 0.07s | 0.22s | 0.06s | 1.41s | 1.000000 |

## Average Performance Statistics

- **Phase 1 (Multi-process VDPF Evaluation)**: 0.97 seconds
- **Phase 2 (e/f Computation)**: 0.07 seconds
- **Phase 3 (Data Exchange)**: 0.22 seconds
- **Phase 4 (Reconstruction)**: 0.06 seconds
- **Server Internal Total**: 1.41 seconds
- **Average Cosine Similarity**: 1.000000

## Performance Analysis

### Time Distribution

- Phase 1 (VDPF Evaluation): 69.2%
- Phase 2 (e/f Computation): 5.3%
- Phase 3 (Data Exchange): 15.4%
- Phase 4 (Reconstruction): 4.5%

### Throughput
- Average query time: 1.41 seconds
- Theoretical throughput: 0.71 queries/second


---

# Test Results Report - siftsmall

**Generation time**: 2025-07-29 13:37:10  
**Dataset**: siftsmall  
**Number of queries**: 10

## Detailed Query Results

| Query ID | Node ID | Phase 1 (VDPF) | Phase 2 (e/f) | Phase 3 (Exchange) | Phase 4 (Reconstruction) | Total Time | Cosine Similarity |
|---------|--------|--------------|-------------|--------------|--------------|--------|-----------|
| 1 | 9270 | 1.17s | 0.08s | 0.22s | 0.06s | 1.60s | 1.000000 |
| 2 | 3624 | 1.18s | 0.07s | 0.22s | 0.06s | 1.60s | 1.000000 |
| 3 | 2128 | 1.21s | 0.07s | 0.22s | 0.06s | 1.63s | 1.000000 |
| 4 | 1937 | 1.17s | 0.08s | 0.22s | 0.06s | 1.59s | 1.000000 |
| 5 | 3882 | 1.23s | 0.07s | 0.22s | 0.06s | 1.65s | 1.000000 |
| 6 | 3180 | 1.20s | 0.07s | 0.22s | 0.06s | 1.62s | 1.000000 |
| 7 | 9005 | 1.21s | 0.08s | 0.22s | 0.06s | 1.63s | 1.000000 |
| 8 | 7599 | 1.18s | 0.08s | 0.22s | 0.06s | 1.61s | 1.000000 |
| 9 | 5896 | 1.21s | 0.07s | 0.22s | 0.06s | 1.63s | 1.000000 |
| 10 | 2850 | 1.12s | 0.08s | 0.22s | 0.06s | 1.55s | 1.000000 |

## Average Performance Statistics

- **Phase 1 (Multi-process VDPF Evaluation)**: 1.19 seconds
- **Phase 2 (e/f Computation)**: 0.08 seconds
- **Phase 3 (Data Exchange)**: 0.22 seconds
- **Phase 4 (Reconstruction)**: 0.06 seconds
- **Server Internal Total**: 1.61 seconds
- **Average Cosine Similarity**: 1.000000

## Performance Analysis

### Time Distribution

- Phase 1 (VDPF Evaluation): 73.7%
- Phase 2 (e/f Computation): 4.7%
- Phase 3 (Data Exchange): 13.5%
- Phase 4 (Reconstruction): 3.9%

### Throughput
- Average query time: 1.61 seconds
- Theoretical throughput: 0.62 queries/second


---

# Test Results Report - siftsmall

**Generation time**: 2025-07-29 13:44:32  
**Dataset**: siftsmall  
**Number of queries**: 10

## Detailed Query Results

| Query ID | Node ID | Phase 1 (VDPF) | Phase 2 (e/f) | Phase 3 (Exchange) | Phase 4 (Reconstruction) | Total Time | Cosine Similarity |
|---------|--------|--------------|-------------|--------------|--------------|--------|-----------|
| 1 | 9279 | 1.05s | 0.08s | 0.22s | 0.06s | 1.48s | 1.000000 |
| 2 | 7310 | 1.07s | 0.08s | 0.22s | 0.06s | 1.49s | 1.000000 |
| 3 | 7676 | 1.10s | 0.07s | 0.22s | 0.06s | 1.52s | 1.000000 |
| 4 | 2548 | 1.12s | 0.08s | 0.22s | 0.06s | 1.58s | 1.000000 |
| 5 | 3245 | 1.09s | 0.07s | 0.22s | 0.06s | 1.54s | 1.000000 |
| 6 | 3254 | 1.07s | 0.07s | 0.22s | 0.06s | 1.49s | 1.000000 |
| 7 | 4721 | 1.06s | 0.08s | 0.22s | 0.06s | 1.49s | 1.000000 |
| 8 | 3830 | 1.09s | 0.07s | 0.22s | 0.06s | 1.54s | 1.000000 |
| 9 | 3879 | 1.05s | 0.07s | 0.22s | 0.06s | 1.47s | 1.000000 |
| 10 | 4515 | 1.08s | 0.08s | 0.22s | 0.06s | 1.51s | 1.000000 |

## Average Performance Statistics

- **Phase 1 (Multi-process VDPF Evaluation)**: 1.08 seconds
- **Phase 2 (e/f Computation)**: 0.07 seconds
- **Phase 3 (Data Exchange)**: 0.22 seconds
- **Phase 4 (Reconstruction)**: 0.06 seconds
- **Server Internal Total**: 1.51 seconds
- **Average Cosine Similarity**: 1.000000

## Performance Analysis

### Time Distribution

- Phase 1 (VDPF Evaluation): 71.4%
- Phase 2 (e/f Computation): 4.9%
- Phase 3 (Data Exchange): 14.4%
- Phase 4 (Reconstruction): 4.1%

### Throughput
- Average query time: 1.51 seconds
- Theoretical throughput: 0.66 queries/second


---

# Test Results Report - siftsmall

**Generation time**: 2025-07-29 13:45:55  
**Dataset**: siftsmall  
**Number of queries**: 10

## Detailed Query Results

| Query ID | Node ID | Phase 1 (VDPF) | Phase 2 (e/f) | Phase 3 (Exchange) | Phase 4 (Reconstruction) | Total Time | Cosine Similarity |
|---------|--------|--------------|-------------|--------------|--------------|--------|-----------|
| 1 | 7393 | 0.98s | 0.08s | 0.22s | 0.06s | 1.41s | 1.000000 |
| 2 | 5165 | 0.90s | 0.07s | 0.22s | 0.06s | 1.35s | 1.000000 |
| 3 | 402 | 0.94s | 0.08s | 0.22s | 0.06s | 1.37s | 1.000000 |
| 4 | 9130 | 0.95s | 0.08s | 0.22s | 0.06s | 1.41s | 1.000000 |
| 5 | 4156 | 0.95s | 0.07s | 0.22s | 0.06s | 1.41s | 1.000000 |
| 6 | 580 | 0.96s | 0.07s | 0.22s | 0.06s | 1.38s | 1.000000 |
| 7 | 9139 | 0.94s | 0.08s | 0.22s | 0.06s | 1.40s | 1.000000 |
| 8 | 1660 | 0.95s | 0.07s | 0.22s | 0.06s | 1.37s | 1.000000 |
| 9 | 8519 | 1.00s | 0.07s | 0.22s | 0.06s | 1.42s | 1.000000 |
| 10 | 267 | 0.93s | 0.08s | 0.22s | 0.06s | 1.36s | 1.000000 |

## Average Performance Statistics

- **Phase 1 (Multi-process VDPF Evaluation)**: 0.95 seconds
- **Phase 2 (e/f Computation)**: 0.08 seconds
- **Phase 3 (Data Exchange)**: 0.22 seconds
- **Phase 4 (Reconstruction)**: 0.06 seconds
- **Server Internal Total**: 1.39 seconds
- **Average Cosine Similarity**: 1.000000

## Performance Analysis

### Time Distribution

- Phase 1 (VDPF Evaluation): 68.5%
- Phase 2 (e/f Computation): 5.4%
- Phase 3 (Data Exchange): 15.6%
- Phase 4 (Reconstruction): 4.5%

### Throughput
- Average query time: 1.39 seconds
- Theoretical throughput: 0.72 queries/second


---

# Test Results Report - siftsmall

**Generation time**: 2025-07-29 13:46:52  
**Dataset**: siftsmall  
**Number of queries**: 10

## Detailed Query Results

| Query ID | Node ID | Phase 1 (VDPF) | Phase 2 (e/f) | Phase 3 (Exchange) | Phase 4 (Reconstruction) | Total Time | Cosine Similarity |
|---------|--------|--------------|-------------|--------------|--------------|--------|-----------|
| 1 | 9908 | 1.79s | 0.08s | 0.23s | 0.06s | 2.34s | 1.000000 |
| 2 | 3268 | 1.70s | 0.08s | 0.22s | 0.06s | 2.17s | 1.000000 |
| 3 | 853 | 1.66s | 0.08s | 0.22s | 0.06s | 2.12s | 1.000000 |
| 4 | 4624 | 1.75s | 0.07s | 0.22s | 0.06s | 2.20s | 1.000000 |
| 5 | 4996 | 1.73s | 0.07s | 0.22s | 0.06s | 2.21s | 1.000000 |
| 6 | 1483 | 1.72s | 0.07s | 0.22s | 0.06s | 2.17s | 1.000000 |
| 7 | 6608 | 1.71s | 0.08s | 0.22s | 0.06s | 2.17s | 1.000000 |
| 8 | 3081 | 1.72s | 0.08s | 0.22s | 0.06s | 2.15s | 1.000000 |
| 9 | 1557 | 1.69s | 0.07s | 0.22s | 0.06s | 2.11s | 1.000000 |
| 10 | 3579 | 1.74s | 0.07s | 0.22s | 0.06s | 2.23s | 1.000000 |

## Average Performance Statistics

- **Phase 1 (Multi-process VDPF Evaluation)**: 1.72 seconds
- **Phase 2 (e/f Computation)**: 0.08 seconds
- **Phase 3 (Data Exchange)**: 0.22 seconds
- **Phase 4 (Reconstruction)**: 0.06 seconds
- **Server Internal Total**: 2.19 seconds
- **Average Cosine Similarity**: 1.000000

## Performance Analysis

### Time Distribution

- Phase 1 (VDPF Evaluation): 78.7%
- Phase 2 (e/f Computation): 3.5%
- Phase 3 (Data Exchange): 9.9%
- Phase 4 (Reconstruction): 2.9%

### Throughput
- Average query time: 2.19 seconds
- Theoretical throughput: 0.46 queries/second


---

# Test Results Report - siftsmall

**Generation time**: 2025-07-29 13:48:09  
**Dataset**: siftsmall  
**Number of queries**: 10

## Detailed Query Results

| Query ID | Node ID | Phase 1 (VDPF) | Phase 2 (e/f) | Phase 3 (Exchange) | Phase 4 (Reconstruction) | Total Time | Cosine Similarity |
|---------|--------|--------------|-------------|--------------|--------------|--------|-----------|
| 1 | 4192 | 1.18s | 0.08s | 0.23s | 0.06s | 1.65s | 1.000000 |
| 2 | 6960 | 1.19s | 0.08s | 0.22s | 0.06s | 1.62s | 1.000000 |
| 3 | 8582 | 1.13s | 0.07s | 0.18s | 0.06s | 1.56s | 1.000000 |
| 4 | 2386 | 1.15s | 0.07s | 0.20s | 0.06s | 1.55s | 1.000000 |
| 5 | 82 | 1.14s | 0.08s | 0.18s | 0.06s | 1.57s | 1.000000 |
| 6 | 2116 | 1.18s | 0.08s | 0.22s | 0.06s | 1.64s | 1.000000 |
| 7 | 7145 | 1.16s | 0.09s | 0.22s | 0.06s | 1.69s | 1.000000 |
| 8 | 9787 | 1.16s | 0.07s | 0.22s | 0.06s | 1.57s | 1.000000 |
| 9 | 207 | 1.16s | 0.08s | 0.22s | 0.06s | 1.59s | 1.000000 |
| 10 | 4242 | 1.14s | 0.07s | 0.22s | 0.06s | 1.56s | 1.000000 |

## Average Performance Statistics

- **Phase 1 (Multi-process VDPF Evaluation)**: 1.16 seconds
- **Phase 2 (e/f Computation)**: 0.08 seconds
- **Phase 3 (Data Exchange)**: 0.21 seconds
- **Phase 4 (Reconstruction)**: 0.06 seconds
- **Server Internal Total**: 1.60 seconds
- **Average Cosine Similarity**: 1.000000

## Performance Analysis

### Time Distribution

- Phase 1 (VDPF Evaluation): 72.5%
- Phase 2 (e/f Computation): 4.8%
- Phase 3 (Data Exchange): 13.1%
- Phase 4 (Reconstruction): 3.9%

### Throughput
- Average query time: 1.60 seconds
- Theoretical throughput: 0.63 queries/second






## Detailed Query Results

| Query ID | Node ID | Phase 1 (VDPF) | Phase 2 (e/f) | Phase 3 (Exchange) | Phase 4 (Reconstruction) | Total Time | Cosine Similarity |
|---------|--------|--------------|-------------|--------------|--------------|--------|-----------|
| 1 | 6914 | 1.04s | 0.08s | 0.22s | 0.06s | 1.61s | 1.000000 |
| 2 | 4686 | 1.00s | 0.08s | 0.21s | 0.06s | 1.52s | 1.000000 |
| 3 | 6785 | 1.00s | 0.07s | 0.22s | 0.06s | 1.52s | 1.000000 |
| 4 | 6040 | 0.96s | 0.07s | 0.22s | 0.06s | 1.48s | 1.000000 |
| 5 | 4498 | 1.03s | 0.08s | 0.28s | 0.06s | 1.59s | 1.000000 |
| 6 | 5605 | 0.97s | 0.08s | 0.22s | 0.06s | 1.49s | 1.000000 |
| 7 | 5527 | 0.99s | 0.07s | 0.22s | 0.06s | 1.50s | 1.000000 |
| 8 | 479 | 0.98s | 0.08s | 0.22s | 0.06s | 1.50s | 1.000000 |
| 9 | 8047 | 0.99s | 0.08s | 0.22s | 0.06s | 1.51s | 1.000000 |
| 10 | 2557 | 1.00s | 0.07s | 0.22s | 0.06s | 1.55s | 1.000000 |

## Average Performance Statistics

- **Phase 1 (Multi-process VDPF Evaluation)**: 1.00 seconds
- **Phase 2 (e/f Computation)**: 0.08 seconds
- **Phase 3 (Data Exchange)**: 0.22 seconds
- **Phase 4 (Reconstruction)**: 0.06 seconds
- **Server Internal Total**: 1.53 seconds
- **Average Cosine Similarity**: 1.000000

## Performance Analysis

### Time Distribution

- Phase 1 (VDPF Evaluation): 65.4%
- Phase 2 (e/f Computation): 4.9%
- Phase 3 (Data Exchange): 14.6%
- Phase 4 (Reconstruction): 3.8%

### Throughput
- Average query time: 1.53 seconds
- Theoretical throughput: 0.66 queries/second


---

# Test Results Report - siftsmall

**Generation time**: 2025-07-30 11:51:54  
**Dataset**: siftsmall  
**Number of queries**: 10

## Detailed Query Results

| Query ID | Node ID | Phase 1 (VDPF) | Phase 2 (e/f) | Phase 3 (Exchange) | Phase 4 (Reconstruction) | Total Time | Cosine Similarity |
|---------|--------|--------------|-------------|--------------|--------------|--------|-----------|
| 1 | 6370 | 0.95s | 0.08s | 0.22s | 0.06s | 1.41s | 1.000000 |
| 2 | 7385 | 0.95s | 0.08s | 0.22s | 0.06s | 1.37s | 1.000000 |
| 3 | 4968 | 0.94s | 0.08s | 0.23s | 0.06s | 1.38s | 1.000000 |
| 4 | 3112 | 0.94s | 0.07s | 0.22s | 0.06s | 1.36s | 1.000000 |
| 5 | 1322 | 0.95s | 0.07s | 0.22s | 0.06s | 1.41s | 1.000000 |
| 6 | 6442 | 0.94s | 0.08s | 0.22s | 0.06s | 1.41s | 1.000000 |
| 7 | 9939 | 0.89s | 0.07s | 0.22s | 0.06s | 1.31s | 1.000000 |
| 8 | 9924 | 1.04s | 0.08s | 0.22s | 0.06s | 1.50s | 1.000000 |
| 9 | 4090 | 0.94s | 0.07s | 0.22s | 0.06s | 1.36s | 1.000000 |
| 10 | 2430 | 0.96s | 0.07s | 0.22s | 0.06s | 1.38s | 1.000000 |

## Average Performance Statistics

- **Phase 1 (Multi-process VDPF Evaluation)**: 0.95 seconds
- **Phase 2 (e/f Computation)**: 0.08 seconds
- **Phase 3 (Data Exchange)**: 0.22 seconds
- **Phase 4 (Reconstruction)**: 0.06 seconds
- **Server Internal Total**: 1.39 seconds
- **Average Cosine Similarity**: 1.000000

## Performance Analysis

### Time Distribution

- Phase 1 (VDPF Evaluation): 68.4%
- Phase 2 (e/f Computation): 5.4%
- Phase 3 (Data Exchange): 15.7%
- Phase 4 (Reconstruction): 4.5%

### Throughput
- Average query time: 1.39 seconds
- Theoretical throughput: 0.72 queries/second


---

# Test Results Report - siftsmall

**Generation time**: 2025-07-30 12:03:04  
**Dataset**: siftsmall  
**Number of queries**: 10

## Detailed Query Results

| Query ID | Node ID | Phase 1 (VDPF) | Phase 2 (e/f) | Phase 3 (Exchange) | Phase 4 (Reconstruction) | Total Time | Cosine Similarity |
|---------|--------|--------------|-------------|--------------|--------------|--------|-----------|
| 1 | 758 | 1.02s | 0.08s | 0.22s | 0.06s | 1.45s | 1.000000 |
| 2 | 6656 | 0.95s | 0.08s | 0.22s | 0.06s | 1.41s | 1.000000 |
| 3 | 4665 | 0.97s | 0.07s | 0.22s | 0.06s | 1.40s | 1.000000 |
| 4 | 1436 | 0.95s | 0.07s | 0.22s | 0.06s | 1.40s | 1.000000 |
| 5 | 2657 | 0.96s | 0.08s | 0.22s | 0.06s | 1.39s | 1.000000 |
| 6 | 4376 | 0.97s | 0.08s | 0.22s | 0.06s | 1.46s | 1.000000 |
| 7 | 5552 | 0.90s | 0.07s | 0.22s | 0.06s | 1.32s | 1.000000 |
| 8 | 118 | 0.96s | 0.09s | 0.22s | 0.06s | 1.53s | 1.000000 |
| 9 | 3446 | 1.01s | 0.07s | 0.22s | 0.06s | 1.46s | 1.000000 |
| 10 | 9249 | 0.95s | 0.07s | 0.22s | 0.06s | 1.37s | 1.000000 |

## Average Performance Statistics

- **Phase 1 (Multi-process VDPF Evaluation)**: 0.96 seconds
- **Phase 2 (e/f Computation)**: 0.08 seconds
- **Phase 3 (Data Exchange)**: 0.22 seconds
- **Phase 4 (Reconstruction)**: 0.06 seconds
- **Server Internal Total**: 1.42 seconds
- **Average Cosine Similarity**: 1.000000

## Performance Analysis

### Time Distribution

- Phase 1 (VDPF Evaluation): 67.9%
- Phase 2 (e/f Computation): 5.4%
- Phase 3 (Data Exchange): 15.4%
- Phase 4 (Reconstruction): 4.4%

### Throughput
- Average query time: 1.42 seconds
- Theoretical throughput: 0.70 queries/second


---

# Test Results Report - siftsmall

**Generation time**: 2025-07-30 14:09:48  
**Dataset**: siftsmall  
**Number of queries**: 8

## Detailed Query Results

| Query ID | Node ID | Phase 1 (VDPF) | Phase 2 (e/f) | Phase 3 (Exchange) | Phase 4 (Reconstruction) | Total Time | Cosine Similarity |
|---------|--------|--------------|-------------|--------------|--------------|--------|-----------|
| 1 | 7988 | 0.32s | 0.08s | 0.22s | 0.06s | 0.74s | 1.000000 |
| 2 | 9273 | 0.34s | 0.07s | 0.21s | 0.06s | 0.74s | 1.000000 |
| 3 | 9240 | 0.31s | 0.07s | 0.21s | 0.06s | 0.72s | 1.000000 |
| 5 | 4576 | 0.32s | 0.07s | 0.25s | 0.06s | 0.76s | 1.000000 |
| 6 | 5321 | 0.32s | 0.07s | 0.21s | 0.06s | 0.72s | 1.000000 |
| 7 | 2528 | 0.35s | 0.07s | 0.22s | 0.06s | 0.76s | 1.000000 |
| 8 | 3060 | 0.33s | 0.07s | 0.21s | 0.06s | 0.74s | 1.000000 |
| 9 | 3530 | 0.32s | 0.08s | 0.21s | 0.06s | 0.73s | 1.000000 |

## Average Performance Statistics

- **Phase 1 (Multi-process VDPF Evaluation)**: 0.32 seconds
- **Phase 2 (e/f Computation)**: 0.07 seconds
- **Phase 3 (Data Exchange)**: 0.22 seconds
- **Phase 4 (Reconstruction)**: 0.06 seconds
- **Server Internal Total**: 0.74 seconds
- **Average Cosine Similarity**: 1.000000

## Performance Analysis

### Time Distribution

- Phase 1 (VDPF Evaluation): 44.0%
- Phase 2 (e/f Computation): 9.9%
- Phase 3 (Data Exchange): 29.5%
- Phase 4 (Reconstruction): 7.6%

### Throughput
- Average query time: 0.74 seconds
- Theoretical throughput: 1.35 queries/second


---

# Test Results Report - siftsmall

**Generation time**: 2025-07-30 14:23:28  
**Dataset**: siftsmall  
**Number of queries**: 10

## Detailed Query Results

| Query ID | Node ID | Phase 1 (VDPF) | Phase 2 (e/f) | Phase 3 (Exchange) | Phase 4 (Reconstruction) | Total Time | Cosine Similarity |
|---------|--------|--------------|-------------|--------------|--------------|--------|-----------|
| 1 | 6262 | 0.33s | 0.08s | 0.23s | 0.06s | 0.75s | 1.000000 |
| 2 | 3097 | 0.32s | 0.07s | 0.22s | 0.06s | 0.73s | 1.000000 |
| 3 | 8231 | 0.34s | 0.07s | 0.22s | 0.06s | 0.75s | 1.000000 |
| 4 | 558 | 0.33s | 0.08s | 0.22s | 0.06s | 0.74s | 1.000000 |
| 5 | 163 | 0.34s | 0.07s | 0.21s | 0.06s | 0.75s | 1.000000 |
| 6 | 4455 | 0.33s | 0.07s | 0.22s | 0.06s | 0.74s | 1.000000 |
| 7 | 6420 | 0.32s | 0.07s | 0.21s | 0.06s | 0.74s | 1.000000 |
| 8 | 5074 | 0.32s | 0.07s | 0.21s | 0.06s | 0.73s | 1.000000 |
| 9 | 8050 | 0.33s | 0.08s | 0.21s | 0.06s | 0.74s | 1.000000 |
| 10 | 2315 | 0.33s | 0.08s | 0.22s | 0.06s | 0.74s | 1.000000 |

## Average Performance Statistics

- **Phase 1 (Multi-process VDPF Evaluation)**: 0.33 seconds
- **Phase 2 (e/f Computation)**: 0.08 seconds
- **Phase 3 (Data Exchange)**: 0.22 seconds
- **Phase 4 (Reconstruction)**: 0.06 seconds
- **Server Internal Total**: 0.74 seconds
- **Average Cosine Similarity**: 1.000000

## Performance Analysis

### Time Distribution

- Phase 1 (VDPF Evaluation): 44.1%
- Phase 2 (e/f Computation): 10.2%
- Phase 3 (Data Exchange): 29.0%
- Phase 4 (Reconstruction): 7.6%

### Throughput
- Average query time: 0.74 seconds
- Theoretical throughput: 1.35 queries/second


---

# Test Results Report - siftsmall

**Generation time**: 2025-07-30 14:24:23  
**Dataset**: siftsmall  
**Number of queries**: 10

## Detailed Query Results

| Query ID | Node ID | Phase 1 (VDPF) | Phase 2 (e/f) | Phase 3 (Exchange) | Phase 4 (Reconstruction) | Total Time | Cosine Similarity |
|---------|--------|--------------|-------------|--------------|--------------|--------|-----------|
| 1 | 3931 | 0.34s | 0.08s | 0.22s | 0.06s | 0.76s | 1.000000 |
| 2 | 7365 | 0.32s | 0.07s | 0.22s | 0.06s | 0.73s | 1.000000 |
| 3 | 7463 | 0.32s | 0.08s | 0.22s | 0.06s | 0.73s | 1.000000 |
| 4 | 2502 | 0.33s | 0.08s | 0.22s | 0.06s | 0.75s | 1.000000 |
| 5 | 9830 | 0.32s | 0.07s | 0.21s | 0.06s | 0.73s | 1.000000 |
| 6 | 9534 | 0.33s | 0.07s | 0.22s | 0.06s | 0.74s | 1.000000 |
| 7 | 3562 | 0.33s | 0.07s | 0.21s | 0.06s | 0.74s | 1.000000 |
| 8 | 7883 | 0.32s | 0.07s | 0.22s | 0.06s | 0.74s | 1.000000 |
| 9 | 625 | 0.32s | 0.08s | 0.21s | 0.06s | 0.74s | 1.000000 |
| 10 | 5926 | 0.32s | 0.07s | 0.22s | 0.06s | 0.73s | 1.000000 |

## Average Performance Statistics

- **Phase 1 (Multi-process VDPF Evaluation)**: 0.32 seconds
- **Phase 2 (e/f Computation)**: 0.08 seconds
- **Phase 3 (Data Exchange)**: 0.22 seconds
- **Phase 4 (Reconstruction)**: 0.06 seconds
- **Server Internal Total**: 0.74 seconds
- **Average Cosine Similarity**: 1.000000

## Performance Analysis

### Time Distribution

- Phase 1 (VDPF Evaluation): 43.8%
- Phase 2 (e/f Computation): 10.2%
- Phase 3 (Data Exchange): 29.3%
- Phase 4 (Reconstruction): 7.7%

### Throughput
- Average query time: 0.74 seconds
- Theoretical throughput: 1.35 queries/second
