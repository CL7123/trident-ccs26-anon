(venv) anonymous@cs-anonymous-1 ~/T/experiment> python faiss_hnsw_query.py --run-all --k 10

================================================================================
Running experiments for SIFTSMALL
================================================================================

Loading dataset: siftsmall
Base vectors: (10000, 128)
Query vectors: (100, 128)
Ground truth: (100, 100)
Building HNSW index with M=64, ef_construction=80
Index built in 0.16 seconds
Index contains 10000 vectors

Searching with ef_search=16, k=1
  Search time: 0.003s
  QPS: 36921.7
  Recall@1: 1.0000
  MRR@1: 1.0000

Searching with ef_search=16, k=10
  Search time: 0.001s
  QPS: 90511.5
  Recall@10: 0.9950
  MRR@10: 1.0000

Searching with ef_search=16, k=100
  Search time: 0.001s
  QPS: 110901.7
  Recall@100: 0.9421
  MRR@100: 1.0000

Searching with ef_search=32, k=1
  Search time: 0.003s
  QPS: 31971.2
  Recall@1: 1.0000
  MRR@1: 1.0000

Searching with ef_search=32, k=10
  Search time: 0.001s
  QPS: 68189.0
  Recall@10: 1.0000
  MRR@10: 1.0000

Searching with ef_search=32, k=100
  Search time: 0.001s
  QPS: 131896.4
  Recall@100: 0.9872
  MRR@100: 1.0000

Searching with ef_search=64, k=1
  Search time: 0.001s
  QPS: 110609.3
  Recall@1: 1.0000
  MRR@1: 1.0000

Searching with ef_search=64, k=10
  Search time: 0.001s
  QPS: 120456.7
  Recall@10: 1.0000
  MRR@10: 1.0000

Searching with ef_search=64, k=100
  Search time: 0.001s
  QPS: 108240.1
  Recall@100: 0.9986
  MRR@100: 1.0000

Searching with ef_search=128, k=1
  Search time: 0.001s
  QPS: 86730.9
  Recall@1: 1.0000
  MRR@1: 1.0000

Searching with ef_search=128, k=10
  Search time: 0.002s
  QPS: 53842.2
  Recall@10: 1.0000
  MRR@10: 1.0000

Searching with ef_search=128, k=100
  Search time: 0.002s
  QPS: 66250.3
  Recall@100: 1.0000
  MRR@100: 1.0000

================================================================================
Running experiments for NFCORPUS
================================================================================

Loading dataset: nfcorpus
Base vectors: (3633, 768)
Query vectors: (323, 768)
Ground truth: (323, 100)
Building HNSW index with M=32, ef_construction=80
Index built in 0.07 seconds
Index contains 3633 vectors

Searching with ef_search=16, k=1
  Search time: 0.002s
  QPS: 165902.5
  Recall@1: 0.0495
  MRR@1: 0.3932

Searching with ef_search=16, k=10
  Search time: 0.002s
  QPS: 203754.0
  Recall@10: 0.1068
  MRR@10: 0.4751

Searching with ef_search=16, k=100
  Search time: 0.003s
  QPS: 103047.1
  Recall@100: 0.0580
  MRR@100: 0.4814

Searching with ef_search=32, k=1
  Search time: 0.004s
  QPS: 90059.2
  Recall@1: 0.0557
  MRR@1: 0.4087

Searching with ef_search=32, k=10
  Search time: 0.007s
  QPS: 47709.5
  Recall@10: 0.1084
  MRR@10: 0.4888

Searching with ef_search=32, k=100
  Search time: 0.004s
  QPS: 87358.8
  Recall@100: 0.0588
  MRR@100: 0.4959

Searching with ef_search=64, k=1
  Search time: 0.006s
  QPS: 55127.6
  Recall@1: 0.0588
  MRR@1: 0.4118

Searching with ef_search=64, k=10
  Search time: 0.006s
  QPS: 57507.4
  Recall@10: 0.1080
  MRR@10: 0.4940

Searching with ef_search=64, k=100
  Search time: 0.012s
  QPS: 27172.9
  Recall@100: 0.0580
  MRR@100: 0.5015

Searching with ef_search=128, k=1
  Search time: 0.010s
  QPS: 33745.0
  Recall@1: 0.0588
  MRR@1: 0.4149

Searching with ef_search=128, k=10
  Search time: 0.009s
  QPS: 36135.6
  Recall@10: 0.1090
  MRR@10: 0.4969

Searching with ef_search=128, k=100
  Search time: 0.007s
  QPS: 45047.6
  Recall@100: 0.0578
  MRR@100: 0.5043

================================================================================
Running experiments for LAION
================================================================================

Loading dataset: laion
Base vectors: (100000, 512)
Query vectors: (1000, 512)
Ground truth: (1000, 10)
Building HNSW index with M=64, ef_construction=80
Index built in 21.61 seconds
Index contains 100000 vectors

Searching with ef_search=16, k=1
  Search time: 0.078s
  QPS: 12744.5
  Recall@1: 0.9940
  MRR@1: 0.9990

Searching with ef_search=16, k=10
  Search time: 0.077s
  QPS: 13026.7
  Recall@10: 0.9818
  MRR@10: 1.0000

Searching with ef_search=32, k=1
  Search time: 0.114s
  QPS: 8742.7
  Recall@1: 0.9960
  MRR@1: 0.9990

Searching with ef_search=32, k=10
  Search time: 0.117s
  QPS: 8514.4
  Recall@10: 0.9873
  MRR@10: 1.0000

Searching with ef_search=64, k=1
  Search time: 0.228s
  QPS: 4386.6
  Recall@1: 0.9970
  MRR@1: 0.9990

Searching with ef_search=64, k=10
  Search time: 0.203s
  QPS: 4937.8
  Recall@10: 0.9895
  MRR@10: 1.0000

Searching with ef_search=128, k=1
  Search time: 0.340s
  QPS: 2945.1
  Recall@1: 0.9970
  MRR@1: 0.9990

Searching with ef_search=128, k=10
  Search time: 0.319s
  QPS: 3135.7
  Recall@10: 0.9903
  MRR@10: 1.0000

================================================================================
Running experiments for TRIPCLICK
================================================================================

Loading dataset: tripclick
Base vectors: (152387, 768)
Query vectors: (1175, 768)
Ground truth: (1175, 10)
Building HNSW index with M=128, ef_construction=160
Index built in 217.10 seconds
Index contains 152387 vectors

Searching with ef_search=18, k=1
  Search time: 0.333s
  QPS: 3527.8
  Recall@1: 0.2494
  MRR@1: 0.8026

Searching with ef_search=18, k=10
  Search time: 0.335s
  QPS: 3503.8
  Recall@10: 0.4919
  MRR@10: 0.8701

Searching with ef_search=36, k=1
  Search time: 0.571s
  QPS: 2056.0
  Recall@1: 0.2502
  MRR@1: 0.8068

Searching with ef_search=36, k=10
  Search time: 0.562s
  QPS: 2089.1
  Recall@10: 0.4923
  MRR@10: 0.8743

Searching with ef_search=72, k=1
  Search time: 1.022s
  QPS: 1149.5
  Recall@1: 0.2502
  MRR@1: 0.8077

Searching with ef_search=72, k=10
  Search time: 0.986s
  QPS: 1191.5
  Recall@10: 0.4926
  MRR@10: 0.8750

Searching with ef_search=144, k=1
  Search time: 1.777s
  QPS: 661.1
  Recall@1: 0.2502
  MRR@1: 0.8077

Searching with ef_search=144, k=10
  Search time: 1.815s
  QPS: 647.5
  Recall@10: 0.4925
  MRR@10: 0.8750

================================================================================
COMPREHENSIVE RESULTS SUMMARY
================================================================================

SIFTSMALL:
  Dataset: 10000 vectors, 128D
  HNSW: M=64, ef_construction=80
  Build time: 0.16s
  Best results by k:
    k=1: MRR=1.0000, Recall=1.0000, QPS=36921.7 (ef_search=16)
    k=10: MRR=1.0000, Recall=0.9950, QPS=90511.5 (ef_search=16)
    k=100: MRR=1.0000, Recall=0.9421, QPS=110901.7 (ef_search=16)

NFCORPUS:
  Dataset: 3633 vectors, 768D
  HNSW: M=32, ef_construction=80
  Build time: 0.07s
  Best results by k:
    k=1: MRR=0.4149, Recall=0.0588, QPS=33745.0 (ef_search=128)
    k=10: MRR=0.4969, Recall=0.1090, QPS=36135.6 (ef_search=128)
    k=100: MRR=0.5043, Recall=0.0578, QPS=45047.6 (ef_search=128)

LAION:
  Dataset: 100000 vectors, 512D
  HNSW: M=64, ef_construction=80
  Build time: 21.61s
  Best results by k:
    k=1: MRR=0.9990, Recall=0.9940, QPS=12744.5 (ef_search=16)
    k=10: MRR=1.0000, Recall=0.9818, QPS=13026.7 (ef_search=16)

TRIPCLICK:
  Dataset: 152387 vectors, 768D
  HNSW: M=128, ef_construction=160
  Build time: 217.10s
  Best results by k:
    k=1: MRR=0.8077, Recall=0.2502, QPS=1149.5 (ef_search=72)
    k=10: MRR=0.8750, Recall=0.4926, QPS=1191.5 (ef_search=72)