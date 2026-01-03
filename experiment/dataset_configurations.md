# Dataset Configuration Parameters Comparison Table

Based on the configuration in `/home/anonymous/Test-Trident/src/domain_config.py`, the parameters for the four datasets are as follows:

## Basic Parameters Comparison

| Parameter | SIFTSMALL | LAION | TRIPCLICK | NFCORPUS |
|-----------|-----------|--------|-----------|-----------|
| **Description** | SIFT Small Dataset | LAION Dataset | TripClick Dataset | NFCorpus Dataset (Biomedical Domain) |
| **Document Count** | 10,000 | 100,000 | 1,523,871 | 3,633 |
| **Query Count** | 100 | 1,000 | 1,175 | 323 |
| **Vector Dimension** | 128 | 512 | 768 | 768 |

## Domain Parameter Configuration

| Parameter | SIFTSMALL | LAION | TRIPCLICK | NFCORPUS |
|-----------|-----------|--------|-----------|-----------|
| **domain_bits** | 16 | 19 | 21 | 15 |
| **domain_size** | 65,536 | 524,288 | 2,097,152 | 32,768 |
| **Supported Neighbor Lists** | 65k | 524k | 2M | 32k |
| **Prime Field** | 2³¹ - 1 | 2³¹ - 1 | 2³¹ - 1 | 2³¹ - 1 |
| **Output Bits** | 31 | 31 | 31 | 31 |
| **Security Parameter (κ)** | 31 | 31 | 31 | 31 |

## HNSW Index Parameters

| Parameter | SIFTSMALL | LAION | TRIPCLICK | NFCORPUS |
|-----------|-----------|--------|-----------|-----------|
| **M** | 64 | 64 | 128 | 32 |
| **efConstruction** | 80 | 80 | 160 | 80 |
| **efSearch** | 32 | 32 | 36 | 32 |
| **layer** | 2 | 2 | 2 | 2 |

## Data Scale Analysis

| Dataset | Document Count | Neighbor List Entries (Docs × 3 Layers) | domain_size | Capacity Utilization |
|---------|--------|---------------------------|-------------|------------|
| SIFTSMALL | 10,000 | 30,000 | 65,536 | 45.8% |
| LAION | 100,000 | 300,000 | 524,288 | 57.2% |
| TRIPCLICK | 1,523,871 | 4,571,613 | 2,097,152 | 218.0%* |
| NFCORPUS | 3,633 | 10,899 | 32,768 | 33.3% |
