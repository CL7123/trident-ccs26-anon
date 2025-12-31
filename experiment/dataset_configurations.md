# datasetconfigureparameters[CN]

[CN] `/home/anonymous/Test-Trident/src/domain_config.py` [CN]configure，[CN]dataset[CN]parameters[CN]：

## [CN]parameters[CN]

| parameters | SIFTSMALL | LAION | TRIPCLICK | NFCORPUS |
|------|-----------|--------|-----------|-----------|
| **[CN]** | SIFT Smalldataset | LAIONdataset | TripClickdataset | NFCorpusdataset（[CN]） |
| **number of documents[CN]** | 10,000 | 100,000 | 1,523,871 | 3,633 |
| **query[CN]** | 100 | 1,000 | 1,175 | 323 |
| **vector dimension** | 128 | 512 | 768 | 768 |

## domain parametersconfigure

| parameters | SIFTSMALL | LAION | TRIPCLICK | NFCORPUS |
|------|-----------|--------|-----------|-----------|
| **domain_bits** | 16 | 19 | 21 | 15 |
| **domain_size** | 65,536 | 524,288 | 2,097,152 | 32,768 |
| **[CN]** | 65k | 524k | 2M | 32k |
| **[CN] (prime)** | 2³¹ - 1 | 2³¹ - 1 | 2³¹ - 1 | 2³¹ - 1 |
| **output[CN] (output_bits)** | 31 | 31 | 31 | 31 |
| **[CN]parameters (κ)** | 31 | 31 | 31 | 31 |

## HNSW indexparameters

| parameters | SIFTSMALL | LAION | TRIPCLICK | NFCORPUS |
|------|-----------|--------|-----------|-----------|
| **M** | 64 | 64 | 128 | 32 |
| **efConstruction** | 80 | 80 | 160 | 80 |
| **efSearch** | 32 | 32 | 36 | 32 |
| **layer** | 2 | 2 | 2 | 2 |

## [CN]

| dataset | number of documents | [CN] (number of documents×3[CN]) | domain_size | [CN] |
|--------|--------|---------------------------|-------------|------------|
| SIFTSMALL | 10,000 | 30,000 | 65,536 | 45.8% |
| LAION | 100,000 | 300,000 | 524,288 | 57.2% |
| TRIPCLICK | 1,523,871 | 4,571,613 | 2,097,152 | 218.0%* |
| NFCORPUS | 3,633 | 10,899 | 32,768 | 33.3% |
