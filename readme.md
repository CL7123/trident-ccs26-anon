## Basicquery-vector neighbor-neighbor list
## /home/anonymous/Test-Trident/src/domain_config.py
Includes all dataset configurations and function calls

 - Supported datasets: siftsmall, laion, tripclick, ms_marco
 - Configuration includes: vector dimension、number of documents、HNSWparameters(M, efConstruction, efSearch)、domain parameters
 - Usage: config = get_config("siftsmall")

## /home/anonymous/Test-Trident/src/index-builder.py
python src/index-builder.py --dataset siftsmall use this format to build index with different dataset configurations

 - Default input path:
  /home/anonymous/Test-Trident/dataset/{dataset}/base.fvecs
  - Default output path: /home/anonymous/Test-Trident/dataset/{dataset}/

## /home/anonymous/Test-Trident/src/searcher.py

python src/searcher.py --dataset siftsmall 

🔍 Input files (lines 325-328)

  1. Query vector file:
  {base_path}/query.fvecs
  Example: /home/anonymous/Test-Trident/dataset/siftsmall/query.fvecs

  2. Ground Truthfile:
  {base_path}/gt.ivecs  
  Example: /home/anonymous/Test-Trident/dataset/siftsmall/gt.ivecs

  3. Index node file (index-builder.py generated):
  {base_path}/nodes.bin
  Example: /home/anonymous/Test-Trident/dataset/siftsmall/nodes.bin

  4. index[CN]file (index-builder.py generated):
  {base_path}/neighbors.bin
  Example: /home/anonymous/Test-Trident/dataset/siftsmall/neighbors.bin

 - outputfile: /home/anonymous/Test-Trident/result.md ([CN]result)

## /home/anonymous/Test-Trident/src/share_data.py
[CN]HNSWindex[CN]，generatedMPC[CN]

python src/share_data.py --dataset siftsmall

 - Input files:
   - {dataset}/nodes.bin ([CN]vector)
   - {dataset}/neighbors.bin ([CN])
 - output[CN]: /home/anonymous/Test-Trident/dataset/{dataset}/
   - server_1/nodes_shares.npy, neighbors_shares.npy, metadata.json
   - server_2/nodes_shares.npy, neighbors_shares.npy, metadata.json
   - server_3/nodes_shares.npy, neighbors_shares.npy, metadata.json

## /home/anonymous/Test-Trident/src/basic_functionalities.py
(2,3)-Shamir[CN]MPC[CN]

 - [CN]: MPC23SSS
 - [CN]:
   - share_secret(): generated[CN]
   - reconstruct(): [CN]
   - Open(): [CN]
 - [CN]MPC[CN]: F_Rand, F_Zero, F_Mult, F_SoP, F_CheckZero

## /home/anonymous/Test-Trident/src/dpf_wrapper.py
VDPF (vectorDPF) [CN]，[CN]index[CN]

 - [CN]: VDPFVectorWrapper
 - [CN]:
   - generate_keys(): generatedDPF[CN]
   - evaluate_vector(): [CN]vector[CN]
   - [CN]

## /home/anonymous/Test-Trident/src/secure_multiplication.py
[CN]server[CN]，[CN]

 - [CN]: NumpyMultiplicationServer
 - [CN]:
   - [CN]Beaver[CN]
   - [CN]vector[CN]
   - [CN]

## /home/anonymous/Test-Trident/src/config.md
HNSWparametersconfigure[CN]，[CN]dataset[CN]parameters

```
dataset dim docs queries efsearch efconstruction layer M
laion 512 100000 1000 32 80 2 64
siftsmall 128 10000 100 32 80 2 64
TripClick 768 1,523,871 1175 36 160 2 128
MS MARCO 768 8,841,823 6980 48 200 2 128
```