## [CN]basic[CN]query-vector neighbor-neighbor list
## /home/anonymous/Test-Trident/src/domain_config.py
package[CN]dataset[CN]configuration[CN]function

 - support[CN]dataset: siftsmall, laion, tripclick, ms_marco
 - configurationcontent: vectordimension, number of documents, HNSWparameters(M, efConstruction, efSearch), [CN]parameters
 - usage[CN]: config = get_config("siftsmall")

## /home/anonymous/Test-Trident/src/index-builder.py
python src/index-builder.py --dataset siftsmall [CN]formatselect[CN]dataset[CN]configuration[CN]indexparameters[CN]rowbuild

 - defaultinputpath:
  /home/anonymous/Test-Trident/dataset/{dataset}/base.fvecs
  - defaultoutputpath: /home/anonymous/Test-Trident/dataset/{dataset}/

## /home/anonymous/Test-Trident/src/searcher.py

python src/searcher.py --dataset siftsmall 

🔍 inputfile (lines 325-328)

  1. queryvectorfile:
  {base_path}/query.fvecs
  [CN]: /home/anonymous/Test-Trident/dataset/siftsmall/query.fvecs

  2. Ground Truthfile:
  {base_path}/gt.ivecs  
  [CN]: /home/anonymous/Test-Trident/dataset/siftsmall/gt.ivecs

  3. indexnodefile (index-builder.py generate):
  {base_path}/nodes.bin
  [CN]: /home/anonymous/Test-Trident/dataset/siftsmall/nodes.bin

  4. indexneighborfile (index-builder.py generate):
  {base_path}/neighbors.bin
  [CN]: /home/anonymous/Test-Trident/dataset/siftsmall/neighbors.bin

 - outputfile: /home/anonymous/Test-Trident/result.md (searchresult)

## /home/anonymous/Test-Trident/src/share_data.py
[CN]HNSWindexdata[CN]row[CN]share,generateMPC[CN]

python src/share_data.py --dataset siftsmall

 - inputfile:
   - {dataset}/nodes.bin (nodevector)
   - {dataset}/neighbors.bin (neighborrelation)
 - outputdirectory: /home/anonymous/Test-Trident/dataset/{dataset}/
   - server_1/nodes_shares.npy, neighbors_shares.npy, metadata.json
   - server_2/nodes_shares.npy, neighbors_shares.npy, metadata.json
   - server_3/nodes_shares.npy, neighbors_shares.npy, metadata.json

## /home/anonymous/Test-Trident/src/basic_functionalities.py
(2,3)-Shamir[CN]share[CN]MPC[CN]functionalityimplementation

 - [CN]class: MPC23SSS
 - [CN]functionality:
   - share_secret(): generate[CN]
   - reconstruct(): refactoring[CN]
   - Open(): [CN]refactoring[CN]detection
 - [CN]comment[CN]MPCprotocol: F_Rand, F_Zero, F_Mult, F_SoP, F_CheckZero

## /home/anonymous/Test-Trident/src/dpf_wrapper.py
VDPF (vectorDPF) wrapper[CN],[CN]protection[CN]indexvisit

 - [CN]class: VDPFVectorWrapper
 - [CN]functionality:
   - generate_keys(): generateDPFkey
   - evaluate_vector(): assessvectorvalue
   - supportbatch processing[CN]cacheoptimization

## /home/anonymous/Test-Trident/src/secure_multiplication.py
securitymultiplicationserverimplementation,[CN]computationin[CN]multiplication[CN]

 - [CN]class: NumpyMultiplicationServer
 - [CN]functionality:
   - usageBeaver[CN]tupleimplementationsecuritymultiplication
   - support vector[CN]multiplication
   - handle[CN]minuterefactoring[CN]

## /home/anonymous/Test-Trident/src/config.md
HNSWparametersconfiguration[CN],column[CN]dataset[CN]detailedparameters

```
dataset dim docs queries efsearch efconstruction layer M
laion 512 100000 1000 32 80 2 64
siftsmall 128 10000 100 32 80 2 64
TripClick 768 1,523,871 1175 36 160 2 128
MS MARCO 768 8,841,823 6980 48 200 2 128
```