## 最基本的query-vector neighbor-neighbor list
## /home/anonymous/Test-Trident/src/domain_config.py
包括数据集的所有配置以及调用函数

 - 支持的数据集: siftsmall, laion, tripclick, ms_marco
 - 配置内容: 向量维度、文档数、HNSW参数(M, efConstruction, efSearch)、域参数
 - 使用方式: config = get_config("siftsmall")

## /home/anonymous/Test-Trident/src/index-builder.py
python src/index-builder.py --dataset siftsmall 这种格式选择不同数据集的不同配置的索引参数进行构建

 - 默认输入路径:
  /home/anonymous/Test-Trident/dataset/{dataset}/base.fvecs
  - 默认输出路径: /home/anonymous/Test-Trident/dataset/{dataset}/

## /home/anonymous/Test-Trident/src/searcher.py

python src/searcher.py --dataset siftsmall 

🔍 输入文件 (lines 325-328)

  1. 查询向量文件:
  {base_path}/query.fvecs
  例如: /home/anonymous/Test-Trident/dataset/siftsmall/query.fvecs

  2. Ground Truth文件:
  {base_path}/gt.ivecs  
  例如: /home/anonymous/Test-Trident/dataset/siftsmall/gt.ivecs

  3. 索引节点文件 (index-builder.py 生成):
  {base_path}/nodes.bin
  例如: /home/anonymous/Test-Trident/dataset/siftsmall/nodes.bin

  4. 索引邻居文件 (index-builder.py 生成):
  {base_path}/neighbors.bin
  例如: /home/anonymous/Test-Trident/dataset/siftsmall/neighbors.bin

 - 输出文件: /home/anonymous/Test-Trident/result.md (搜索结果)

## /home/anonymous/Test-Trident/src/share_data.py
将HNSW索引数据进行秘密共享，生成MPC所需的三方份额

python src/share_data.py --dataset siftsmall

 - 输入文件:
   - {dataset}/nodes.bin (节点向量)
   - {dataset}/neighbors.bin (邻居关系)
 - 输出目录: /home/anonymous/Test-Trident/dataset/{dataset}/
   - server_1/nodes_shares.npy, neighbors_shares.npy, metadata.json
   - server_2/nodes_shares.npy, neighbors_shares.npy, metadata.json
   - server_3/nodes_shares.npy, neighbors_shares.npy, metadata.json

## /home/anonymous/Test-Trident/src/basic_functionalities.py
(2,3)-Shamir秘密共享的MPC基础功能实现

 - 核心类: MPC23SSS
 - 主要功能:
   - share_secret(): 生成秘密份额
   - reconstruct(): 重构秘密
   - Open(): 公开重构带恶意检测
 - 已注释的MPC协议: F_Rand, F_Zero, F_Mult, F_SoP, F_CheckZero

## /home/anonymous/Test-Trident/src/dpf_wrapper.py
VDPF (向量DPF) 包装器，用于隐私保护的索引访问

 - 核心类: VDPFVectorWrapper
 - 主要功能:
   - generate_keys(): 生成DPF密钥
   - evaluate_vector(): 评估向量值
   - 支持批量处理和缓存优化

## /home/anonymous/Test-Trident/src/secure_multiplication.py
安全乘法服务器实现，用于多方计算中的乘法操作

 - 核心类: NumpyMultiplicationServer
 - 主要功能:
   - 使用Beaver三元组实现安全乘法
   - 支持向量和矩阵乘法
   - 处理部分重构和通信

## /home/anonymous/Test-Trident/src/config.md
HNSW参数配置表，列出各数据集的详细参数

```
dataset dim docs queries efsearch efconstruction layer M
laion 512 100000 1000 32 80 2 64
siftsmall 128 10000 100 32 80 2 64
TripClick 768 1,523,871 1175 36 160 2 128
MS MARCO 768 8,841,823 6980 48 200 2 128
```