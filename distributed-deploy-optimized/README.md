# [CN]deploy[CN]

[CN]deploy Trident [CN]file。

## [CN]

```
distributed-deploy/
├── server.py          # [CN]server[CN]
├── client.py          # [CN]client[CN]
├── config.py          # [CN]configurefile
├── deploy.sh          # [CN]deploy[CN]
└── README.md          # [CN]
```

## [CN]

1. **[CN]server**：[CN]3[CN]server（[CN]AWS EC2[CN]）
2. **Python[CN]**：[CN]server[CN]Python 3.8+
3. **SSH[CN]**：[CN]SSH[CN]server
4. **[CN]configure**：[CN] 8001-8003

## [CN]

### 1. configureserver[CN]

[CN] `config.py` file，[CN]serverIP[CN]：

```python
SERVERS = {
    1: {"host": "192.168.1.101", "port": 8001},
    2: {"host": "YOUR_SERVER_2_IP", "port": 8002},
    3: {"host": "YOUR_SERVER_3_IP", "port": 8003}
}
```

### 2. configuredeploy[CN]

[CN] `deploy.sh` file，[CN]serverIP：

```bash
SERVERS[1]="192.168.1.101"
SERVERS[2]="YOUR_SERVER_2_IP"
SERVERS[3]="YOUR_SERVER_3_IP"
```

### 3. deployserver

[CN]deploy[CN]deploy：

```bash
./deploy.sh
```

[CN] 1 [CN]deploy[CN]start[CN]server。

### 4. runclienttest

[CN]runclient：

```bash
cd ~/trident/distributed-deploy
python3 client.py --dataset siftsmall --num-queries 10
```

## [CN]deploy[CN]

[CN]deploy，[CN]：

### [CN]server[CN]：

1. **[CN]**：
```bash
rsync -avz -e "ssh -i your-key.pem" --exclude='venv/' ./ ubuntu@SERVER_IP:~/test/
```

2. **startserver**（[CN]server[CN]）：
```bash
# server1
python3 server.py --server-id 1 --dataset siftsmall

# server2
python3 server.py --server-id 2 --dataset siftsmall

# server3
python3 server.py --server-id 3 --dataset siftsmall
```

### [CN]client[CN]：

```bash
python3 client.py --dataset siftsmall --num-queries 10
```

## [CN]parameters

### serverparameters

- `--server-id`: serverID (1, 2, [CN] 3)
- `--dataset`: dataset[CN] (siftsmall, laion, tripclick, ms_marco, nfcorpus)
- `--vdpf-processes`: VDPF[CN] ([CN]: 4)

### clientparameters

- `--dataset`: dataset[CN] ([CN]: siftsmall)
- `--num-queries`: testquery[CN] ([CN]: 10)
- `--no-report`: [CN]test[CN]
- `--config`: [CN]configurefilepath
- `--status-only`: [CN]server[CN]

## [CN]

### AWS EC2 [CN]configure

1. [CN]AWS[CN]
2. [CN]EC2 -> [CN] -> [CN] -> [CN] -> [CN]
3. [CN]，[CN]：
   - [CN]: [CN]TCP
   - [CN]: 8001-8003
   - [CN]: 0.0.0.0/0 ([CN]IP)

### [CN]configure（[CN]）

```bash
# Ubuntu/Debian
sudo ufw allow 8001:8003/tcp

# CentOS/RHEL
sudo firewall-cmd --permanent --add-port=8001-8003/tcp
sudo firewall-cmd --reload
```

## [CN]

### 1. [CN]

- [CN]/[CN]configure
- [CN]serverIP[CN]
- [CN]server[CN]run

### 2. SSH[CN]

- [CN]file[CN]：`chmod 400 your-key.pem`
- [CN]（ubuntu/ec2-user）
- [CN]SSH[CN]

### 3. [CN]start[CN]

- [CN]Python[CN]install[CN]
- [CN]server[CN]：`tail -f server_X.log`
- [CN]file[CN]

### 4. query[CN]

- [CN]2[CN]server[CN]run
- [CN]
- [CN]dataset[CN]

## [CN]

1. **[CN]**：
   - [CN]serverdeploy[CN]（AWS Region）
   - [CN]IP[CN]（[CN]）
   - [CN]TCP_NODELAY[CN]

2. **[CN]configure**：
   - [CN]CPU[CN] `--vdpf-processes`
   - [CN]CPU[CN]

3. **[CN]**：
   - [CN]SSD[CN]file
   - [CN]start[CN]

## [CN]

- server[CN]：`~/test/distributed-deploy/server_X.log`
- clienttest[CN]：`~/test/distributed-deploy/distributed_result.md`

[CN]deploy[CN]：
```bash
./deploy.sh
# [CN] 7
```

## [CN]

1. **[CN]**：
   - [CN]TLS[CN]
   - [CN]IP[CN]
   - [CN]

2. **[CN]**：
   - [CN]SSH[CN]
   - [CN]

## [CN]

[CN]，[CN]：
1. server[CN]file
2. [CN]
3. [CN]configure

[CN]。