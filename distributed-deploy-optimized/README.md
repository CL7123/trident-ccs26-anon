# distributeddeployment[CN]

[CN]directorypackage[CN]true[CN]networkenvironmentindeployment Trident system[CN]file.

## directory[CN]

```
distributed-deploy/
├── server.py          # distributedservercode
├── client.py          # distributedclientcode
├── config.py          # networkprofile
├── deploy.sh          # automatic[CN]deployment[CN]
└── README.md          # [CN]documentation
```

## before[CN]require

1. **[CN]server**:[CN]3[CN]server([CN]usageAWS EC2[CN])
2. **Pythonenvironment**:[CN]serveron[CN]Python 3.8+
3. **SSHvisit**:[CN]SSHvisit[CN]server
4. **firewallconfiguration**:[CN]port 8001-8003

## faststart

### 1. configurationserverinformation

[CN] `config.py` file,[CN]serverIPaddress:

```python
SERVERS = {
    1: {"host": "192.168.1.101", "port": 8001},
    2: {"host": "YOUR_SERVER_2_IP", "port": 8002},
    3: {"host": "YOUR_SERVER_3_IP", "port": 8003}
}
```

### 2. configurationdeployment[CN]

[CN] `deploy.sh` file,updateserverIP:

```bash
SERVERS[1]="192.168.1.101"
SERVERS[2]="YOUR_SERVER_2_IP"
SERVERS[3]="YOUR_SERVER_3_IP"
```

### 3. deploymentserver

usagedeployment[CN]keydeployment:

```bash
./deploy.sh
```

selectoptions 1 [CN]deployment[CN]start[CN]server.

### 4. runclienttest

[CN]onrunclient:

```bash
cd ~/trident/distributed-deploy
python3 client.py --dataset siftsmall --num-queries 10
```

## manualdeployment[CN]

[CN]manualdeployment,[CN]under[CN]:

### [CN]serveron:

1. **synchronouscode**:
```bash
rsync -avz -e "ssh -i your-key.pem" --exclude='venv/' ./ ubuntu@SERVER_IP:~/test/
```

2. **startserver**([CN]serveron):
```bash
# server1
python3 server.py --server-id 1 --dataset siftsmall

# server2
python3 server.py --server-id 2 --dataset siftsmall

# server3
python3 server.py --server-id 3 --dataset siftsmall
```

### [CN]client[CN]on:

```bash
python3 client.py --dataset siftsmall --num-queries 10
```

## command-line argument

### serverparameters

- `--server-id`: serverID (1, 2, [CN] 3)
- `--dataset`: datasetname (siftsmall, laion, tripclick, ms_marco, nfcorpus)
- `--vdpf-processes`: VDPFassessprocess[CN] (default: 4)

### clientparameters

- `--dataset`: datasetname (default: siftsmall)
- `--num-queries`: testqueryquantity (default: 10)
- `--no-report`: [CN]savetestreport
- `--config`: customprofilepath
- `--status-only`: [CN]fetchserverstate

## networkrequire

### AWS EC2 security[CN]configuration

1. [CN]AWS[CN]
2. enterEC2 -> [CN] -> select[CN] -> security -> security[CN]
3. [CN],add:
   - type: customTCP
   - portrange: 8001-8003
   - [CN]: 0.0.0.0/0 ([CN]limit[CN]IP)

### localfirewallconfiguration([CN])

```bash
# Ubuntu/Debian
sudo ufw allow 8001:8003/tcp

# CentOS/RHEL
sudo firewall-cmd --permanent --add-port=8001-8003/tcp
sudo firewall-cmd --reload
```

## [CN]

### 1. connecttimeout

- checkfirewall/security[CN]configuration
- verificationserverIPaddressyesnocorrect
- [CN]serverpositive[CN]run

### 2. SSHconnectfailure

- checkkeyfilepermission:`chmod 400 your-key.pem`
- verificationusername(ubuntu/ec2-user)
- checkSSHportyesno[CN]

### 3. servicestartfailure

- checkPythondependentyesnoinstall[CN]
- [CN]serverlog:`tail -f server_X.log`
- [CN]datafile[CN]correctposition

### 4. queryfailure

- [CN]2[CN]serverpositive[CN]run
- checknetworkconnectyesnostable
- verificationdatasetyesnocorrectload

## performanceoptimizationsuggestion

1. **networkoptimization**:
   - [CN]serverdeployment[CN]locale(AWS Region)
   - usageinside[CN]IP[CN]row[CN]([CN])
   - enableTCP_NODELAY[CN]late

2. **processconfiguration**:
   - root[CN]CPU[CN] `--vdpf-processes`
   - monitorCPU[CN]memoryusage[CN]

3. **dataoptimization**:
   - usageSSD[CN]datafile
   - [CN]processpool[CN]startlate

## monitor[CN]log

- serverlogposition:`~/test/distributed-deploy/server_X.log`
- clienttestreport:`~/test/distributed-deploy/distributed_result.md`

usagedeployment[CN]log:
```bash
./deploy.sh
# selectoptions 7
```

## securitynote[CN]item

1. **production environmentsuggestion**:
   - usageTLSencryption[CN]
   - limitIPvisitrange
   - [CN]updatesystem[CN]dependent

2. **key[CN]**:
   - [CN]SSHkey
   - [CN]keycommit[CN]version control

## [CN]support

[CN]issue,[CN]check:
1. serverlogfile
2. networkconnectstate
3. firewallconfiguration

detailed[CN]errorinformation[CN]fast[CN]bit[CN]resolveissue.