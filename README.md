# llm-infer-scheduler

```
python Coordinator.py
```


### 完全自動，不需要指定 IP
```
python Worker.py --model /home/rickychen/Desktop/llm/models/Llama-3.2-Infinirc-11B-Vision-Instruct --node-id worker5090 --max-concurrent 3
```

### 也可以指定 GPU 和節點名稱
```
python Worker.py --model /home/rickychen/Desktop/llm/models/Llama-3.2-Infinirc-11B-Vision-Instruct --gpu 1 --node-id worker-gpu0
```
```
python Worker.py --model /home/rickychen/Desktop/llm/models/Llama-3.2-Infinirc-3B-Instruct --gpu 0 --node-id worker-gpu1
```
### 如果自動發現失敗，仍可手動指定
```
python Worker.py --model /home/rickychen/Desktop/llm/models/Llama-3.2-Infinirc-3B-Instruct --host 192.168.201.10 --gpu 0
```


## 壓力測試

### 基本測試：3個客戶端，每個發送2個請求
```
python stress_test.py
```
### 更激進的測試：5個客戶端，每個發送3個請求
```
python stress_test.py --clients 5 --requests 2
```
### 快速連續測試：10個客戶端，每個發送1個請求，無延遲
```
python stress_test.py --clients 10 --requests 1 --delay 0
```

### 長時間測試：2個客戶端，每個發送10個請求，間隔0.5秒
```
python stress_test.py --clients 2 --requests 10 --delay 0.5
```
### 連接到遠端服務器
```
python stress_test.py --host 192.168.1.100 --port 9000 --clients 3 --requests 2
```