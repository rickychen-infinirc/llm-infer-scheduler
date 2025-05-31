# llm-infer-scheduler


python Coordinator.py

# 節點1 (192.168.1.151)
python Worker.py --host 192.168.201.10 --node-id worker-1 --model /home/rickychen/Desktop/llm/models/Llama-3.2-Infinirc-3B-Instruct

# 節點2 (192.168.1.153)
python Worker.py --host 192.168.201.10 --node-id worker-2 --model /home/rickychen/Desktop/llm/models/Llama-3.2-Infinirc-3B-Instruct

python Client.py --host 192.168.201.10


# 在 GPU 1 上運行
python Worker.py --host 192.168.201.10 --node-id worker-gpu1 --gpu 1 --model /home/rickychen/Desktop/llm/models/Llama-3.2-Infinirc-3B-Instruct

# 或者在 GPU 0 上運行
python Worker.py --host 192.168.201.10 --node-id worker-gpu0 --gpu 0 --model /home/rickychen/Desktop/llm/models/Llama-3.2-Infinirc-3B-Instruct


自動發現
# 完全自動，不需要指定 IP
python Worker.py --model /home/rickychen/Desktop/llm/models/Llama-3.2-Infinirc-3B-Instruct --gpu 0

# 也可以指定 GPU 和節點名稱
python Worker.py --model /home/rickychen/Desktop/llm/models/Llama-3.2-Infinirc-3B-Instruct --gpu 1 --node-id worker-gpu1

python Worker.py --model /home/rickychen/Desktop/llm/models/Llama-3.2-Infinirc-3B-Instruct --gpu 0 --node-id worker-gpu1

# 如果自動發現失敗，仍可手動指定
python Worker.py --model /home/rickychen/Desktop/llm/models/Llama-3.2-Infinirc-3B-Instruct --host 192.168.201.10 --gpu 0
