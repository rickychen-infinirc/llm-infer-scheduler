# llm-infer-scheduler


python Coordinator.py

# 節點1 (192.168.1.151)
python Worker.py --host 192.168.1.151 --node-id worker-1 --model /home/hpc/llm/models/output/Llama-3.2-Infinirc-1B-Instruct

# 節點2 (192.168.1.153)
python Worker.py --host 192.168.1.151 --node-id worker-2 --model /home/hpc/llm/models/output/Llama-3.2-Infinirc-1B-Instruct

python Client.py --host 192.168.1.151
