source activate vllm-py312
cd /home/rickychen/Desktop/llm/llm-infer-scheduler
python Worker.py --model /home/rickychen/Desktop/llm/models/Llama-3.2-Infinirc-11B-Vision-Instruct --node-id worker5090 --max-concurrent 3