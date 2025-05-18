import socket
import json
import threading
import time
import logging
import sys
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("LLM-Worker")

class LLMWorker:
    def __init__(self, coordinator_host, coordinator_port=9000, 
                 model_path="/home/hpc/llm/models/output/Llama-3.2-Infinirc-1B-Instruct", 
                 node_id="worker-1"):
        self.coordinator_host = coordinator_host
        self.coordinator_port = coordinator_port
        self.model_path = model_path
        self.node_id = node_id
        
        # 連接到協調器
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        # 載入LLM模型
        logger.info(f"Loading model from {model_path}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # 檢查模型路徑是否存在
        if not os.path.exists(model_path):
            logger.error(f"Model path does not exist: {model_path}")
            raise FileNotFoundError(f"Model path not found: {model_path}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto"
            )
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def connect_to_coordinator(self):
        """連接到協調器並註冊"""
        try:
            self.socket.connect((self.coordinator_host, self.coordinator_port))
            # 發送工作節點標記
            self.socket.sendall(b'W')
            
            # 發送註冊信息
            worker_info = {
                'node_id': self.node_id,
                'model': self.model_path,
                'device': str(self.device),
                'max_length': 2048
            }
            self.socket.sendall(json.dumps(worker_info).encode('utf-8') + b'\n')
            logger.info(f"Sent registration info to coordinator")
            
            # 接收確認信息
            data = b''
            while b'\n' not in data:
                chunk = self.socket.recv(4096)
                if not chunk:
                    logger.error("Connection closed by coordinator")
                    return False
                data += chunk
            
            response = json.loads(data.split(b'\n')[0].decode('utf-8'))
            logger.info(f"Registered with coordinator as worker_id: {response.get('worker_id')}")
            return True
        except Exception as e:
            logger.error(f"Error connecting to coordinator: {e}")
            return False

    def handle_heartbeat(self):
        """處理心跳檢測"""
        while True:
            try:
                data = self.socket.recv(10)
                if not data:
                    logger.warning("Coordinator disconnected")
                    break
                
                if data == b'ping\n':
                    self.socket.sendall(b'pong\n')
                    logger.debug("Heartbeat: pong sent")
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                break
            time.sleep(1)
        
        # 重新連接
        self.reconnect()

    def handle_requests(self):
        """處理來自協調器的請求"""
        buffer = b''
        while True:
            try:
                chunk = self.socket.recv(4096)
                if not chunk:
                    logger.warning("Coordinator disconnected during request handling")
                    break
                
                buffer += chunk
                
                # 檢查是否有完整的消息（以換行符結尾）
                if b'\n' in buffer:
                    messages = buffer.split(b'\n')
                    for i in range(len(messages) - 1):
                        # 處理完整的消息
                        if messages[i]:
                            try:
                                request = json.loads(messages[i].decode('utf-8'))
                                logger.info(f"Received request: {request.get('data', {}).get('prompt', '')[:30]}...")
                                Thread(target=self.process_request, args=(request,)).start()
                            except json.JSONDecodeError as e:
                                logger.error(f"JSON decode error: {e}, message: {messages[i][:100]}")
                    
                    # 保存剩餘的不完整消息
                    buffer = messages[-1]
            except Exception as e:
                logger.error(f"Error receiving request: {e}")
                break
        
        # 重新連接
        self.reconnect()

    def get_client_connection(self, client_id, client_addr):
        """獲取與客戶端的直接連接"""
        try:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # 使用協調器的IP和端口來建立連接
            client_socket.connect((self.coordinator_host, self.coordinator_port))
            
            # 標識為工作節點到客戶端的連接
            client_socket.sendall(b'O')
            
            # 發送連接信息
            conn_info = {
                'worker_id': self.node_id,
                'client_id': client_id,
                'client_addr': client_addr
            }
            client_socket.sendall(json.dumps(conn_info).encode('utf-8') + b'\n')
            
            # 接收確認
            response = client_socket.recv(1024)
            if response:
                confirmation = json.loads(response.decode('utf-8'))
                if confirmation.get('status') == 'connected':
                    logger.info(f"Direct connection to client {client_id} established")
                    return client_socket
            
            client_socket.close()
            return None
        except Exception as e:
            logger.error(f"Error creating client connection: {e}")
            return None

    def process_request(self, request):
        """處理單個請求"""
        try:
            # 解析請求
            client_id = request.get('client_id')
            client_addr = request.get('client_addr')
            data = request.get('data', {})
            
            prompt = data.get('prompt', '')
            max_length = data.get('max_length', 1024)
            temperature = data.get('temperature', 0.7)
            
            logger.info(f"Processing request for client {client_id}: prompt length {len(prompt)}")
            
            # 向協調器的socket發送流式響應
            try:
                # 將任務狀態發送給客戶端
                start_msg = {
                    'type': 'start',
                    'message': 'Starting generation...'
                }
                self.socket.sendall(json.dumps(start_msg).encode('utf-8') + b'\n')
                
                # 設置流式生成
                streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
                
                # 記錄推理開始時間
                start_time = time.time()
                
                # 準備輸入
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                
                # 在單獨線程中生成文本
                generation_kwargs = {
                    'input_ids': inputs.input_ids,
                    'max_new_tokens': max_length,
                    'temperature': temperature,
                    'top_p': 0.9,
                    'streamer': streamer,
                }
                
                thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
                thread.start()
                
                # 流式發送生成的token
                generated_text = ""
                for new_text in streamer:
                    generated_text += new_text
                    token_msg = {
                        'type': 'token',
                        'token': new_text
                    }
                    self.socket.sendall(json.dumps(token_msg).encode('utf-8') + b'\n')
                    # 小延遲以避免網絡堵塞
                    time.sleep(0.01)
                
                # 發送完成消息
                inference_time = time.time() - start_time
                end_msg = {
                    'type': 'end',
                    'message': 'Generation complete',
                    'inference_time': inference_time,
                    'total_length': len(generated_text)
                }
                self.socket.sendall(json.dumps(end_msg).encode('utf-8') + b'\n')
                
                logger.info(f"Streaming completed for client {client_id}: {len(generated_text)} chars, {inference_time:.2f}s")
            
            except Exception as e:
                logger.error(f"Error streaming response: {e}")
                # 發送錯誤信息
                error_msg = {
                    'type': 'error',
                    'error': str(e)
                }
                self.socket.sendall(json.dumps(error_msg).encode('utf-8') + b'\n')
            
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            # 發送錯誤信息
            try:
                error_msg = {
                    'type': 'error',
                    'error': str(e)
                }
                self.socket.sendall(json.dumps(error_msg).encode('utf-8') + b'\n')
            except:
                pass

    def reconnect(self):
        """重新連接到協調器"""
        max_retries = 10
        retry_delay = 5  # 初始延遲5秒
        
        for attempt in range(max_retries):
            logger.info(f"Attempting to reconnect to coordinator (attempt {attempt+1}/{max_retries})...")
            try:
                self.socket.close()
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                if self.connect_to_coordinator():
                    logger.info("Reconnected to coordinator")
                    # 啟動心跳和請求處理線程
                    threading.Thread(target=self.handle_heartbeat, daemon=True).start()
                    threading.Thread(target=self.handle_requests, daemon=True).start()
                    return True
            except Exception as e:
                logger.error(f"Reconnection attempt failed: {e}")
            
            # 延遲重試，指數增長策略
            time.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, 60)  # 最大延遲60秒
        
        logger.error("Failed to reconnect after maximum attempts")
        return False

    def start(self):
        """啟動工作節點"""
        if self.connect_to_coordinator():
            # 啟動心跳檢測線程
            threading.Thread(target=self.handle_heartbeat, daemon=True).start()
            
            # 啟動請求處理線程
            threading.Thread(target=self.handle_requests, daemon=True).start()
            
            logger.info(f"Worker started, connected to coordinator {self.coordinator_host}:{self.coordinator_port}")
            
            # 保持程序運行
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Worker shutting down...")
            finally:
                self.socket.close()
        else:
            logger.error("Failed to connect to coordinator")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='LLM Worker Node')
    parser.add_argument('--host', type=str, default='localhost', help='Coordinator host')
    parser.add_argument('--port', type=int, default=9000, help='Coordinator port')
    parser.add_argument('--model', type=str, default='/home/hpc/llm/models/output/Llama-3.2-Infinirc-1B-Instruct', 
                        help='Model path')
    parser.add_argument('--node-id', type=str, default='worker-1', help='Node identifier')
    
    args = parser.parse_args()
    
    try:
        worker = LLMWorker(
            coordinator_host=args.host,
            coordinator_port=args.port,
            model_path=args.model,
            node_id=args.node_id
        )
        worker.start()
    except Exception as e:
        logger.error(f"Worker error: {e}")
