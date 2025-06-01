import socket
import json
import threading
import time
import logging
import sys
import os
import torch
import queue
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

# 日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("LLM-Worker")

class LLMWorker:
    def __init__(self, model_path, node_id="worker-auto", gpu_id=None, 
                 discovery_port=9001, coordinator_host=None, coordinator_port=9000,
                 max_concurrent=2):
        self.model_path = model_path
        self.node_id = node_id
        self.gpu_id = gpu_id
        self.discovery_port = discovery_port
        self.coordinator_host = coordinator_host
        self.coordinator_port = coordinator_port
        
        # 並行處理相關屬性
        self.max_concurrent = max_concurrent
        self.active_requests = 0
        self.active_requests_lock = threading.Lock()
        self.request_queue = queue.Queue()
        
        # 連接到協調器
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        # 檢查模型路徑是否存在
        if not os.path.exists(model_path):
            logger.error(f"Model path does not exist: {model_path}")
            raise FileNotFoundError(f"Model path not found: {model_path}")
        
        # 設置設備並載入模型
        self.setup_device_and_load_model()
        
        # 如果沒有指定協調器地址，進行自動發現
        if not self.coordinator_host:
            self.discover_coordinator()
        
        # 啟動多個並行處理線程
        for i in range(self.max_concurrent):
            worker_thread = threading.Thread(
                target=self.parallel_request_processor, 
                name=f"RequestProcessor-{i}",
                daemon=True
            )
            worker_thread.start()
            logger.info(f"Started request processor thread {i}")

    def parallel_request_processor(self):
        """並行處理請求的工作線程"""
        thread_name = threading.current_thread().name
        logger.info(f"{thread_name} started")
        
        while True:
            try:
                # 從隊列中取得請求
                request = self.request_queue.get(timeout=1)
                
                with self.active_requests_lock:
                    self.active_requests += 1
                    logger.info(f"{thread_name} processing request, active: {self.active_requests}/{self.max_concurrent}")
                
                try:
                    # 處理請求
                    self.process_single_request(request)
                except Exception as e:
                    logger.error(f"{thread_name} error processing request: {e}")
                finally:
                    with self.active_requests_lock:
                        self.active_requests -= 1
                        logger.info(f"{thread_name} completed request, active: {self.active_requests}/{self.max_concurrent}")
                    
                    self.request_queue.task_done()
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"{thread_name} unexpected error: {e}")

    def discover_coordinator(self):
        """自動發現控制器"""
        logger.info("Starting coordinator discovery...")
        
        # 創建發現 socket
        discovery_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        discovery_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        discovery_socket.settimeout(5.0)
        
        # 嘗試多種發現方式
        coordinators_found = []
        
        # 方式1: 發送廣播發現請求
        try:
            broadcast_addrs = ['255.255.255.255', '192.168.255.255', '10.255.255.255', '172.31.255.255']
            
            for broadcast_addr in broadcast_addrs:
                try:
                    logger.info(f"Broadcasting discovery request to {broadcast_addr}:{self.discovery_port}")
                    discovery_socket.sendto(b"DISCOVER_COORDINATOR", (broadcast_addr, self.discovery_port))
                except Exception as e:
                    logger.debug(f"Failed to broadcast to {broadcast_addr}: {e}")
            
            # 監聽回應
            start_time = time.time()
            while time.time() - start_time < 10:  # 等待10秒
                try:
                    data, addr = discovery_socket.recvfrom(1024)
                    response = json.loads(data.decode('utf-8'))
                    
                    if response.get('type') == 'COORDINATOR_INFO':
                        coordinator_info = {
                            'host': response['host'],
                            'port': response['port'],
                            'web_port': response.get('web_port'),
                            'timestamp': response.get('timestamp', 0),
                            'source_ip': addr[0]
                        }
                        coordinators_found.append(coordinator_info)
                        logger.info(f"Found coordinator at {response['host']}:{response['port']}")
                
                except socket.timeout:
                    continue
                except Exception as e:
                    logger.debug(f"Error receiving discovery response: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error in broadcast discovery: {e}")
        
        # 方式2: 監聽協調器廣播公告
        if not coordinators_found:
            logger.info("No response to discovery request, listening for coordinator announcements...")
            
            listen_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listen_socket.bind(('', self.discovery_port))
            listen_socket.settimeout(30.0)  # 監聽30秒
            
            try:
                start_time = time.time()
                while time.time() - start_time < 30:
                    try:
                        data, addr = listen_socket.recvfrom(1024)
                        announcement = json.loads(data.decode('utf-8'))
                        
                        if announcement.get('type') == 'COORDINATOR_ANNOUNCEMENT':
                            coordinator_info = {
                                'host': announcement['host'],
                                'port': announcement['port'],
                                'web_port': announcement.get('web_port'),
                                'timestamp': announcement.get('timestamp', 0),
                                'source_ip': addr[0]
                            }
                            coordinators_found.append(coordinator_info)
                            logger.info(f"Received announcement from coordinator at {announcement['host']}:{announcement['port']}")
                            break
                            
                    except socket.timeout:
                        continue
                    except Exception as e:
                        logger.debug(f"Error receiving announcement: {e}")
                        continue
                        
            except Exception as e:
                logger.error(f"Error listening for announcements: {e}")
            finally:
                listen_socket.close()
        
        # 方式3: 掃描常見 IP 範圍
        if not coordinators_found:
            logger.info("No coordinator found via broadcast, trying IP scan...")
            coordinators_found = self.scan_for_coordinators()
        
        discovery_socket.close()
        
        # 選擇最佳協調器
        if coordinators_found:
            coordinators_found.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
            best_coordinator = coordinators_found[0]
            
            self.coordinator_host = best_coordinator['host']
            self.coordinator_port = best_coordinator['port']
            
            logger.info(f"Selected coordinator: {self.coordinator_host}:{self.coordinator_port}")
            
            if len(coordinators_found) > 1:
                logger.info(f"Found {len(coordinators_found)} coordinators, selected the most recent one")
        else:
            logger.error("No coordinator found! Please ensure a coordinator is running on the network.")
            raise RuntimeError("Coordinator discovery failed")

    def scan_for_coordinators(self):
        """掃描本地網絡尋找協調器"""
        logger.info("Scanning local network for coordinators...")
        
        # 獲取本機 IP 來確定掃描範圍
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            
            # 計算網絡段
            ip_parts = local_ip.split('.')
            network_base = f"{ip_parts[0]}.{ip_parts[1]}.{ip_parts[2]}"
            
            logger.info(f"Scanning network {network_base}.1-254")
            
            coordinators_found = []
            
            # 使用多線程掃描以提高速度
            def scan_ip(ip):
                try:
                    test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    test_socket.settimeout(1.0)
                    result = test_socket.connect_ex((ip, self.coordinator_port))
                    test_socket.close()
                    
                    if result == 0:
                        logger.info(f"Found potential coordinator at {ip}:{self.coordinator_port}")
                        coordinators_found.append({
                            'host': ip,
                            'port': self.coordinator_port,
                            'timestamp': time.time(),
                            'source_ip': ip
                        })
                except:
                    pass
            
            # 並行掃描
            threads = []
            for i in range(1, 255):
                ip = f"{network_base}.{i}"
                if ip != local_ip:  # 跳過自己
                    thread = threading.Thread(target=scan_ip, args=(ip,))
                    threads.append(thread)
                    thread.start()
            
            # 等待所有掃描完成
            for thread in threads:
                thread.join()
            
            return coordinators_found
            
        except Exception as e:
            logger.error(f"Error in IP scanning: {e}")
            return []

    def setup_device_and_load_model(self):
        """設置設備並載入模型"""
        if not torch.cuda.is_available():
            logger.info("CUDA not available, using CPU")
            self.device = torch.device("cpu")
            self.gpu_id = None
        else:
            gpu_count = torch.cuda.device_count()
            logger.info(f"Found {gpu_count} GPU(s)")
            
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                logger.info(f"  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
            
            if self.gpu_id is not None:
                if self.gpu_id >= gpu_count or self.gpu_id < 0:
                    available_gpus = list(range(gpu_count))
                    logger.error(f"GPU {self.gpu_id} not available. Available GPUs: {available_gpus}")
                    raise ValueError(f"Invalid GPU ID: {self.gpu_id}")
                
                self.device = torch.device(f"cuda:{self.gpu_id}")
                gpu_name = torch.cuda.get_device_name(self.gpu_id)
                gpu_memory = torch.cuda.get_device_properties(self.gpu_id).total_memory / 1024**3
                logger.info(f"Selected GPU {self.gpu_id}: {gpu_name} ({gpu_memory:.1f} GB)")
            else:
                # 自動選擇第一個 GPU
                self.device = torch.device("cuda:0")
                self.gpu_id = 0
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                logger.info(f"Auto-selected GPU 0: {gpu_name} ({gpu_memory:.1f} GB)")
        
        # 載入模型
        logger.info(f"Loading model from {self.model_path}")
        logger.info(f"Target device: {self.device}")
        
        try:
            # 載入 tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            # 根據設備類型載入模型
            if self.device.type == 'cuda':
                logger.info("Loading model in FP16 for GPU...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True
                )
                # 明確移動模型到指定 GPU
                logger.info(f"Moving model to {self.device}...")
                self.model = self.model.to(self.device)
            else:
                logger.info("Loading model in FP32 for CPU...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True
                )
                self.model = self.model.to(self.device)
            
            logger.info("Model loaded successfully")
            
            # 顯示模型內存使用
            if self.device.type == 'cuda':
                memory_used = torch.cuda.memory_allocated(self.device) / 1024**3
                logger.info(f"Model loaded, GPU memory used: {memory_used:.2f} GB")
                
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
                'gpu_id': self.gpu_id,
                'max_length': 2048,
                'max_concurrent': self.max_concurrent,
                'active_requests': 0
            }
            
            # 如果有 GPU 信息，添加到註冊信息中
            if self.device.type == 'cuda' and self.gpu_id is not None:
                worker_info['gpu_name'] = torch.cuda.get_device_name(self.gpu_id)
                worker_info['gpu_memory'] = f"{torch.cuda.get_device_properties(self.gpu_id).total_memory / 1024**3:.1f} GB"
            
            self.socket.sendall(json.dumps(worker_info).encode('utf-8') + b'\n')
            logger.info(f"Sent registration info to coordinator (max_concurrent: {self.max_concurrent})")
            
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

    def handle_requests(self):
        """處理來自協調器的請求和心跳"""
        buffer = b''
        while True:
            try:
                chunk = self.socket.recv(4096)
                if not chunk:
                    logger.warning("Coordinator disconnected during request handling")
                    break
                
                buffer += chunk
                
                # 檢查是否有完整的消息（以換行符結尾）
                while b'\n' in buffer:
                    pos = buffer.find(b'\n')
                    message = buffer[:pos]
                    buffer = buffer[pos+1:]
                    
                    if not message:
                        continue
                    
                    # 處理心跳消息
                    if message == b'ping':
                        try:
                            self.socket.sendall(b'pong\n')
                            logger.debug("Heartbeat: pong sent")
                        except Exception as e:
                            logger.error(f"Error sending pong: {e}")
                            break
                        continue
                    
                    # 嘗試解析 JSON 請求
                    try:
                        request = json.loads(message.decode('utf-8'))
                        logger.info(f"Received request: {request.get('data', {}).get('prompt', '')[:30]}...")
                        
                        # 放入隊列而不是直接處理
                        self.request_queue.put(request)
                        logger.debug(f"Request queued, queue size: {self.request_queue.qsize()}")
                        
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON decode error: {e}")
                        logger.error(f"Problematic message: {message[:100]}")
                        continue
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")
                        continue
                        
            except Exception as e:
                logger.error(f"Error receiving data: {e}")
                break
        
        # 重新連接
        self.reconnect()

    def format_prompt(self, user_input):
        """格式化prompt以獲得更好的回應"""
        formatted = f"用戶：{user_input}\n助手："
        return formatted

    def process_single_request(self, request):
        """處理單個請求"""
        try:
            # 解析請求
            client_id = request.get('client_id')
            client_addr = request.get('client_addr')
            data = request.get('data', {})
            
            prompt = data.get('prompt', '')
            max_length = data.get('max_length', 1024)
            temperature = data.get('temperature', 0.7)
            
            thread_name = threading.current_thread().name
            logger.info(f"{thread_name} processing request for client {client_id}: {prompt[:30]}...")
            
            # 添加系統提示來控制模型行為
            formatted_prompt = self.format_prompt(prompt)
            
            # 顯示當前 GPU 使用情況
            if self.device.type == 'cuda':
                memory_used = torch.cuda.memory_allocated(self.device) / 1024**3
                memory_total = torch.cuda.get_device_properties(self.device).total_memory / 1024**3
                logger.info(f"{thread_name} GPU {self.gpu_id} memory usage: {memory_used:.2f}/{memory_total:.1f} GB")
            
            # 向控制器發送streaming
            try:
                start_msg = {
                    'type': 'start',
                    'message': 'Starting generation...',
                    'worker_thread': thread_name
                }
                self.send_response(start_msg)
                
                streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
                
                # 記錄推理開始時間
                start_time = time.time()
                
                inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                generation_kwargs = {
                    'input_ids': inputs['input_ids'],
                    'attention_mask': inputs.get('attention_mask', None),
                    'max_new_tokens': max_length,
                    'temperature': max(temperature, 0.1),  
                    'top_p': 0.9,                          
                    'top_k': 40,                           
                    'repetition_penalty': 1.05,            
                    'streamer': streamer,
                    'do_sample': True,
                    'pad_token_id': self.tokenizer.eos_token_id,
                    'eos_token_id': self.tokenizer.eos_token_id,
                    'early_stopping': True,                
                    'no_repeat_ngram_size': 3,            
                }
                
                # 使用 CUDA context 確保線程安全
                with torch.cuda.device(self.device):
                    thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
                    thread.start()
                    
                    generated_text = ""
                    token_count = 0
                    for new_text in streamer:
                        generated_text += new_text
                        token_count += 1
                        token_msg = {
                            'type': 'token',
                            'token': new_text,
                            'worker_thread': thread_name
                        }
                        self.send_response(token_msg)
                        
                        if token_count % 5 == 0:  
                            time.sleep(0.01)
                    
                    thread.join()
                
                inference_time = time.time() - start_time
                end_msg = {
                    'type': 'end',
                    'message': 'Generation complete',
                    'inference_time': inference_time,
                    'total_length': len(generated_text),
                    'gpu_id': self.gpu_id,
                    'worker_thread': thread_name
                }
                self.send_response(end_msg)
                
                logger.info(f"{thread_name} completed for client {client_id}: {len(generated_text)} chars, {inference_time:.2f}s, {token_count} tokens")
            
            except Exception as e:
                logger.error(f"{thread_name} error streaming response: {e}")
                error_msg = {
                    'type': 'error',
                    'error': str(e),
                    'worker_thread': thread_name
                }
                self.send_response(error_msg)
            
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            error_msg = {
                'type': 'error',
                'error': str(e)
            }
            self.send_response(error_msg)

    def send_response(self, response):
        try:
            response_json = json.dumps(response, ensure_ascii=False).encode('utf-8') + b'\n'
            total_sent = 0
            while total_sent < len(response_json):
                sent = self.socket.send(response_json[total_sent:])
                if sent == 0:
                    raise RuntimeError("Socket connection broken")
                total_sent += sent
            logger.debug(f"Sent complete response: {response.get('type', 'unknown')}")
        except Exception as e:
            logger.error(f"Error sending response: {e}")
            raise

    def reconnect(self):
        """重新連接到控制器"""
        max_retries = 10
        retry_delay = 5  
        
        for attempt in range(max_retries):
            logger.info(f"Attempting to reconnect to coordinator (attempt {attempt+1}/{max_retries})...")
            try:
                self.socket.close()
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                
                if not self.coordinator_host:
                    self.discover_coordinator()
                
                if self.connect_to_coordinator():
                    logger.info("Reconnected to coordinator")
                    threading.Thread(target=self.handle_requests, daemon=True).start()
                    return True
            except Exception as e:
                logger.error(f"Reconnection attempt failed: {e}")
            
            time.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, 60)  
        
        logger.error("Failed to reconnect after maximum attempts")
        return False

    def start(self):
        logger.info(f"Attempting to connect to coordinator at {self.coordinator_host}:{self.coordinator_port}")
        
        if self.connect_to_coordinator():
            # 啟動請求處理線程（包含心跳處理）
            threading.Thread(target=self.handle_requests, daemon=True).start()
            
            logger.info(f"Worker started, connected to coordinator {self.coordinator_host}:{self.coordinator_port}")
            
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
   
   parser = argparse.ArgumentParser(description='LLM Worker Node with Parallel Processing')
   parser.add_argument('--model', type=str, required=True, help='Model path')
   parser.add_argument('--node-id', type=str, default=None, help='Node identifier (auto-generated if not specified)')
   parser.add_argument('--gpu', type=int, default=None, help='Specific GPU ID to use (e.g., 0, 1). If not specified, will use GPU 0')
   parser.add_argument('--host', type=str, default=None, help='Coordinator host (optional, will auto-discover if not specified)')
   parser.add_argument('--port', type=int, default=9000, help='Coordinator port')
   parser.add_argument('--discovery-port', type=int, default=9001, help='Discovery service port')
   parser.add_argument('--max-concurrent', type=int, default=2, help='Maximum concurrent requests (default: 2)')
   
   args = parser.parse_args()
   
   # 自動生成 node_id
   if not args.node_id:
       import uuid
       hostname = socket.gethostname()
       node_id = f"worker-{hostname}-{str(uuid.uuid4())[:8]}"
       if args.gpu is not None:
           node_id += f"-gpu{args.gpu}"
       args.node_id = node_id
   
   try:
       worker = LLMWorker(
           model_path=args.model,
           node_id=args.node_id,
           gpu_id=args.gpu,
           coordinator_host=args.host,
           coordinator_port=args.port,
           discovery_port=args.discovery_port,
           max_concurrent=args.max_concurrent
       )
       worker.start()
   except Exception as e:
       logger.error(f"Worker error: {e}")