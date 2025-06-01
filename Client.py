#Desktop/llm-infer-scheduler/Client.py
import socket
import json
import sys
import logging
import argparse
import time
import threading
import select

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("LLM-Client")

class LLMClient:
    def __init__(self, coordinator_host="auto", coordinator_port=9000, discovery_port=9001):
        self.coordinator_host = coordinator_host
        self.coordinator_port = coordinator_port
        self.discovery_port = discovery_port
        self.socket = None
        self.running = True
        self.connected = False
        
        # 如果需要自動發現協調器
        if self.coordinator_host == "auto" or not self.coordinator_host:
            self.discover_coordinator()
        
    def discover_coordinator(self):
        """自動發現協調器"""
        logger.info("Starting controller discovery...")
        
        # 創建發現 socket
        discovery_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        discovery_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        discovery_socket.settimeout(5.0)
        
        coordinators_found = []
        
        # 方式1: 廣播發現請求
        try:
            broadcast_addrs = ['255.255.255.255', '192.168.255.255', '10.255.255.255', '172.31.255.255']
            
            logger.info("Broadcasting discovery requests...")
            for broadcast_addr in broadcast_addrs:
                try:
                    discovery_socket.sendto(b"DISCOVER_COORDINATOR", (broadcast_addr, self.discovery_port))
                    logger.debug(f"Sent discovery request to {broadcast_addr}:{self.discovery_port}")
                except Exception as e:
                    logger.debug(f"Failed to broadcast to {broadcast_addr}: {e}")
            
            # 監聽回應
            start_time = time.time()
            while time.time() - start_time < 8:  # 等待8秒
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
                        logger.info(f"Found controller at {response['host']}:{response['port']}")
                
                except socket.timeout:
                    continue
                except Exception as e:
                    logger.debug(f"Error receiving discovery response: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error in broadcast discovery: {e}")
        
        # 方式2: 監聽協調器廣播公告
        if not coordinators_found:
            logger.info("Listening for controller announcements...")
            
            listen_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            try:
                listen_socket.bind(('', self.discovery_port))
                listen_socket.settimeout(25.0)  # 監聽25秒
                
                start_time = time.time()
                while time.time() - start_time < 25:
                    try:
                        data, addr = listen_socket.recvfrom(1024)
                        announcement = json.loads(data.decode('utf-8'))
                        
                        if announcement.get('type') == 'COORDINATOR_ANNOUNCEMENT':
                            coordinator_info = {
                                'host': announcement['host'],
                                'port': announcement['port'],
                                'web_port': announcement.get('web_port'),
                                'timestamp': announcement.get('timestamp', 0),
                                'source_ip': addr[0],
                                'worker_count': announcement.get('worker_count', 0)
                            }
                            coordinators_found.append(coordinator_info)
                            logger.info(f"📢 Received announcement from coordinator at {announcement['host']}:{announcement['port']} ({announcement.get('worker_count', 0)} workers)")
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
        
        # 方式3: 檢查常見地址
        if not coordinators_found:
            logger.info(" Checking common addresses...")
            common_addresses = ['localhost', '127.0.0.1', '192.168.1.1', '192.168.0.1']
            
            for addr in common_addresses:
                try:
                    test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    test_socket.settimeout(2.0)
                    result = test_socket.connect_ex((addr, self.coordinator_port))
                    test_socket.close()
                    
                    if result == 0:
                        coordinators_found.append({
                            'host': addr,
                            'port': self.coordinator_port,
                            'timestamp': time.time(),
                            'source_ip': addr
                        })
                        logger.info(f"Found controller common address {addr}:{self.coordinator_port}")
                        break
                except:
                    continue
        
        # 方式4: 網段掃描 
        if not coordinators_found:
            logger.info(" Scanning local network...")
            coordinators_found = self.scan_for_coordinators()
        
        discovery_socket.close()
        

        if coordinators_found:
            coordinators_found.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
            best_coordinator = coordinators_found[0]
            
            self.coordinator_host = best_coordinator['host']
            self.coordinator_port = best_coordinator['port']
            
            logger.info(f"Selected controller: {self.coordinator_host}:{self.coordinator_port}")
            

            if best_coordinator.get('web_port'):
                logger.info(f"Web UI available at: http://{self.coordinator_host}:{best_coordinator['web_port']}")
            
            if best_coordinator.get('worker_count'):
                logger.info(f"Connected workers: {best_coordinator['worker_count']}")
            
            if len(coordinators_found) > 1:
                logger.info(f"Found {len(coordinators_found)} coordinators, selected the most recent one")
        else:
            logger.error("No coordinator found!")
            logger.info("Please ensure a coordinator is running on the network")
            logger.info("You can also specify a coordinator manually with --host <ip>")
            raise RuntimeError("Coordinator discovery failed")

    def scan_for_coordinators(self):
        """掃描本地網絡尋找協調器"""
        logger.info("Scanning local network for coordinators...")
        
        try:
            # 獲取本機 IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            
            # 計算網絡段
            ip_parts = local_ip.split('.')
            network_base = f"{ip_parts[0]}.{ip_parts[1]}.{ip_parts[2]}"
            
            logger.info(f"🔍 Scanning network {network_base}.1-254")
            
            coordinators_found = []
            
            # 快速掃描常見 IP
            common_ips = [1, 2, 10, 11, 100, 101, 110, 111, 200, 201, 254]
            
            def scan_ip(ip):
                try:
                    test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    test_socket.settimeout(0.5)
                    result = test_socket.connect_ex((ip, self.coordinator_port))
                    test_socket.close()
                    
                    if result == 0:
                        logger.info(f" Found coordinator at {ip}:{self.coordinator_port}")
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
            for i in common_ips:
                ip = f"{network_base}.{i}"
                if ip != local_ip:
                    thread = threading.Thread(target=scan_ip, args=(ip,))
                    threads.append(thread)
                    thread.start()
            
            # 等待掃描完成
            for thread in threads:
                thread.join()
            
            return coordinators_found
            
        except Exception as e:
            logger.error(f"Error in network scanning: {e}")
            return []
        
    def connect(self):
        """連接到協調器"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.coordinator_host, self.coordinator_port))
            logger.info(f" Connected to coordinator at {self.coordinator_host}:{self.coordinator_port}")
            
            # 發送客戶端標記
            self.socket.sendall(b'C')
            logger.debug("Sent client identifier")
            
            # 嘗試讀取響應（可選）
            try:
                self.socket.settimeout(5)
                data = self.socket.recv(1024)
                if data:
                    try:
                        response = json.loads(data.decode('utf-8'))
                        if response.get('status') == 'connected':
                            logger.info(f"Connected to worker {response.get('worker_id')}")
                    except json.JSONDecodeError:
                        logger.debug("No JSON response from coordinator")
                        
                self.connected = True
                self.socket.settimeout(None)
                return True
            except socket.timeout:
                logger.debug("No immediate response from coordinator")
                self.connected = True
                self.socket.settimeout(None)
                return True
            
        except Exception as e:
            logger.error(f"Connection error: {e}")
            if self.socket:
                try:
                    self.socket.close()
                except:
                    pass
            return False
    
    def send_request(self, prompt, max_length=8192, temperature=0.9):
        """發送推理請求"""
        if not self.connected:
            logger.error("Not connected to server")
            return False
            
        try:
            # 準備請求數據
            request_data = {
                'prompt': prompt,
                'max_length': max_length,
                'temperature': temperature,
                'timestamp': time.time()
            }
            
            # 發送請求
            request_json = json.dumps(request_data).encode('utf-8') + b'\n'
            self.socket.sendall(request_json)
            logger.debug(f"Request sent: prompt length {len(prompt)}")
            return True
        except ConnectionResetError:
            logger.error("Connection reset by server")
            self.running = False
            return False
        except Exception as e:
            logger.error(f"Error sending request: {e}")
            self.running = False
            return False

    def receive_response(self):
        """接收並處理單個響應"""
        buffer = b''
        response_complete = False
        generation_started = False
        
        try:
            while not response_complete and self.running:
                # 檢查是否有數據可讀
                ready = select.select([self.socket], [], [], 1.0)
                if not ready[0]:
                    continue
                    
                chunk = self.socket.recv(4096)
                if not chunk:
                    logger.warning("Server disconnected")
                    self.running = False
                    break
                
                buffer += chunk
                
                # 處理完整的消息
                while b'\n' in buffer:
                    pos = buffer.find(b'\n')
                    message = buffer[:pos]
                    buffer = buffer[pos+1:]
                    
                    if message:
                        try:
                            response = json.loads(message.decode('utf-8'))
                            result = self.process_response(response)
                            
                            # 檢查響應類型
                            response_type = response.get('type', '')
                            if response_type == 'start':
                                generation_started = True
                            elif response_type == 'end' and generation_started:
                                response_complete = True
                                break
                            elif response_type == 'error':
                                response_complete = True
                                break
                                
                        except json.JSONDecodeError as e:
                            # 忽略心跳和其他非JSON消息
                            if message == b'pong':
                                logger.debug("Received pong message")
                                continue
                            logger.debug(f"Non-JSON message: {message[:50]}")
                            continue
                            
        except Exception as e:
            logger.error(f"Error receiving response: {e}")
            self.running = False
            
        return response_complete

    def process_response(self, response):
        """處理來自伺服器的響應，返回True表示響應完成"""
        response_type = response.get('type', '')
        
        if response_type == 'start':
            sys.stdout.write("\n🤖 LLM: ")
            sys.stdout.flush()
            return False
        
        elif response_type == 'token':
            token = response.get('token', '')
            sys.stdout.write(token)
            sys.stdout.flush()
            return False
        
        elif response_type == 'end':
            inference_time = response.get('inference_time', 0)
            total_length = response.get('total_length', 0)
            gpu_id = response.get('gpu_id')
            
            print()  # 換行用
            print(f"Generation complete: {total_length} chars, {inference_time:.2f}s", end="")
            if gpu_id is not None:
                print(f", GPU {gpu_id}")
            else:
                print()
            return True  
        
        elif response_type == 'error':
            error = response.get('error', 'Unknown error')
            print(f"\n Error: {error}")
            return True  
        

        elif 'error' in response:
            print(f"\n Error: {response.get('error')}")
            return True  
        

        elif 'status' in response:
            status = response.get('status', '')
            if status == 'connected':
                worker_id = response.get('worker_id', 'unknown')
                print(f"\n Connected to worker {worker_id}")
            return False  
            
        return False
    
    def interactive_session(self, max_length=1024, temperature=0.7):
        """啟動交互式會話"""
        print(f"LLM Client with Auto-Discovery")
        print(f"   controller: {self.coordinator_host}:{self.coordinator_port}")
        print(f"   Max Length: {max_length}, Temperature: {temperature}")
        print("="*60)
        
        if not self.connect():
            print("Failed to connect to server. Please check the connection and try again.")
            return
            
        print("Connected to LLM service!")
        print("Type your message and press Enter. Type 'exit' to quit.")
        print("Commands: 'exit', 'quit', 'help', 'status'")
        print("="*60)
        
        while self.running:
            try:
                print("\n💬 You: ", end='')
                sys.stdout.flush()
                
                prompt = input()
                
                # 處理特殊命令
                if prompt.lower() in ('exit', 'quit'):
                    print("👋 Goodbye!")
                    self.running = False
                    break
                
                elif prompt.lower() == 'help':
                    print("📖 Available commands:")
                    print("   exit/quit - Exit the client")
                    print("   help      - Show this help message")
                    print("   status    - Show connection status")
                    print("   Just type your message to chat with the LLM!")
                    continue
                
                elif prompt.lower() == 'status':
                    print(f" Connected to: {self.coordinator_host}:{self.coordinator_port}")
                    print(f"⚙️  Settings: max_length={max_length}, temperature={temperature}")
                    continue
                
                if prompt.strip():
                    success = self.send_request(prompt, max_length, temperature)
                    if not success:
                        print(" Failed to send message. Connection may be lost.")
                        break
                    
                    # 接收響應
                    response_received = self.receive_response()
                    if not response_received and self.running:
                        print("\n No response received or connection lost.")
                        break
                        
            except KeyboardInterrupt:
                print("\n👋 Interrupted by user. Goodbye!")
                self.running = False
                break
            except EOFError:
                print("\n👋 End of input. Goodbye!")
                self.running = False
                break
            except Exception as e:
                logger.error(f"Error in interactive session: {e}")
                break
        
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
        print("Connection closed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LLM Client with Auto-Discovery')
    parser.add_argument('--host', type=str, default='auto', 
                       help='Coordinator host (use "auto" for auto-discovery, default: auto)')
    parser.add_argument('--port', type=int, default=9000, help='Coordinator port (default: 9000)')
    parser.add_argument('--discovery-port', type=int, default=9001, help='Discovery port (default: 9001)')
    parser.add_argument('--max-length', type=int, default=8192, help='Maximum length of generated text')
    parser.add_argument('--temperature', type=float, default=0.9, help='Temperature for text generation')
    
    args = parser.parse_args()
    
    try:
        client = LLMClient(
            coordinator_host=args.host, 
            coordinator_port=args.port,
            discovery_port=args.discovery_port
        )
        client.interactive_session(args.max_length, args.temperature)
    except Exception as e:
        logger.error(f"Client error: {e}")
