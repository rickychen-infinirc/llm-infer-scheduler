import socket
import json
import sys
import logging
import argparse
import time
import threading
import readline  # 提供更好的命令行輸入體驗

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("LLM-Client")

class LLMClient:
    def __init__(self, coordinator_host, coordinator_port=9000):
        self.coordinator_host = coordinator_host
        self.coordinator_port = coordinator_port
        self.socket = None
        self.running = True
        self.response_buffer = ""
        self.connected = False
        
    def connect(self):
        """連接到協調器"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.coordinator_host, self.coordinator_port))
            logger.info(f"Connected to coordinator at {self.coordinator_host}:{self.coordinator_port}")
            
            # 發送客戶端標記
            self.socket.sendall(b'C')
            logger.info("Sent client identifier")
            
            # 嘗試讀取響應
            try:
                # 設置超時
                self.socket.settimeout(10)
                data = self.socket.recv(1024)
                if data:
                    try:
                        response = json.loads(data.decode('utf-8'))
                        logger.info(f"Received response: {response}")
                        if response.get('status') == 'connected':
                            logger.info(f"Connected to worker {response.get('worker_id')}")
                            self.connected = True
                            # 重置超時
                            self.socket.settimeout(None)
                            return True
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse response: {data}")
                        
                logger.warning("Did not receive connection confirmation")
                # 即使沒有收到確認，也繼續嘗試（可能是協調器的問題）
                self.connected = True
                # 重置超時
                self.socket.settimeout(None)
                return True
            except socket.timeout:
                logger.warning("Timeout waiting for connection confirmation")
                # 即使超時，也繼續嘗試
                self.connected = True
                # 重置超時
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
    
    def receive_responses(self):
        """接收並處理來自服務器的流式響應"""
        buffer = b''
        try:
            while self.running:
                try:
                    # 非阻塞接收
                    self.socket.setblocking(0)
                    ready = select.select([self.socket], [], [], 0.5)
                    if ready[0]:
                        chunk = self.socket.recv(4096)
                        if not chunk:
                            logger.warning("Server disconnected")
                            break
                        
                        buffer += chunk
                        
                        # 處理完整的消息（以換行符分隔）
                        while b'\n' in buffer:
                            pos = buffer.find(b'\n')
                            message = buffer[:pos]
                            buffer = buffer[pos+1:]
                            
                            if message:
                                try:
                                    response = json.loads(message.decode('utf-8'))
                                    self.process_response(response)
                                except json.JSONDecodeError as e:
                                    logger.error(f"JSON decode error: {e}, message: {message}")
                except Exception as e:
                    # 忽略非阻塞socket的異常
                    pass
                
                time.sleep(0.1)  # 短暫休眠以減少CPU使用
                
        except Exception as e:
            if self.running:
                logger.error(f"Error receiving responses: {e}")
        finally:
            self.running = False
            print("\nConnection closed.")
    
    def process_response(self, response):
        """處理來自伺服器的響應"""
        # 記錄接收到的響應以幫助調試
        logger.debug(f"Processing response: {response}")
        
        response_type = response.get('type', '')
        
        if response_type == 'start':
            sys.stdout.write("\nLLM: ")
            sys.stdout.flush()
            self.response_buffer = ""
        
        elif response_type == 'token':
            token = response.get('token', '')
            sys.stdout.write(token)
            sys.stdout.flush()
            self.response_buffer += token
        
        elif response_type == 'end':
            inference_time = response.get('inference_time', 0)
            total_length = response.get('total_length', 0)
            print(f"\n\n[Generation complete: {total_length} chars, {inference_time:.2f}s]")
            print("\nYou: ", end='')
            sys.stdout.flush()
        
        elif response_type == 'error':
            error = response.get('error', 'Unknown error')
            print(f"\nError: {error}")
            print("\nYou: ", end='')
            sys.stdout.flush()
        
        # 處理普通消息（無類型）
        elif 'error' in response:
            print(f"\nError: {response.get('error')}")
            print("\nYou: ", end='')
            sys.stdout.flush()
        
        # 處理可能的確認消息
        elif 'status' in response:
            status = response.get('status', '')
            if status == 'connected':
                worker_id = response.get('worker_id', 'unknown')
                print(f"\nConnected to worker {worker_id}")
                print("\nYou: ", end='')
                sys.stdout.flush()
    
    def send_request(self, prompt, max_length=1024, temperature=0.7):
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
            logger.info(f"Request sent: prompt length {len(prompt)}")
            return True
        except ConnectionResetError:
            logger.error("Connection reset by server")
            self.running = False
            return False
        except Exception as e:
            logger.error(f"Error sending request: {e}")
            self.running = False
            return False
    
    def interactive_session(self, max_length=1024, temperature=0.7):
        """啟動交互式會話"""
        if not self.connect():
            print("Failed to connect to server. Please check the connection and try again.")
            return
            
        print("\nConnected to LLM service. Type 'exit' to quit.")
        print("\nYou: ", end='')
        sys.stdout.flush()
        
        # 輸入處理循環
        while self.running:
            try:
                prompt = input()
                if prompt.lower() in ('exit', 'quit'):
                    self.running = False
                    break
                
                if prompt.strip():
                    success = self.send_request(prompt, max_length, temperature)
                    if not success:
                        print("Failed to send message. Connection may be lost.")
                        break
                    # 手動處理響應，因為receive_responses似乎有問題
                    try:
                        sys.stdout.write("\nLLM: ")
                        sys.stdout.flush()
                        
                        while self.running:
                            data = self.socket.recv(4096)
                            if not data:
                                break
                                
                            try:
                                # 分割多個響應
                                messages = data.split(b'\n')
                                for msg in messages:
                                    if not msg:
                                        continue
                                    try:
                                        response = json.loads(msg.decode('utf-8'))
                                        if response.get('type') == 'token':
                                            token = response.get('token', '')
                                            sys.stdout.write(token)
                                            sys.stdout.flush()
                                        elif response.get('type') == 'end':
                                            # 生成結束
                                            sys.stdout.write("\n")
                                            sys.stdout.flush()
                                            print("\nYou: ", end='')
                                            sys.stdout.flush()
                                            break
                                    except:
                                        continue
                            except:
                                continue
                    except Exception as e:
                        logger.error(f"Error receiving response: {e}")
            except KeyboardInterrupt:
                self.running = False
                break
            except EOFError:
                self.running = False
                break
        
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
        print("\nSession ended.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LLM Client - Interactive Chat')
    parser.add_argument('--host', type=str, required=True, help='Coordinator host')
    parser.add_argument('--port', type=int, default=9000, help='Coordinator port')
    parser.add_argument('--max-length', type=int, default=1024, help='Maximum length of generated text')
    parser.add_argument('--temperature', type=float, default=0.7, help='Temperature for text generation')
    
    args = parser.parse_args()
    
    client = LLMClient(args.host, args.port)
    try:
        client.interactive_session(args.max_length, args.temperature)
    except Exception as e:
        logger.error(f"Client error: {e}")
