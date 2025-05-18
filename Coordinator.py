import socket
import threading
import json
import queue
import time
import logging
import sys

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("LLM-Coordinator")

class LLMCoordinator:
    def __init__(self, host='0.0.0.0', port=9000):
        self.host = host
        self.port = port
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        
        # 存儲工作節點的信息
        self.worker_nodes = []
        self.worker_lock = threading.Lock()
        
        # 請求隊列
        self.requests_queue = queue.Queue()
        
        logger.info(f"Coordinator initialized on {self.host}:{self.port}")
        self.dispatching = False  # Flag to control the dispatcher thread

    def register_worker(self, worker_socket, worker_info):
        """註冊新的工作節點"""
        with self.worker_lock:
            worker_id = len(self.worker_nodes)
            worker = {
                'id': worker_id,
                'socket': worker_socket,
                'info': worker_info,
                'busy': False,
                'last_active': time.time()
            }
            self.worker_nodes.append(worker)
            logger.info(f"Worker {worker_id} registered: {worker_info}")
            
            # 如果這是第一個工作節點，且分發線程尚未啟動，則啟動分發線程
            if len(self.worker_nodes) == 1 and not self.dispatching:
                self.dispatching = True
                threading.Thread(target=self.dispatch_requests, daemon=True).start()
                
            return worker_id

    def get_available_worker(self):
        """獲取可用的工作節點"""
        with self.worker_lock:
            for worker in self.worker_nodes:
                if not worker['busy']:
                    worker['busy'] = True
                    return worker
            return None

    def release_worker(self, worker_id):
        """釋放工作節點"""
        with self.worker_lock:
            for worker in self.worker_nodes:
                if worker['id'] == worker_id:
                    worker['busy'] = False
                    worker['last_active'] = time.time()
                    logger.info(f"Worker {worker_id} released")
                    return True
            return False

    def dispatch_requests(self):
        """分發請求到可用的工作節點"""
        logger.info("Request dispatcher started")
        while True:
            try:
                # 從隊列中獲取請求
                client_socket, request_data = self.requests_queue.get()
                logger.info(f"Dispatching request: {request_data.get('prompt', '')[:30]}...")
                
                # 等待可用的工作節點
                worker = None
                while worker is None:
                    worker = self.get_available_worker()
                    if worker is None:
                        logger.info("No available workers, waiting...")
                        time.sleep(1)
                
                try:
                    # 將請求發送給工作節點
                    worker_socket = worker['socket']
                    request_with_client = {
                        'client_id': id(client_socket),
                        'client_addr': client_socket.getpeername(),
                        'data': request_data
                    }
                    
                    # 發送請求到工作節點
                    request_json = json.dumps(request_with_client).encode('utf-8') + b'\n'
                    worker_socket.sendall(request_json)
                    logger.info(f"Request dispatched to worker {worker['id']}, size: {len(request_json)} bytes")
                    
                    # 啟動線程等待工作節點的響應並轉發給客戶端
                    threading.Thread(
                        target=self.handle_worker_response, 
                        args=(worker, client_socket)
                    ).start()
                    
                except Exception as e:
                    logger.error(f"Error dispatching request: {e}")
                    self.release_worker(worker['id'])
                    # 發送錯誤消息給客戶端
                    error_msg = {'error': str(e)}
                    try:
                        client_socket.sendall(json.dumps(error_msg).encode('utf-8') + b'\n')
                    except:
                        pass
                    finally:
                        client_socket.close()
            except Exception as e:
                logger.error(f"Error in dispatch_requests: {e}")
                time.sleep(1)  # 避免因錯誤導致CPU占用過高

    def handle_worker_response(self, worker, client_socket):
        """處理工作節點的響應並轉發給客戶端"""
        worker_socket = worker['socket']
        worker_id = worker['id']
        
        try:
            # 從工作節點接收響應
            buffer = b''
            while True:
                try:
                    chunk = worker_socket.recv(4096)
                    if not chunk:
                        logger.warning(f"Worker {worker_id} disconnected")
                        break
                    
                    buffer += chunk
                    
                    # 檢查是否有完整的消息（以換行符結尾）
                    if b'\n' in buffer:
                        messages = buffer.split(b'\n')
                        # 處理所有完整的消息
                        for i in range(len(messages) - 1):
                            if messages[i]:
                                # 將響應發送給客戶端
                                try:
                                    client_socket.sendall(messages[i] + b'\n')
                                    logger.info(f"Response from worker {worker_id} forwarded to client, size: {len(messages[i])} bytes")
                                except Exception as e:
                                    logger.error(f"Error sending to client: {e}")
                                    raise
                        
                        # 保留未完成的消息
                        buffer = messages[-1]
                except socket.timeout:
                    continue
                except Exception as e:
                    logger.error(f"Error receiving from worker {worker_id}: {e}")
                    break
        except Exception as e:
            logger.error(f"Error handling worker response: {e}")
            # 發送錯誤消息給客戶端
            error_msg = {'type': 'error', 'error': str(e)}
            try:
                client_socket.sendall(json.dumps(error_msg).encode('utf-8') + b'\n')
            except:
                pass
        finally:
            # 釋放工作節點
            self.release_worker(worker_id)

    def handle_client(self, client_socket, addr):
        """處理客戶端連接"""
        logger.info(f"Client connected from {addr}")
        client_id = id(client_socket)
        
        try:
            while True:
                # 從客戶端接收數據
                data = b''
                while True:
                    try:
                        chunk = client_socket.recv(4096)
                        if not chunk:
                            logger.info(f"Client {client_id} disconnected")
                            return
                        
                        data += chunk
                        
                        # 檢查是否是完整的消息 (以換行符結尾)
                        if b'\n' in data:
                            break
                    except socket.timeout:
                        continue
                
                # 解析請求數據
                messages = data.split(b'\n')
                for message in messages:
                    if not message:
                        continue
                    
                    try:
                        request_data = json.loads(message.decode('utf-8'))
                        logger.info(f"Received request from client {client_id}: {request_data.get('prompt', '')[:30]}...")
                        
                        # 將請求加入隊列
                        self.requests_queue.put((client_socket, request_data))
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON decode error: {e}")
                        error_msg = {'type': 'error', 'error': f"JSON decode error: {e}"}
                        client_socket.sendall(json.dumps(error_msg).encode('utf-8') + b'\n')
        except ConnectionResetError:
            logger.info(f"Connection reset by client {client_id}")
        except Exception as e:
            logger.error(f"Error handling client {client_id}: {e}")
        finally:
            try:
                client_socket.close()
            except:
                pass
            logger.info(f"Client {client_id} connection closed")

    def handle_worker_connection(self, worker_socket, addr):
        """處理工作節點的連接"""
        logger.info(f"Worker connected from {addr}")
        try:
            # 設置接收超時
            worker_socket.settimeout(5.0)
            
            # 接收工作節點的註冊信息
            data = b''
            while True:
                try:
                    chunk = worker_socket.recv(4096)
                    if not chunk:
                        logger.warning(f"Worker disconnected during registration")
                        return
                    
                    data += chunk
                    if b'\n' in data or len(data) > 4096:  # 如果收到換行符或數據足夠長就停止等待
                        break
                except socket.timeout:
                    continue
            
            if data:
                try:
                    # 嘗試解析JSON
                    worker_info = json.loads(data.decode('utf-8').strip())
                    worker_id = self.register_worker(worker_socket, worker_info)
                    
                    # 發送確認消息
                    confirmation = {'worker_id': worker_id, 'status': 'registered'}
                    worker_socket.sendall(json.dumps(confirmation).encode('utf-8') + b'\n')
                    
                    # 保持連接直到斷開
                    worker_socket.settimeout(60.0)  # 設置較長的超時時間用於心跳檢測
                    while True:
                        # 心跳檢測
                        try:
                            worker_socket.sendall(b'ping\n')
                            response = worker_socket.recv(10)
                            if not response:
                                logger.warning(f"Worker {worker_id} disconnected")
                                break
                            elif response != b'pong\n':
                                logger.warning(f"Worker {worker_id} sent invalid heartbeat response: {response}")
                        except socket.timeout:
                            logger.warning(f"Worker {worker_id} heartbeat timeout")
                            continue
                        except Exception as e:
                            logger.error(f"Worker {worker_id} heartbeat error: {e}")
                            break
                        
                        time.sleep(30)  # 每30秒檢測一次
                except json.JSONDecodeError as e:
                    logger.error(f"Worker registration JSON decode error: {e}")
                except Exception as e:
                    logger.error(f"Error registering worker: {e}")
            
            # 關閉連接
            worker_socket.close()
        except Exception as e:
            logger.error(f"Error handling worker connection: {e}")
        
        # 嘗試移除工作節點
        try:
            with self.worker_lock:
                self.worker_nodes = [w for w in self.worker_nodes if w['socket'] != worker_socket]
            
            worker_socket.close()
        except:
            pass

    def start(self):
        """啟動協調器服務"""
        self.server_socket.listen(10)
        logger.info(f"Coordinator listening on {self.host}:{self.port}")
        
        # 等待連接
        while True:
            try:
                client_socket, addr = self.server_socket.accept()
                client_socket.settimeout(5.0)  # 設置接收超時
                
                try:
                    # 讀取第一個字節來確定連接類型
                    first_byte = client_socket.recv(1)
                    
                    if not first_byte:
                        logger.warning(f"Empty connection from {addr}")
                        client_socket.close()
                        continue
                    
                    if first_byte == b'C':
                        # 客戶端連接
                        threading.Thread(target=self.handle_client, args=(client_socket, addr)).start()
                    elif first_byte == b'W':
                        # 工作節點連接
                        threading.Thread(target=self.handle_worker_connection, args=(client_socket, addr)).start()
                    else:
                        logger.warning(f"Unknown connection type from {addr}: {first_byte}")
                        client_socket.close()
                except socket.timeout:
                    logger.warning(f"Connection timeout from {addr} while determining type")
                    client_socket.close()
                except Exception as e:
                    logger.error(f"Error determining connection type: {e}")
                    client_socket.close()
            except Exception as e:
                logger.error(f"Error accepting connection: {e}")

if __name__ == "__main__":
    try:
        coordinator = LLMCoordinator()
        coordinator.start()
    except KeyboardInterrupt:
        logger.info("Coordinator shutting down...")
    except Exception as e:
        logger.error(f"Coordinator error: {e}")
