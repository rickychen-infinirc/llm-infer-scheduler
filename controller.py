#Desktop/llm-infer-scheduler/controller.py
import socket
import threading
import json
import queue
import time
import logging
import sys
import struct
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO
from datetime import datetime
import uuid

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("LLM-Coordinator")

class LLMCoordinator:
    def __init__(self, host='0.0.0.0', port=9000, web_port=5000, discovery_port=9001):
        self.host = host
        self.port = port
        self.web_port = web_port
        self.discovery_port = discovery_port
        
        # 主服務器 socket
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        
        # 發現服務 socket (UDP)
        self.discovery_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.discovery_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.discovery_socket.bind(('', self.discovery_port))
        
        # 廣播公告 socket
        self.broadcast_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.broadcast_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        
        # 工作節點和客戶端
        self.worker_nodes = []
        self.worker_lock = threading.Lock()
        self.clients = {}
        self.client_lock = threading.Lock()
        
        # 請求隊列和統計
        self.requests_queue = queue.Queue()
        self.stats = {
            'total_requests': 0,
            'completed_requests': 0,
            'failed_requests': 0,
            'active_requests': 0,
            'queue_size': 0,
            'uptime': time.time()
        }
        self.request_history = []
        self.history_lock = threading.Lock()
        
        # 獲取本機 IP
        self.local_ip = self.get_local_ip()
        
        logger.info(f"Coordinator initialized on {self.local_ip}:{self.port}")
        logger.info(f"Discovery service on port {self.discovery_port}")
        logger.info(f"Web UI will be available at http://{self.local_ip}:{self.web_port}")
        self.dispatching = False
        
        # 初始化 Flask 應用
        self.setup_web_app()
        
        # 啟動服務發現
        self.start_discovery_services()

    def get_local_ip(self):
        """獲取本機 IP 地址"""
        try:
            # 連接到一個外部地址來獲取本機 IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return "127.0.0.1"

    def start_discovery_services(self):
        """啟動服務發現相關服務"""
        # 啟動發現請求處理器
        threading.Thread(target=self.handle_discovery_requests, daemon=True).start()
        
        # 啟動定期廣播公告
        threading.Thread(target=self.broadcast_announcement, daemon=True).start()

    def handle_discovery_requests(self):
        """處理來自 Worker 的發現請求"""
        logger.info(f"Discovery service listening on port {self.discovery_port}")
        
        while True:
            try:
                data, addr = self.discovery_socket.recvfrom(1024)
                message = data.decode('utf-8')
                
                if message == "DISCOVER_COORDINATOR":
                    logger.info(f"Discovery request from {addr[0]}")
                    
                    # 回應協調器信息
                    response = {
                        'type': 'COORDINATOR_INFO',
                        'host': self.local_ip,
                        'port': self.port,
                        'web_port': self.web_port,
                        'timestamp': time.time()
                    }
                    
                    response_data = json.dumps(response).encode('utf-8')
                    self.discovery_socket.sendto(response_data, addr)
                    logger.info(f"Sent coordinator info to {addr[0]}")
                    
            except Exception as e:
                logger.error(f"Error in discovery service: {e}")

    def broadcast_announcement(self):
        """定期廣播協調器存在"""
        while True:
            try:
                announcement = {
                    'type': 'COORDINATOR_ANNOUNCEMENT',
                    'host': self.local_ip,
                    'port': self.port,
                    'web_port': self.web_port,
                    'timestamp': time.time(),
                    'worker_count': len(self.worker_nodes)
                }
                
                message = json.dumps(announcement).encode('utf-8')
                
                # 廣播到所有子網
                broadcast_addrs = ['255.255.255.255', '192.168.255.255', '10.255.255.255', '172.31.255.255']
                
                for broadcast_addr in broadcast_addrs:
                    try:
                        self.broadcast_socket.sendto(message, (broadcast_addr, self.discovery_port))
                    except:
                        pass
                
                logger.debug(f"Broadcasted coordinator announcement")
                
            except Exception as e:
                logger.error(f"Error broadcasting announcement: {e}")
                
            time.sleep(30)  # 每30秒廣播一次

    def setup_web_app(self):
        """設置 Web 應用"""
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'llm-scheduler'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        @self.app.route('/')
        def index():
            return render_template('dashboard.html')
        
        @self.app.route('/api/status')
        def get_status():
            return jsonify(self.get_system_status())
        
        @self.app.route('/api/discovery')
        def get_discovery_info():
            return jsonify({
                'coordinator_ip': self.local_ip,
                'coordinator_port': self.port,
                'discovery_port': self.discovery_port,
                'web_port': self.web_port
            })
        
        @self.socketio.on('connect')
        def handle_connect():
            self.socketio.emit('system_update', self.get_system_status())
        
        # 定期更新客戶端
        def update_clients():
            while True:
                self.socketio.emit('system_update', self.get_system_status())
                time.sleep(1)
        
        threading.Thread(target=update_clients, daemon=True).start()

    def get_system_status(self):
        """獲取系統狀態"""
        with self.worker_lock:
            workers = []
            for worker in self.worker_nodes:
                workers.append({
                    'id': worker['id'],
                    'node_id': worker['info']['node_id'],
                    'device': worker['info']['device'],
                    'gpu_id': worker['info'].get('gpu_id'),
                    'model': worker['info']['model'].split('/')[-1],
                    'status': 'busy' if worker['busy'] else 'idle',
                    'total_requests': 0,
                    'discovery_ip': worker.get('discovery_ip', 'N/A')  # 顯示 Worker 的 IP
                })
        
        with self.client_lock:
            clients = list(self.clients.values())
        
        self.stats['queue_size'] = self.requests_queue.qsize()
        
        with self.history_lock:
            recent_history = self.request_history[-20:]
        
        return {
            'workers': workers,
            'clients': clients,
            'stats': self.stats,
            'request_history': recent_history,
            'uptime': time.time() - self.stats['uptime'],
            'coordinator_ip': self.local_ip
        }

    def register_worker(self, worker_socket, worker_info, worker_ip=None):
        """註冊新的工作節點"""
        with self.worker_lock:
            worker_id = len(self.worker_nodes)
            worker = {
                'id': worker_id,
                'socket': worker_socket,
                'info': worker_info,
                'busy': False,
                'last_active': time.time(),
                'discovery_ip': worker_ip  # 記錄 Worker 的 IP
            }
            self.worker_nodes.append(worker)
            logger.info(f"Worker {worker_id} registered from {worker_ip}: {worker_info}")
            
            if len(self.worker_nodes) == 1 and not self.dispatching:
                self.dispatching = True
                threading.Thread(target=self.dispatch_requests, daemon=True).start()
                
            return worker_id

    # ... 其他方法保持不變 ...
    def register_client(self, client_socket, addr):
        """註冊客戶端"""
        with self.client_lock:
            client_id = str(uuid.uuid4())[:8]
            self.clients[client_id] = {
                'id': client_id,
                'address': f"{addr[0]}:{addr[1]}",
                'connected_at': datetime.now().strftime('%H:%M:%S'),
                'requests_sent': 0
            }
            return client_id

    def unregister_client(self, client_id):
        """取消註冊客戶端"""
        with self.client_lock:
            if client_id in self.clients:
                del self.clients[client_id]

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

    def add_request_to_history(self, request_id, client_id, prompt, status='pending'):
        """添加請求到歷史記錄"""
        with self.history_lock:
            self.request_history.append({
                'id': request_id,
                'client_id': client_id,
                'prompt': prompt[:30] + '...' if len(prompt) > 30 else prompt,
                'status': status,
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'worker_id': None
            })
            if len(self.request_history) > 100:
                self.request_history = self.request_history[-100:]

    def update_request_status(self, request_id, status, worker_id=None):
        """更新請求狀態"""
        with self.history_lock:
            for record in reversed(self.request_history):
                if record['id'] == request_id:
                    record['status'] = status
                    if worker_id is not None:
                        record['worker_id'] = worker_id
                    break

    def dispatch_requests(self):
        """分發請求到可用的工作節點"""
        logger.info("Request dispatcher started")
        while True:
            try:
                client_socket, request_data, request_id, client_id = self.requests_queue.get()
                
                self.stats['total_requests'] += 1
                self.stats['active_requests'] += 1
                
                logger.info(f"Dispatching request {request_id}: {request_data.get('prompt', '')[:30]}...")
                
                worker = None
                while worker is None:
                    worker = self.get_available_worker()
                    if worker is None:
                        logger.info("No available workers, waiting...")
                        time.sleep(1)
                
                try:
                    self.update_request_status(request_id, 'processing', worker['id'])
                    
                    request_with_client = {
                        'request_id': request_id,
                        'client_id': id(client_socket),
                        'client_addr': client_socket.getpeername(),
                        'data': request_data
                    }
                    
                    request_json = json.dumps(request_with_client).encode('utf-8') + b'\n'
                    worker['socket'].sendall(request_json)
                    logger.info(f"Request {request_id} dispatched to worker {worker['id']}")
                    
                    threading.Thread(
                        target=self.handle_worker_response, 
                        args=(worker, client_socket, request_id),
                        daemon=True
                    ).start()
                    
                except Exception as e:
                    logger.error(f"Error dispatching request {request_id}: {e}")
                    self.release_worker(worker['id'])
                    self.stats['failed_requests'] += 1
                    self.stats['active_requests'] -= 1
                    self.update_request_status(request_id, 'failed')
                        
            except Exception as e:
                logger.error(f"Error in dispatch_requests: {e}")
                time.sleep(1)

    def handle_worker_response(self, worker, client_socket, request_id):
        """處理工作節點的響應並轉發給客戶端"""
        worker_socket = worker['socket']
        worker_id = worker['id']
        buffer = b''
        
        try:
            while True:
                try:
                    worker_socket.settimeout(30.0)
                    chunk = worker_socket.recv(4096)
                    
                    if not chunk:
                        logger.warning(f"Worker {worker_id} disconnected")
                        break
                    
                    buffer += chunk
                    
                    while b'\n' in buffer:
                        pos = buffer.find(b'\n')
                        message = buffer[:pos]
                        buffer = buffer[pos+1:]
                        
                        if not message or message == b'pong':
                            continue
                        
                        try:
                            response = json.loads(message.decode('utf-8'))
                            
                            response_json = json.dumps(response).encode('utf-8') + b'\n'
                            client_socket.sendall(response_json)
                            
                            if response.get('type') == 'end':
                                self.stats['completed_requests'] += 1
                                self.stats['active_requests'] -= 1
                                self.update_request_status(request_id, 'completed')
                                return
                            elif response.get('type') == 'error':
                                self.stats['failed_requests'] += 1
                                self.stats['active_requests'] -= 1
                                self.update_request_status(request_id, 'failed')
                                return
                                
                        except json.JSONDecodeError:
                            continue
                            
                except socket.timeout:
                    break
                    
        except Exception as e:
            logger.error(f"Error handling worker response: {e}")
        finally:
            self.release_worker(worker_id)

    def handle_client(self, client_socket, addr):
        """處理客戶端連接"""
        client_id = self.register_client(client_socket, addr)
        logger.info(f"Client {client_id} connected from {addr}")
        
        try:
            while True:
                buffer = b''
                while True:
                    try:
                        client_socket.settimeout(300.0)
                        chunk = client_socket.recv(4096)
                        if not chunk:
                            return
                        buffer += chunk
                        if b'\n' in buffer:
                            break
                    except:
                        return
                
                messages = buffer.split(b'\n')
                for message in messages:
                    if not message:
                        continue
                    
                    try:
                        request_data = json.loads(message.decode('utf-8'))
                        request_id = str(uuid.uuid4())[:8]
                        
                        with self.client_lock:
                            if client_id in self.clients:
                                self.clients[client_id]['requests_sent'] += 1
                        
                        self.add_request_to_history(request_id, client_id, request_data.get('prompt', ''))
                        self.requests_queue.put((client_socket, request_data, request_id, client_id))
                        
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON decode error from client {client_id}: {e}")
                        
        except Exception as e:
            logger.error(f"Error handling client {client_id}: {e}")
        finally:
            self.unregister_client(client_id)
            try:
                client_socket.close()
            except:
                pass

    def handle_worker_connection(self, worker_socket, addr):
        """處理工作節點的連接"""
        logger.info(f"Worker connected from {addr}")
        worker_id = None
        
        try:
            worker_socket.settimeout(10.0)
            buffer = b''
            
            while b'\n' not in buffer:
                chunk = worker_socket.recv(4096)
                if not chunk:
                    return
                buffer += chunk
            
            message = buffer.split(b'\n')[0]
            worker_info = json.loads(message.decode('utf-8'))
            worker_id = self.register_worker(worker_socket, worker_info, addr[0])
            
            confirmation = {'worker_id': worker_id, 'status': 'registered'}
            worker_socket.sendall(json.dumps(confirmation).encode('utf-8') + b'\n')
            
            # 心跳檢測
            while True:
                try:
                    worker_socket.settimeout(60.0)
                    worker_socket.sendall(b'ping\n')
                    data = worker_socket.recv(10)
                    if not data:
                        break
                    time.sleep(30)
                except:
                    break
                    
        except Exception as e:
            logger.error(f"Error handling worker connection: {e}")
        finally:
            if worker_id is not None:
                with self.worker_lock:
                    self.worker_nodes = [w for w in self.worker_nodes if w['id'] != worker_id]
                    logger.info(f"Worker {worker_id} removed")
            try:
                worker_socket.close()
            except:
                pass

    def start_web_server(self):
        """啟動 Web 服務器"""
        self.socketio.run(self.app, host='0.0.0.0', port=self.web_port, debug=False, allow_unsafe_werkzeug=True)

    def start(self):
        """啟動協調器服務"""
        # 啟動 Web 服務器
        web_thread = threading.Thread(target=self.start_web_server, daemon=True)
        web_thread.start()
        
        self.server_socket.listen(10)
        logger.info(f"Coordinator listening on {self.local_ip}:{self.port}")
        logger.info(f"Web UI available at http://{self.local_ip}:{self.web_port}")
        
        while True:
            try:
                client_socket, addr = self.server_socket.accept()
                
                try:
                    client_socket.settimeout(10.0)
                    first_byte = client_socket.recv(1)
                    
                    if first_byte == b'C':
                        threading.Thread(target=self.handle_client, args=(client_socket, addr), daemon=True).start()
                    elif first_byte == b'W':
                        threading.Thread(target=self.handle_worker_connection, args=(client_socket, addr), daemon=True).start()
                    else:
                        client_socket.close()
                        
                except:
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
