#!/usr/bin/env python3
import socket
import json
import threading
import time
import logging
import sys
import argparse
from datetime import datetime
import uuid
import select

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("StressTest")

class StressTestClient:
    def __init__(self, coordinator_host="localhost", coordinator_port=9000, 
                 num_clients=3, requests_per_client=2, delay_between_requests=1.0):
        self.coordinator_host = coordinator_host
        self.coordinator_port = coordinator_port
        self.num_clients = num_clients
        self.requests_per_client = requests_per_client
        self.delay_between_requests = delay_between_requests
        
        # 統計資料
        self.stats = {
            'total_requests': 0,
            'completed_requests': 0,
            'failed_requests': 0,
            'start_time': None,
            'end_time': None,
            'response_times': [],
            'errors': []
        }
        self.stats_lock = threading.Lock()
        
        # 測試提示詞
        self.test_prompts = [
            "你好，請介紹一下你自己",
            "什麼是人工智慧？",
            "請寫一個簡單的Python Hello World程式",
            "解釋一下機器學習的基本概念",
            "今天天氣如何？",
            "請推薦一些好看的電影",
            "如何學習程式設計？",
            "什麼是深度學習？",
            "請解釋量子計算的基本原理",
            "如何保持健康的生活方式？"
        ]

    def create_connection(self):
        """創建到協調器的連接"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((self.coordinator_host, self.coordinator_port))
            # 發送客戶端標記
            sock.sendall(b'C')
            logger.debug(f"Connected to coordinator at {self.coordinator_host}:{self.coordinator_port}")
            return sock
        except Exception as e:
            logger.error(f"Failed to connect to coordinator: {e}")
            return None

    def send_request(self, sock, prompt, max_length=512, temperature=0.7):
        """發送單個請求"""
        try:
            request_data = {
                'prompt': prompt,
                'max_length': max_length,
                'temperature': temperature,
                'timestamp': time.time()
            }
            
            request_json = json.dumps(request_data).encode('utf-8') + b'\n'
            sock.sendall(request_json)
            logger.debug(f"Sent request: {prompt[:30]}...")
            return True
        except Exception as e:
            logger.error(f"Error sending request: {e}")
            return False

    def receive_response(self, sock, request_start_time):
        """接收完整的串流響應"""
        buffer = b''
        response_complete = False
        generation_started = False
        full_response = ""
        
        try:
            while not response_complete:
                # 使用 select 檢查是否有數據可讀
                ready = select.select([sock], [], [], 30.0)  # 30秒超時
                if not ready[0]:
                    logger.warning("Response timeout")
                    return False, "Timeout", 0
                    
                chunk = sock.recv(4096)
                if not chunk:
                    logger.warning("Server disconnected")
                    return False, "Disconnected", 0
                
                buffer += chunk
                
                # 處理完整的消息
                while b'\n' in buffer:
                    pos = buffer.find(b'\n')
                    message = buffer[:pos]
                    buffer = buffer[pos+1:]
                    
                    if message:
                        try:
                            response = json.loads(message.decode('utf-8'))
                            
                            response_type = response.get('type', '')
                            if response_type == 'start':
                                generation_started = True
                                logger.debug("Response generation started")
                            elif response_type == 'token':
                                token = response.get('token', '')
                                full_response += token
                            elif response_type == 'end' and generation_started:
                                response_time = time.time() - request_start_time
                                total_length = response.get('total_length', len(full_response))
                                inference_time = response.get('inference_time', 0)
                                worker_id = response.get('worker_id', 'unknown')
                                
                                logger.info(f"Response completed: {total_length} chars, "
                                           f"{inference_time:.2f}s inference, "
                                           f"{response_time:.2f}s total, "
                                           f"worker {worker_id}")
                                
                                response_complete = True
                                return True, full_response, response_time
                            elif response_type == 'error':
                                error_msg = response.get('error', 'Unknown error')
                                logger.error(f"Server error: {error_msg}")
                                response_complete = True
                                return False, error_msg, time.time() - request_start_time
                                
                        except json.JSONDecodeError:
                            # 忽略非JSON消息
                            continue
                            
        except Exception as e:
            logger.error(f"Error receiving response: {e}")
            return False, str(e), time.time() - request_start_time

    def client_worker(self, client_id):
        """單個客戶端工作線程"""
        logger.info(f"Client {client_id} started")
        
        for request_num in range(self.requests_per_client):
            try:
                # 創建新連接
                sock = self.create_connection()
                if not sock:
                    with self.stats_lock:
                        self.stats['failed_requests'] += 1
                    continue
                
                # 選擇測試提示詞
                prompt_idx = (client_id * self.requests_per_client + request_num) % len(self.test_prompts)
                prompt = self.test_prompts[prompt_idx]
                
                logger.info(f"Client {client_id} sending request {request_num + 1}/{self.requests_per_client}: {prompt[:30]}...")
                
                # 記錄請求開始時間
                request_start_time = time.time()
                
                with self.stats_lock:
                    self.stats['total_requests'] += 1
                
                # 發送請求
                if self.send_request(sock, prompt):
                    # 接收響應
                    success, response, response_time = self.receive_response(sock, request_start_time)
                    
                    with self.stats_lock:
                        if success:
                            self.stats['completed_requests'] += 1
                            self.stats['response_times'].append(response_time)
                            logger.info(f"Client {client_id} request {request_num + 1} completed in {response_time:.2f}s")
                        else:
                            self.stats['failed_requests'] += 1
                            self.stats['errors'].append(f"Client {client_id}: {response}")
                            logger.error(f"Client {client_id} request {request_num + 1} failed: {response}")
                else:
                    with self.stats_lock:
                        self.stats['failed_requests'] += 1
                        self.stats['errors'].append(f"Client {client_id}: Failed to send request")
                
                # 關閉連接
                try:
                    sock.close()
                except:
                    pass
                
                # 延遲後發送下一個請求
                if request_num < self.requests_per_client - 1:
                    time.sleep(self.delay_between_requests)
                    
            except Exception as e:
                logger.error(f"Client {client_id} error: {e}")
                with self.stats_lock:
                    self.stats['failed_requests'] += 1
                    self.stats['errors'].append(f"Client {client_id}: {str(e)}")
        
        logger.info(f"Client {client_id} finished")

    def print_progress(self):
        """打印進度信息"""
        while True:
            time.sleep(5)  # 每5秒打印一次進度
            
            with self.stats_lock:
                total = self.stats['total_requests']
                completed = self.stats['completed_requests']
                failed = self.stats['failed_requests']
                
                if total > 0:
                    progress = ((completed + failed) / total) * 100
                    logger.info(f"Progress: {progress:.1f}% ({completed + failed}/{total}) - "
                               f"Success: {completed}, Failed: {failed}")
                
                if completed + failed >= total and total > 0:
                    break

    def run_stress_test(self):
        """執行壓力測試"""
        logger.info(f"Starting stress test:")
        logger.info(f"  Clients: {self.num_clients}")
        logger.info(f"  Requests per client: {self.requests_per_client}")
        logger.info(f"  Total requests: {self.num_clients * self.requests_per_client}")
        logger.info(f"  Delay between requests: {self.delay_between_requests}s")
        logger.info(f"  Target: {self.coordinator_host}:{self.coordinator_port}")
        logger.info("=" * 60)
        
        self.stats['start_time'] = time.time()
        
        # 啟動進度監控線程
        progress_thread = threading.Thread(target=self.print_progress, daemon=True)
        progress_thread.start()
        
        # 啟動所有客戶端線程
        threads = []
        for client_id in range(self.num_clients):
            thread = threading.Thread(target=self.client_worker, args=(client_id,))
            threads.append(thread)
            thread.start()
            
            # 錯開啟動時間，避免同時連接
            time.sleep(0.1)
        
        # 等待所有線程完成
        for thread in threads:
            thread.join()
        
        self.stats['end_time'] = time.time()
        
        # 打印最終統計
        self.print_final_stats()

    def print_final_stats(self):
        """打印最終統計結果"""
        total_time = self.stats['end_time'] - self.stats['start_time']
        
        logger.info("=" * 60)
        logger.info("STRESS TEST COMPLETED")
        logger.info("=" * 60)
        logger.info(f"Total time: {total_time:.2f} seconds")
        logger.info(f"Total requests: {self.stats['total_requests']}")
        logger.info(f"Completed requests: {self.stats['completed_requests']}")
        logger.info(f"Failed requests: {self.stats['failed_requests']}")
        
        if self.stats['completed_requests'] > 0:
            success_rate = (self.stats['completed_requests'] / self.stats['total_requests']) * 100
            logger.info(f"Success rate: {success_rate:.1f}%")
            
            response_times = self.stats['response_times']
            avg_response_time = sum(response_times) / len(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
            
            logger.info(f"Response times:")
            logger.info(f"  Average: {avg_response_time:.2f}s")
            logger.info(f"  Min: {min_response_time:.2f}s")
            logger.info(f"  Max: {max_response_time:.2f}s")
            
            requests_per_second = self.stats['completed_requests'] / total_time
            logger.info(f"Throughput: {requests_per_second:.2f} requests/second")
        
        if self.stats['errors']:
            logger.info(f"\nErrors encountered:")
            for error in self.stats['errors'][:10]:  # 只顯示前10個錯誤
                logger.info(f"  {error}")
            if len(self.stats['errors']) > 10:
                logger.info(f"  ... and {len(self.stats['errors']) - 10} more errors")

def main():
    parser = argparse.ArgumentParser(description='LLM Stress Test Client')
    parser.add_argument('--host', type=str, default='localhost', 
                       help='Coordinator host (default: localhost)')
    parser.add_argument('--port', type=int, default=9000, 
                       help='Coordinator port (default: 9000)')
    parser.add_argument('--clients', type=int, default=3, 
                       help='Number of concurrent clients (default: 3)')
    parser.add_argument('--requests', type=int, default=2, 
                       help='Requests per client (default: 2)')
    parser.add_argument('--delay', type=float, default=1.0, 
                       help='Delay between requests in seconds (default: 1.0)')
    
    args = parser.parse_args()
    
    try:
        stress_tester = StressTestClient(
            coordinator_host=args.host,
            coordinator_port=args.port,
            num_clients=args.clients,
            requests_per_client=args.requests,
            delay_between_requests=args.delay
        )
        stress_tester.run_stress_test()
    except KeyboardInterrupt:
        logger.info("Stress test interrupted by user")
    except Exception as e:
        logger.error(f"Stress test error: {e}")

if __name__ == "__main__":
    main()