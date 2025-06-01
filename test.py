import torch
import argparse
import logging
import sys
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_model_inference(model_path, gpu_id=None, test_prompts=None):
    """
    直接測試模型推理功能
    """
    

    if not os.path.exists(model_path):
        logger.error(f"模型路徑不存在: {model_path}")
        return False
    

    if not torch.cuda.is_available():
        logger.info("CUDA跑不了，CPU")
        device = torch.device("cpu")
        gpu_id = None
    else:
        gpu_count = torch.cuda.device_count()
        logger.info(f"發現 {gpu_count} 個GPU")
        
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
        
        if gpu_id is not None:
            if gpu_id >= gpu_count or gpu_id < 0:
                logger.error(f"GPU {gpu_id} 不可用。可用GPU: {list(range(gpu_count))}")
                return False
            device = torch.device(f"cuda:{gpu_id}")
        else:
            device = torch.device("cuda:0")
            gpu_id = 0
        
        gpu_name = torch.cuda.get_device_name(gpu_id)
        gpu_memory = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
        logger.info(f"使用 GPU {gpu_id}: {gpu_name} ({gpu_memory:.1f} GB)")
    
    try:
        # 載入tokenizer
        logger.info("載入tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # 設置pad_token
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            logger.info("設置pad_token_id為eos_token_id")
        
        # 載入模型
        logger.info("載入模型...")
        if device.type == 'cuda':
            logger.info("以FP16載入模型到GPU...")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map=f"cuda:{gpu_id}"  # 直接指定設備
            )
        else:
            logger.info("以FP32載入模型到CPU...")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            model = model.to(device)
        
        logger.info("模型載入成功！")
        
        # 顯示模型資訊
        logger.info(f"模型類型: {type(model)}")
        logger.info(f"模型設備: {next(model.parameters()).device}")
        logger.info(f"模型精度: {next(model.parameters()).dtype}")
        logger.info(f"詞彙表大小: {len(tokenizer)}")
        logger.info(f"模型詞彙表大小: {model.config.vocab_size}")
        
        # 顯示記憶體使用
        if device.type == 'cuda':
            memory_used = torch.cuda.memory_allocated(device) / 1024**3
            memory_total = torch.cuda.get_device_properties(device).total_memory / 1024**3
            logger.info(f"GPU記憶體使用: {memory_used:.2f}/{memory_total:.1f} GB")
        
        # 預設測試prompt
        if test_prompts is None:
            test_prompts = [
                "你好",
                "你好，請介紹一下你自己",
                "什麼是人工智慧？",
                "請用一句話總結機器學習的概念。",
                "Hello, how are you today?"
            ]
        
        # 測試不同的生成參數組合
        test_configs = [
            {
                "name": "確定性生成 (greedy)",
                "params": {
                    "do_sample": False,
                    "max_new_tokens": 100,
                    "pad_token_id": tokenizer.eos_token_id,
                    "eos_token_id": tokenizer.eos_token_id
                }
            },
            {
                "name": "低溫度採樣",
                "params": {
                    "do_sample": True,
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "top_k": 40,
                    "max_new_tokens": 100,
                    "repetition_penalty": 1.05,
                    "pad_token_id": tokenizer.eos_token_id,
                    "eos_token_id": tokenizer.eos_token_id
                }
            },
            {
                "name": "中等溫度採樣",
                "params": {
                    "do_sample": True,
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "top_k": 50,
                    "max_new_tokens": 100,
                    "repetition_penalty": 1.1,
                    "pad_token_id": tokenizer.eos_token_id,
                    "eos_token_id": tokenizer.eos_token_id
                }
            }
        ]
        

        for prompt in test_prompts:
            print("\n" + "="*80)
            print(f"測試prompt: '{prompt}'")
            print("="*80)
            

            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            input_length = inputs['input_ids'].shape[1]
            
            logger.info(f"輸入token數量: {input_length}")
            
            for config in test_configs:
                print(f"\n--- {config['name']} ---")
                
                try:
                    with torch.no_grad():
                        outputs = model.generate(
                            input_ids=inputs['input_ids'],
                            attention_mask=inputs.get('attention_mask'),
                            **config['params']
                        )
                    

                    generated_ids = outputs[0][input_length:]
                    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                    
                    print(f"生成結果: {generated_text}")
                    print(f"生成token數: {len(generated_ids)}")
                    

                    
                except Exception as e:
                    print(f"生成失敗: {e}")
                    logger.error(f"生成錯誤: {e}")
        

        if device.type == 'cuda':
            memory_used = torch.cuda.memory_allocated(device) / 1024**3
            print(f"\n最終GPU記憶體使用: {memory_used:.2f} GB")
        
        print("\n" + "="*80)
        print("測試完成！")
        print("="*80)
        
        return True
        
    except Exception as e:
        logger.error(f"測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description='直接測試LLM模型推理')
    parser.add_argument('--model', type=str, required=True, help='模型路徑')
    parser.add_argument('--gpu', type=int, default=None, help='指定GPU ID (例如: 0, 1)')
    parser.add_argument('--prompt', type=str, action='append', help='自定義測試prompt (可多次使用)')
    parser.add_argument('--verbose', action='store_true', help='詳細輸出')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("LLM模型直接推理測試")
    print(f"模型路徑: {args.model}")
    if args.gpu is not None:
        print(f"指定GPU: {args.gpu}")
    else:
        print("GPU: 自動選擇")
    print("-" * 50)
    
    success = test_model_inference(
        model_path=args.model,
        gpu_id=args.gpu,
        test_prompts=args.prompt
    )
    
    if success:


        sys.exit(0)
    else:

        sys.exit(1)

if __name__ == "__main__":
    main()