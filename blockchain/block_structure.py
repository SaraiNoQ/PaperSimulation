import hashlib
import time
import json
import torch
from typing import Dict, List, Any

class ClientInfo:
    """客户端信息类"""
    def __init__(self, client_id: str, reputation: float):
        self.client_id = hashlib.sha256(client_id.encode()).hexdigest()  # 客户端ID的哈希值
        self.reputation = reputation  # 信誉值

    def to_dict(self) -> Dict:
        return {
            'client_id': self.client_id,
            'reputation': self.reputation
        }

class SubBlock:
    """子区块类"""
    def __init__(self, 
                 cluster_id: int,
                 round_num: int,
                 prev_hash: str,
                 model_params: Dict[str, Any],
                 clients_info: List[ClientInfo]):
        self.timestamp = time.time()
        self.cluster_id = cluster_id
        self.round_num = round_num
        self.prev_hash = prev_hash
        self.model_params = model_params
        # 预计算模型参数哈希
        model_params_str = json.dumps(model_params, sort_keys=True)
        self.model_params_hash = hashlib.sha256(model_params_str.encode()).hexdigest()
        self.clients_info = clients_info
        self.nonce = 0
        self.hash = self.calculate_hash()

    def calculate_hash(self) -> str:
        """计算区块哈希值"""
        # 使用预计算的模型参数哈希
        block_content = {
            'timestamp': self.timestamp,
            'cluster_id': self.cluster_id,
            'round_num': self.round_num,
            'prev_hash': self.prev_hash,
            'model_params_hash': self.model_params_hash,
            'clients_info': [client.to_dict() for client in self.clients_info],
            'nonce': self.nonce
        }
        block_string = json.dumps(block_content, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()

    def mine_block(self, difficulty: int):
        """挖掘区块"""
        target = '0' * difficulty
        max_attempts = 1000000  # 限制最大尝试次数
        
        for i in range(max_attempts):
            self.nonce = i
            self.hash = self.calculate_hash()
            if self.hash[:difficulty] == target:
                return
            
            # 每1000次尝试打印一次进度
            if i % 1000 == 0:
                print(f"Mining progress: {i} attempts")
        
        print("Warning: Maximum mining attempts reached")

    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'cluster_id': self.cluster_id,
            'round_num': self.round_num,
            'prev_hash': self.prev_hash,
            'hash': self.hash,
            'model_params': self.model_params,
            'clients_info': [client.to_dict() for client in self.clients_info],
            'nonce': self.nonce
        }

class MainBlock:
    """主区块类"""
    def __init__(self,
                 round_num: int,
                 prev_hash: str,
                 sub_blocks: List[SubBlock],
                 global_model_params: Dict[str, Any]):
        self.timestamp = time.time()
        self.round_num = round_num
        self.prev_hash = prev_hash
        self.sub_blocks_pointers = [block.hash for block in sub_blocks]
        self.global_model_params = global_model_params
        
        # 预计算所有哈希值
        self.global_model_params_hash = hashlib.sha256(
            json.dumps(global_model_params, sort_keys=True).encode()
        ).hexdigest()
        self.sub_chain_cid = self.calculate_sub_chain_cid(sub_blocks)
        self.nonce = 0
        self.hash = self.calculate_hash()

    def calculate_sub_chain_cid(self, sub_blocks: List[SubBlock]) -> str:
        """计算子区块链的CID"""
        sub_blocks_content = [block.to_dict() for block in sub_blocks]
        sub_blocks_str = json.dumps(sub_blocks_content, sort_keys=True)
        return hashlib.sha256(sub_blocks_str.encode()).hexdigest()

    def calculate_hash(self) -> str:
        """计算区块哈希值"""
        block_content = {
            'timestamp': self.timestamp,
            'round_num': self.round_num,
            'prev_hash': self.prev_hash,
            'sub_blocks_pointers': self.sub_blocks_pointers,
            'global_model_params_hash': self.global_model_params_hash,
            'sub_chain_cid': self.sub_chain_cid,
            'nonce': self.nonce
        }
        block_string = json.dumps(block_content, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()

    def mine_block(self, difficulty: int):
        """挖掘区块"""
        target = '0' * difficulty
        max_attempts = 1000000  # 限制最大尝试次数
        
        print(f"开始挖掘主区块（轮次 {self.round_num}）...")
        for i in range(max_attempts):
            self.nonce = i
            self.hash = self.calculate_hash()
            if self.hash[:difficulty] == target:
                print(f"主区块挖掘成功（轮次 {self.round_num}）")
                return
            
            # 每1000次尝试打印一次进度
            if i % 10000 == 0:
                print(f"主区块挖掘进度: {i} attempts")
        
        print(f"警告：主区块（轮次 {self.round_num}）达到最大挖掘尝试次数")

    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'round_num': self.round_num,
            'prev_hash': self.prev_hash,
            'hash': self.hash,
            'sub_blocks_pointers': self.sub_blocks_pointers,
            'global_model_params': self.global_model_params,
            'sub_chain_cid': self.sub_chain_cid,
            'nonce': self.nonce
        }

class BlockChain:
    """区块链类"""
    def __init__(self, difficulty: int = 4):
        self.difficulty = difficulty
        self.sub_chains: Dict[int, List[SubBlock]] = {}  # cluster_id -> sub_blocks
        self.main_chain: List[MainBlock] = []
        self.pending_sub_blocks: Dict[int, SubBlock] = {}  # cluster_id -> latest_sub_block

    def create_sub_block(self,
                        cluster_id: int,
                        round_num: int,
                        model_params: Dict[str, Any],
                        clients_info: List[ClientInfo]) -> SubBlock:
        """创建新的子区块"""
        prev_hash = '0' * 64 if cluster_id not in self.sub_chains or not self.sub_chains[cluster_id] \
                   else self.sub_chains[cluster_id][-1].hash
        
        new_block = SubBlock(cluster_id, round_num, prev_hash, model_params, clients_info)
        new_block.mine_block(self.difficulty)
        
        if cluster_id not in self.sub_chains:
            self.sub_chains[cluster_id] = []
        self.sub_chains[cluster_id].append(new_block)
        self.pending_sub_blocks[cluster_id] = new_block
        
        return new_block

    def create_main_block(self,
                         round_num: int,
                         global_model_params: Dict[str, Any]) -> MainBlock:
        """创建新的主区块"""
        prev_hash = '0' * 64 if not self.main_chain else self.main_chain[-1].hash
        sub_blocks = list(self.pending_sub_blocks.values())
        
        print(f"开始创建主区块（轮次 {round_num}）...")
        try:
            new_block = MainBlock(round_num, prev_hash, sub_blocks, global_model_params)
            new_block.mine_block(self.difficulty)
            
            self.main_chain.append(new_block)
            self.pending_sub_blocks.clear()
            
            print(f"主区块创建完成（轮次 {round_num}）")
            return new_block
            
        except Exception as e:
            print(f"创建主区块时出错（轮次 {round_num}）: {str(e)}")
            raise e

    def verify_chain(self) -> bool:
        """验证区块链的完整性"""
        # 验证主链
        for i in range(1, len(self.main_chain)):
            current_block = self.main_chain[i]
            previous_block = self.main_chain[i-1]
            
            if current_block.prev_hash != previous_block.hash:
                return False
            if current_block.hash != current_block.calculate_hash():
                return False
        
        # 验证子链
        for cluster_chain in self.sub_chains.values():
            for i in range(1, len(cluster_chain)):
                current_block = cluster_chain[i]
                previous_block = cluster_chain[i-1]
                
                if current_block.prev_hash != previous_block.hash:
                    return False
                if current_block.hash != current_block.calculate_hash():
                    return False
        
        return True

    def get_chain_info(self) -> Dict:
        """获取区块链信息"""
        return {
            'main_chain_length': len(self.main_chain),
            'sub_chains_info': {
                cluster_id: len(chain)
                for cluster_id, chain in self.sub_chains.items()
            },
            'latest_main_block_hash': self.main_chain[-1].hash if self.main_chain else None,
            'pending_sub_blocks': list(self.pending_sub_blocks.keys())
        } 