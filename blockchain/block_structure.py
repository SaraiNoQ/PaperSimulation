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
            
            # 每100次尝试打印一次进度
            if i % 100 == 0:
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

class BlockHeader:
    """区块头类"""
    def __init__(self,
                 height: int,
                 prev_hash: str,
                 timestamp: float,
                 merkle_root: str,
                 nonce: int = 0):
        self.height = height          # 区块高度
        self.prev_hash = prev_hash    # 前一个区块的哈希
        self.timestamp = timestamp    # 时间戳
        self.merkle_root = merkle_root  # Merkle树根哈希
        self.nonce = nonce           # 工作量证明的随机数
        self.hash = self.calculate_hash()

    def calculate_hash(self) -> str:
        """计算区块头的哈希值"""
        header_content = {
            'height': self.height,
            'prev_hash': self.prev_hash,
            'timestamp': self.timestamp,
            'merkle_root': self.merkle_root,
            'nonce': self.nonce
        }
        header_string = json.dumps(header_content, sort_keys=True)
        return hashlib.sha256(header_string.encode()).hexdigest()

    def to_dict(self) -> Dict:
        return {
            'height': self.height,
            'prev_hash': self.prev_hash,
            'timestamp': self.timestamp,
            'merkle_root': self.merkle_root,
            'nonce': self.nonce,
            'hash': self.hash
        }

class MainBlock:
    """主区块类"""
    def __init__(self,
                 round_num: int,
                 prev_hash: str,
                 sub_blocks: List[SubBlock],
                 global_model_params: Dict[str, Any]):
        # 计算Merkle树根
        self.merkle_root = self._calculate_merkle_root(sub_blocks)
        
        # 创建区块头
        self.header = BlockHeader(
            height=round_num,
            prev_hash=prev_hash,
            timestamp=time.time(),
            merkle_root=self.merkle_root
        )
        
        # 区块体
        self.body = {
            'sub_chain_cid': self._calculate_sub_chain_cid(sub_blocks),
            'global_model_params': global_model_params,
            'sub_blocks_pointers': [block.hash for block in sub_blocks]
        }
        
        # 预计算所有哈希值
        self.global_model_params_hash = hashlib.sha256(
            json.dumps(global_model_params, sort_keys=True).encode()
        ).hexdigest()
        
        # 区块哈希就是区块头的哈希
        self.hash = self.header.hash

    def _calculate_merkle_root(self, sub_blocks: List[SubBlock]) -> str:
        """计算Merkle树根哈希"""
        if not sub_blocks:
            return hashlib.sha256('empty'.encode()).hexdigest()
        
        # 获取所有子区块的哈希作为叶子节点
        leaves = [block.hash for block in sub_blocks]
        
        # 如果叶子节点数量为奇数，复制最后一个节点
        if len(leaves) % 2 == 1:
            leaves.append(leaves[-1])
        
        # 构建Merkle树
        while len(leaves) > 1:
            temp = []
            for i in range(0, len(leaves), 2):
                combined = leaves[i] + leaves[i+1]
                temp.append(hashlib.sha256(combined.encode()).hexdigest())
            leaves = temp
            
            # 如果剩余节点为奇数，复制最后一个节点
            if len(leaves) % 2 == 1 and len(leaves) > 1:
                leaves.append(leaves[-1])
        
        return leaves[0]

    def _calculate_sub_chain_cid(self, sub_blocks: List[SubBlock]) -> str:
        """计算子区块链的CID"""
        sub_blocks_content = [block.to_dict() for block in sub_blocks]
        sub_blocks_str = json.dumps(sub_blocks_content, sort_keys=True)
        return hashlib.sha256(sub_blocks_str.encode()).hexdigest()

    def mine_block(self, difficulty: int):
        """挖掘区块"""
        target = '0' * difficulty
        max_attempts = 1000000  # 限制最大尝试次数
        
        print(f"开始挖掘主区块（高度 {self.header.height}）...")
        for i in range(max_attempts):
            self.header.nonce = i
            self.header.hash = self.header.calculate_hash()
            self.hash = self.header.hash
            
            if self.hash[:difficulty] == target:
                print(f"主区块挖掘成功（高度 {self.header.height}）")
                return
            
            # 每100次尝试打印一次进度
            if i % 100 == 0:
                print(f"主区块挖掘进度: {i} attempts")
        
        print(f"警告：主区块（高度 {self.header.height}）达到最大挖掘尝试次数")

    def to_dict(self) -> Dict:
        return {
            'header': self.header.to_dict(),
            'body': {
                'sub_chain_cid': self.body['sub_chain_cid'],
                'global_model_params': self.body['global_model_params'],
                'sub_blocks_pointers': self.body['sub_blocks_pointers']
            }
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
        
        try:
            new_block = MainBlock(round_num, prev_hash, sub_blocks, global_model_params)
            new_block.mine_block(self.difficulty)
            
            self.main_chain.append(new_block)
            self.pending_sub_blocks.clear()

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
        """获取区块链详细信息"""
        return {
            'main_chain': {
                'length': len(self.main_chain),
                'latest_block': {
                    'height': self.main_chain[-1].header.height if self.main_chain else None,
                    'hash': self.main_chain[-1].hash if self.main_chain else None,
                    'timestamp': self.main_chain[-1].header.timestamp if self.main_chain else None,
                    'merkle_root': self.main_chain[-1].header.merkle_root if self.main_chain else None,
                    'num_sub_blocks': len(self.main_chain[-1].body['sub_blocks_pointers']) if self.main_chain else 0
                } if self.main_chain else None,
                'blocks_summary': [
                    {
                        'height': block.header.height,
                        'hash': block.hash,
                        'prev_hash': block.header.prev_hash,
                        'num_sub_blocks': len(block.body['sub_blocks_pointers'])
                    }
                    for block in self.main_chain
                ]
            },
            'sub_chains': {
                cluster_id: {
                    'length': len(chain),
                    'latest_block': {
                        'round_num': chain[-1].round_num if chain else None,
                        'hash': chain[-1].hash if chain else None,
                        'timestamp': chain[-1].timestamp if chain else None,
                        'num_clients': len(chain[-1].clients_info) if chain else 0
                    } if chain else None,
                    'blocks_summary': [
                        {
                            'round_num': block.round_num,
                            'hash': block.hash,
                            'prev_hash': block.prev_hash,
                            'num_clients': len(block.clients_info)
                        }
                        for block in chain
                    ]
                }
                for cluster_id, chain in self.sub_chains.items()
            },
            'pending_sub_blocks': {
                cluster_id: {
                    'round_num': block.round_num,
                    'hash': block.hash,
                    'timestamp': block.timestamp,
                    'num_clients': len(block.clients_info)
                }
                for cluster_id, block in self.pending_sub_blocks.items()
            },
            'statistics': {
                'total_main_blocks': len(self.main_chain),
                'total_sub_chains': len(self.sub_chains),
                'total_pending_blocks': len(self.pending_sub_blocks),
                'sub_chains_lengths': {
                    cluster_id: len(chain)
                    for cluster_id, chain in self.sub_chains.items()
                },
                'difficulty': self.difficulty
            }
        } 