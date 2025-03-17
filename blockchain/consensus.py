import hashlib
import time
import json
import random
from typing import Dict, List, Any, Set, Tuple, Optional
import numpy as np
from scipy.stats import wasserstein_distance
import torch

class Vote:
    """投票类，用于DPoS选举"""
    def __init__(self, voter_id: str, candidate_id: str, weight: float):
        self.voter_id = voter_id
        self.candidate_id = candidate_id
        self.weight = weight  # 投票权重，基于声誉值
        self.timestamp = time.time()
        self.signature = self._sign()
    
    def _sign(self) -> str:
        """对投票进行签名"""
        vote_content = {
            'voter_id': self.voter_id,
            'candidate_id': self.candidate_id,
            'weight': self.weight,
            'timestamp': self.timestamp
        }
        vote_string = json.dumps(vote_content, sort_keys=True)
        return hashlib.sha256(vote_string.encode()).hexdigest()
    
    def to_dict(self) -> Dict:
        return {
            'voter_id': self.voter_id,
            'candidate_id': self.candidate_id,
            'weight': self.weight,
            'timestamp': self.timestamp,
            'signature': self.signature
        }

class SuperNode:
    """超级节点类，由DPoS选举产生"""
    def __init__(self, node_id: str, reputation: float):
        self.node_id = node_id
        self.reputation = reputation
        self.votes_received: List[Vote] = []  # 收到的投票
        self.total_votes = 0.0  # 总投票权重
    
    def add_vote(self, vote: Vote) -> None:
        """添加投票"""
        if vote.candidate_id == self.node_id:
            self.votes_received.append(vote)
            self.total_votes += vote.weight
    
    def to_dict(self) -> Dict:
        return {
            'node_id': self.node_id,
            'reputation': self.reputation,
            'total_votes': self.total_votes,
            'votes_received': [vote.to_dict() for vote in self.votes_received]
        }

class Transaction:
    """交易类，表示客户端提交的模型更新"""
    def __init__(self, client_id: str, model_update: Dict[str, Any], 
                 model_params: Dict[str, torch.Tensor], reputation: float):
        self.client_id = client_id
        self.model_update = model_update  # 序列化后的模型更新
        self.model_params = model_params  # 原始模型参数
        self.reputation = reputation
        self.timestamp = time.time()
        self.signature = self._sign()
    
    def _sign(self) -> str:
        """对交易进行签名"""
        # 使用模型更新的哈希值和其他信息生成签名
        model_update_str = json.dumps(self.model_update, sort_keys=True)
        model_update_hash = hashlib.sha256(model_update_str.encode()).hexdigest()
        
        tx_content = {
            'client_id': self.client_id,
            'model_update_hash': model_update_hash,
            'reputation': self.reputation,
            'timestamp': self.timestamp
        }
        tx_string = json.dumps(tx_content, sort_keys=True)
        return hashlib.sha256(tx_string.encode()).hexdigest()
    
    def to_dict(self) -> Dict:
        return {
            'client_id': self.client_id,
            'model_update': self.model_update,
            'reputation': self.reputation,
            'timestamp': self.timestamp,
            'signature': self.signature
        }
    
    def update_reputation(self, new_reputation: float):
        """更新信誉值"""
        self.reputation = new_reputation
        # 更新签名
        self.signature = self._sign()

class HotStuffMessage:
    """HotStuff协议消息类"""
    def __init__(self, sender_id: str, phase: str, block_hash: str, view_number: int):
        self.sender_id = sender_id
        self.phase = phase  # 'prepare', 'pre-commit', 'commit', 'decide'
        self.block_hash = block_hash
        self.view_number = view_number
        self.timestamp = time.time()
        self.signature = self._sign()
    
    def _sign(self) -> str:
        """对消息进行签名"""
        message_content = {
            'sender_id': self.sender_id,
            'phase': self.phase,
            'block_hash': self.block_hash,
            'view_number': self.view_number,
            'timestamp': self.timestamp
        }
        message_string = json.dumps(message_content, sort_keys=True)
        return hashlib.sha256(message_string.encode()).hexdigest()
    
    def to_dict(self) -> Dict:
        return {
            'sender_id': self.sender_id,
            'phase': self.phase,
            'block_hash': self.block_hash,
            'view_number': self.view_number,
            'timestamp': self.timestamp,
            'signature': self.signature
        }

class DPoSElection:
    """DPoS选举机制"""
    def __init__(self, clients_info: Dict[str, float], num_super_nodes: int = 3):
        self.clients_info = clients_info  # client_id -> reputation
        self.num_super_nodes = num_super_nodes
        self.candidates: Dict[str, SuperNode] = {}
        self.votes: List[Vote] = []
        self.elected_super_nodes: List[SuperNode] = []
    
    def nominate_candidates(self) -> None:
        """提名候选人，基于声誉值"""
        # 按声誉值排序
        sorted_clients = sorted(self.clients_info.items(), key=lambda x: x[1], reverse=True)
        
        # 选择声誉值最高的一部分客户端作为候选人
        num_candidates = min(len(sorted_clients), self.num_super_nodes * 2)
        
        for i in range(num_candidates):
            client_id, reputation = sorted_clients[i]
            self.candidates[client_id] = SuperNode(client_id, reputation)
    
    def vote(self) -> None:
        """所有客户端进行投票"""
        for voter_id, reputation in self.clients_info.items():
            # 每个客户端可以投票给一个候选人
            if self.candidates:  # 确保有候选人
                # 简单起见，这里随机选择一个候选人
                # 实际应用中可能基于更复杂的策略
                candidate_id = random.choice(list(self.candidates.keys()))
                
                # 创建投票，权重基于声誉值
                vote = Vote(voter_id, candidate_id, reputation)
                self.votes.append(vote)
                
                # 更新候选人收到的投票
                if candidate_id in self.candidates:
                    self.candidates[candidate_id].add_vote(vote)
    
    def elect_super_nodes(self) -> List[SuperNode]:
        """选举超级节点"""
        # 按总投票权重排序
        sorted_candidates = sorted(
            self.candidates.values(),
            key=lambda x: x.total_votes,
            reverse=True
        )
        
        # 选择投票最多的作为超级节点
        num_elected = min(len(sorted_candidates), self.num_super_nodes)
        self.elected_super_nodes = sorted_candidates[:num_elected]
        
        return self.elected_super_nodes
    
    def select_leader(self) -> str:
        """从超级节点中选择一个作为Leader"""
        if not self.elected_super_nodes:
            raise ValueError("No super nodes elected yet")
        
        # 简单起见，选择投票最多的超级节点作为Leader
        leader = max(self.elected_super_nodes, key=lambda x: x.total_votes)
        return leader.node_id

class HotStuffConsensus:
    """HotStuff共识协议实现"""
    def __init__(self, leader_id: str, super_node_ids: List[str], 
                 timeout: float = 30.0, max_retries: int = 3):
        self.leader_id = leader_id
        self.super_node_ids = super_node_ids
        self.view_number = 0
        self.current_phase = ""  # 当前阶段: prepare, pre-commit, commit, decide
        self.prepared_block_hash = ""
        self.pre_committed_block_hash = ""
        self.committed_block_hash = ""
        self.decided_block_hash = ""
        
        # 各阶段的消息
        self.prepare_messages: Dict[str, HotStuffMessage] = {}
        self.pre_commit_messages: Dict[str, HotStuffMessage] = {}
        self.commit_messages: Dict[str, HotStuffMessage] = {}
        self.decide_messages: Dict[str, HotStuffMessage] = {}
        
        # 超时和重试相关
        self.timeout = timeout  # 每个阶段的超时时间（秒）
        self.max_retries = max_retries  # 最大重试次数
        self.start_time = 0.0  # 当前阶段开始时间
    
    def reset_phase(self) -> None:
        """重置当前阶段"""
        self.prepare_messages.clear()
        self.pre_commit_messages.clear()
        self.commit_messages.clear()
        self.decide_messages.clear()
        self.current_phase = ""
        self.start_time = 0.0
    
    def start_phase_timer(self) -> None:
        """开始计时"""
        self.start_time = time.time()
    
    def is_phase_timeout(self) -> bool:
        """检查当前阶段是否超时"""
        return time.time() - self.start_time > self.timeout
    
    def start_consensus(self, block_hash: str) -> None:
        """开始共识过程"""
        self.view_number += 1
        self.current_phase = "prepare"
        self.prepared_block_hash = block_hash
        
        # Leader发送prepare消息
        prepare_msg = HotStuffMessage(
            sender_id=self.leader_id,
            phase="prepare",
            block_hash=block_hash,
            view_number=self.view_number
        )
        self.prepare_messages[self.leader_id] = prepare_msg
    
    def receive_prepare(self, message: HotStuffMessage) -> None:
        """接收prepare消息"""
        if message.phase != "prepare" or message.view_number != self.view_number:
            print(f"Invalid prepare message: phase={message.phase}, view={message.view_number}")
            return
        
        self.prepare_messages[message.sender_id] = message
        print(f"Received prepare message from {message.sender_id}, total: {len(self.prepare_messages)}")
        
        if self._has_quorum(self.prepare_messages):
            print("Prepare phase reached quorum")
            self.current_phase = "pre-commit"
            self.pre_committed_block_hash = self.prepared_block_hash
            
            # Leader发送pre-commit消息
            pre_commit_msg = HotStuffMessage(
                sender_id=self.leader_id,
                phase="pre-commit",
                block_hash=self.pre_committed_block_hash,
                view_number=self.view_number
            )
            self.pre_commit_messages[self.leader_id] = pre_commit_msg
    
    def receive_pre_commit(self, message: HotStuffMessage) -> None:
        """接收pre-commit消息"""
        if message.phase != "pre-commit" or message.view_number != self.view_number:
            return
        
        self.pre_commit_messages[message.sender_id] = message
        
        # 检查是否收到足够的pre-commit消息
        if self._has_quorum(self.pre_commit_messages):
            self.current_phase = "commit"
            self.committed_block_hash = self.pre_committed_block_hash
            
            # Leader发送commit消息
            commit_msg = HotStuffMessage(
                sender_id=self.leader_id,
                phase="commit",
                block_hash=self.committed_block_hash,
                view_number=self.view_number
            )
            self.commit_messages[self.leader_id] = commit_msg
    
    def receive_commit(self, message: HotStuffMessage) -> None:
        """接收commit消息"""
        if message.phase != "commit" or message.view_number != self.view_number:
            return
        
        self.commit_messages[message.sender_id] = message
        
        # 检查是否收到足够的commit消息
        if self._has_quorum(self.commit_messages):
            self.current_phase = "decide"
            self.decided_block_hash = self.committed_block_hash
            
            # Leader发送decide消息
            decide_msg = HotStuffMessage(
                sender_id=self.leader_id,
                phase="decide",
                block_hash=self.decided_block_hash,
                view_number=self.view_number
            )
            self.decide_messages[self.leader_id] = decide_msg
            
            return True  # 共识达成
        
        return False
    
    def receive_decide(self, message: HotStuffMessage) -> None:
        """接收decide消息"""
        if message.phase != "decide" or message.view_number != self.view_number:
            return
        
        self.decide_messages[message.sender_id] = message
    
    def _has_quorum(self, messages: Dict[str, HotStuffMessage]) -> bool:
        """检查是否达到法定人数(2f+1)"""
        f = (len(self.super_node_ids) - 1) // 3  # 最多容忍f个节点故障
        return len(messages) >= 2 * f + 1
    
    def is_consensus_reached(self) -> bool:
        """检查是否达成共识"""
        return self.current_phase == "decide" and self._has_quorum(self.decide_messages)
    
    def get_consensus_result(self) -> str:
        """获取共识结果"""
        if self.is_consensus_reached():
            return self.decided_block_hash
        return ""

class MemPool:
    """内存池，存储待处理的交易"""
    def __init__(self, max_size: int = 1000):
        self.transactions: Dict[str, Transaction] = {}  # tx_id -> Transaction
        self.max_size = max_size
    
    def _remove_lowest_reputation_transaction(self) -> None:
        """移除声誉值最低的交易"""
        if not self.transactions:
            return
        
        # 找到声誉值最低的交易
        lowest_tx = min(self.transactions.values(), key=lambda tx: tx.reputation)
        # 从内存池中移除该交易
        self.transactions.pop(lowest_tx.signature)
    
    def add_transaction(self, transaction: Transaction) -> bool:
        """添加交易到内存池"""
        if len(self.transactions) >= self.max_size:
            # 如果内存池已满，移除声誉值最低的交易
            self._remove_lowest_reputation_transaction()
        
        # 使用交易的签名作为唯一标识
        self.transactions[transaction.signature] = transaction
        return True
    
    def get_transaction(self, tx_id: str) -> Transaction:
        """根据交易ID获取交易"""
        return self.transactions.get(tx_id)
    
    def get_all_transactions(self) -> List[Transaction]:
        """获取所有交易"""
        return list(self.transactions.values())
    
    def remove_transaction(self, tx_id: str) -> bool:
        """移除指定交易"""
        if tx_id in self.transactions:
            del self.transactions[tx_id]
            return True
        return False

class Block:
    """区块类，用于共识过程"""
    def __init__(self, previous_hash: str, transactions: List[Transaction], 
                 creator_id: str, cluster_id: int, round_num: int,
                 model_params: Dict, clients_info: Dict[str, float],
                 super_nodes: List[Dict]):
        self.previous_hash = previous_hash
        self.transactions = transactions
        self.creator_id = creator_id
        self.cluster_id = cluster_id
        self.round_num = round_num
        self.model_params = model_params
        self.clients_info = clients_info
        self.super_nodes = super_nodes
        self.timestamp = time.time()
        self.hash = self._calculate_hash()

    def _calculate_hash(self) -> str:
        """计算区块哈希值"""
        block_content = {
            'previous_hash': self.previous_hash,
            'transactions': [tx.to_dict() for tx in self.transactions],
            'creator_id': self.creator_id,
            'cluster_id': self.cluster_id,
            'round_num': self.round_num,
            'model_params': self.model_params,
            'clients_info': self.clients_info,
            'super_nodes': self.super_nodes,
            'timestamp': self.timestamp
        }
        block_string = json.dumps(block_content, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()

    def to_dict(self) -> Dict:
        return {
            'previous_hash': self.previous_hash,
            'transactions': [tx.to_dict() for tx in self.transactions],
            'creator_id': self.creator_id,
            'cluster_id': self.cluster_id,
            'round_num': self.round_num,
            'model_params': self.model_params,
            'clients_info': self.clients_info,
            'super_nodes': self.super_nodes,
            'timestamp': self.timestamp,
            'hash': self.hash
        }

class BlockValidator:
    """区块验证器，验证区块的有效性"""
    def __init__(self, difficulty: int = 4):
        self.difficulty = difficulty
    
    def validate_block(self, block: Block, previous_block: Block = None) -> bool:
        """验证区块的有效性"""
        # 验证区块哈希
        if block.hash != block._calculate_hash():
            return False
        
        # 如果有前一个区块，验证previous_hash
        if previous_block and block.previous_hash != previous_block.hash:
            return False
        
        # 验证工作量证明
        if not self._check_proof_of_work(block.hash):
            return False
        
        # 验证交易
        return self._validate_transactions(block.transactions)
    
    def _check_proof_of_work(self, block_hash: str) -> bool:
        """检查工作量证明"""
        return block_hash.startswith('0' * self.difficulty)
    
    def _validate_transactions(self, transactions: List[Transaction]) -> bool:
        """验证交易列表的有效性"""
        # 这里可以添加更多的交易验证逻辑
        # 例如：检查交易签名、检查双重支付等
        return all(self._validate_transaction(tx) for tx in transactions)
    
    def _validate_transaction(self, transaction: Transaction) -> bool:
        """验证单个交易的有效性"""
        # 验证交易签名
        tx_content = {
            'client_id': transaction.client_id,
            'model_update': transaction.model_update,
            'reputation': transaction.reputation,
            'timestamp': transaction.timestamp
        }
        tx_string = json.dumps(tx_content, sort_keys=True)
        calculated_signature = hashlib.sha256(tx_string.encode()).hexdigest()
        
        return calculated_signature == transaction.signature

class ConsensusBlockChain:
    """共识区块链类，用于共识过程"""
    def __init__(self):
        self.chain: List[Block] = []
    
    def add_block(self, block: Block) -> None:
        """添加区块到区块链"""
        self.chain.append(block)
    
    def create_sub_block(self, cluster_id: int, round_num: int, 
                        model_params: Dict[str, Any], 
                        transactions: List[Transaction],
                        super_nodes: List[Dict]) -> Block:
        """创建子区块
        
        Args:
            cluster_id: 簇ID
            round_num: 轮次号
            model_params: 序列化后的模型参数
            transactions: 交易列表
            super_nodes: 超级节点列表
        """
        # 获取前一个区块的哈希值
        previous_hash = self.chain[-1].hash if self.chain else "0"
        
        # 创建新区块
        block = Block(
            previous_hash=previous_hash,
            transactions=transactions,
            creator_id=f"cluster_{cluster_id}",
            cluster_id=cluster_id,
            round_num=round_num,
            model_params=model_params,  # 已序列化的模型参数
            clients_info={tx.client_id: tx.reputation for tx in transactions},
            super_nodes=super_nodes
        )
        
        return block

class ModelValidator:
    """模型验证器，用于验证模型参数的相似度"""
    def __init__(self, super_node_model: Dict[str, torch.Tensor]):
        self.super_node_model = super_node_model
        
    def calculate_similarity_scores(self, worker_models: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, float]:
        """计算模型相似度得分"""
        scores = {}
        for worker_id, worker_model in worker_models.items():
            # 计算Wasserstein距离
            distance = 0
            for key in self.super_node_model:
                if key in worker_model:
                    super_param = self.super_node_model[key]
                    worker_param = worker_model[key]
                    
                    # 将参数展平并转换为numpy数组
                    super_flat = super_param.cpu().detach().numpy().flatten()
                    worker_flat = worker_param.cpu().detach().numpy().flatten()
                    
                    # 计算Wasserstein距离
                    distance += wasserstein_distance(
                        super_flat, worker_flat
                    )
            
            # 将距离转换为相似度得分（距离越小，得分越高）
            scores[worker_id] = 1.0 / (1.0 + distance)
        
        return scores

class SuperNodeConsensus:
    """超级节点共识类"""
    def __init__(self, node_id: str, model_params: Dict[str, torch.Tensor]):
        self.node_id = node_id
        self.model_params = model_params
        self.validator = ModelValidator(model_params)
        self.trusted_nodes: Set[str] = set()
        self.current_block: Optional[Block] = None
    
    def validate_worker_models(self, worker_models: Dict[str, Dict[str, torch.Tensor]]) -> None:
        """验证工作节点的模型并选择可信节点"""
        print(f"\n超级节点 {self.node_id} 开始验证工作节点模型...")
        
        # 计算相似度得分
        scores = self.validator.calculate_similarity_scores(worker_models)
        
        # 对得分排序
        sorted_scores = sorted(scores.items(), key=lambda x: x[1])
        
        # 选择中间2/3的节点
        n = len(sorted_scores)
        start_idx = n // 6
        end_idx = n - (n // 6)
        
        # 记录可信节点
        self.trusted_nodes = {node_id for node_id, _ in sorted_scores[start_idx:end_idx]}
        
        # 输出验证结果
        print(f"超级节点 {self.node_id} 验证完成:")
        print(f"- 总工作节点数: {len(worker_models)}")
        print(f"- 可信节点数: {len(self.trusted_nodes)}")
        print(f"- 可信节点列表: {sorted(list(self.trusted_nodes))}")
        
        # 输出得分分布
        print("\n得分分布:")
        score_ranges = {
            "高分(0.8-1.0)": 0,
            "中高分(0.6-0.8)": 0,
            "中分(0.4-0.6)": 0,
            "中低分(0.2-0.4)": 0,
            "低分(0.0-0.2)": 0
        }
        
        for _, score in scores.items():
            if score >= 0.8:
                score_ranges["高分(0.8-1.0)"] += 1
            elif score >= 0.6:
                score_ranges["中高分(0.6-0.8)"] += 1
            elif score >= 0.4:
                score_ranges["中分(0.4-0.6)"] += 1
            elif score >= 0.2:
                score_ranges["中低分(0.2-0.4)"] += 1
            else:
                score_ranges["低分(0.0-0.2)"] += 1
        
        for range_name, count in score_ranges.items():
            print(f"- {range_name}: {count}个节点")
    
    def validate_block(self, block):
        """
        验证区块是否通过共识
        
        Args:
            block: 待验证的区块
            
        Returns:
            bool: 如果区块中的可信节点列表与本节点的可信节点列表有足够重叠则返回True
        """
        # 获取Leader提议的区块中的可信节点列表
        leader_trusted_nodes = set()
        for transaction in block.transactions:
            if transaction.client_id in self.trusted_nodes:
                leader_trusted_nodes.add(transaction.client_id)
        
        # 计算本节点的可信节点列表与Leader提议的可信节点列表的交集大小
        common_trusted_nodes = self.trusted_nodes.intersection(leader_trusted_nodes)
        
        # 计算重叠率
        # 使用本节点的可信节点列表长度作为基准
        overlap_ratio = len(common_trusted_nodes) / len(self.trusted_nodes) if self.trusted_nodes else 0
        
        print(f"超级节点 {self.node_id} 验证区块:")
        print(f"- 本节点可信节点数量: {len(self.trusted_nodes)}")
        print(f"- Leader提议的可信节点数量: {len(leader_trusted_nodes)}")
        print(f"- 共同可信节点数量: {len(common_trusted_nodes)}")
        print(f"- 重叠率: {overlap_ratio:.2f}")
        
        # 如果重叠率大于等于0.5（即有一半及以上的节点重叠），则通过验证
        is_valid = overlap_ratio >= 0.5
        print(f"- 验证结果: {'通过' if is_valid else '未通过'}")
        
        return is_valid

class EnhancedHotStuffConsensus(HotStuffConsensus):
    """增强版HotStuff共识"""
    def __init__(self, super_nodes: Dict[str, SuperNodeConsensus], 
                 timeout: float = 30.0, max_retries: int = 5):
        self.super_nodes = super_nodes
        self.current_leader = None
        self.timeout = timeout
        self.max_retries = max_retries
        self.tried_leaders: Set[str] = set()
    
    def select_new_leader(self) -> Optional[str]:
        """选择新的Leader"""
        available_leaders = set(self.super_nodes.keys()) - self.tried_leaders
        if not available_leaders:
            return None
        new_leader = random.choice(list(available_leaders))
        print(f"\n选择新的Leader节点: {new_leader}")
        print(f"- 剩余可选Leader数: {len(available_leaders) - 1}")
        return new_leader
    
    def run_consensus(self, block: Block) -> Tuple[bool, str]:
        """运行共识流程"""
        round_count = 0
        print("\n开始增强版HotStuff共识流程...")
        print(f"- 总超级节点数: {len(self.super_nodes)}")
        print(f"- 需要票数: {len(self.super_nodes) * 2 // 3 + 1}")
        
        while len(self.tried_leaders) < len(self.super_nodes):
            round_count += 1
            print(f"\n=== 共识轮次 {round_count} ===")
            
            # 选择新的Leader
            self.current_leader = self.select_new_leader()
            if not self.current_leader:
                print("没有可用的Leader节点，共识失败")
                break
                
            self.tried_leaders.add(self.current_leader)
            
            # 收集投票
            print(f"\nLeader节点 {self.current_leader} 开始收集投票...")
            votes = {}
            for node_id, super_node in self.super_nodes.items():
                vote_result = super_node.validate_block(block)
                if vote_result:
                    votes[node_id] = True
            
            # 检查是否达到2/3多数
            vote_threshold = len(self.super_nodes) * 2 / 3
            print(f"\n投票结果统计:")
            print(f"- 同意票数: {len(votes)}")
            print(f"- 所需票数: {vote_threshold}")
            print(f"- 投票结果: {'通过' if len(votes) >= vote_threshold else '未通过'}")
            
            if len(votes) >= vote_threshold:
                print(f"\n共识达成！")
                print(f"- 最终Leader: {self.current_leader}")
                print(f"- 总轮次: {round_count}")
                return True, self.current_leader
            else:
                print(f"\n本轮共识未达成，切换Leader重试")
        
        print(f"\n共识最终失败")
        print(f"- 尝试轮次: {round_count}")
        print(f"- 已尝试所有可能的Leader节点")
        return False, ""

class SubBlock:
    """子区块类"""
    def __init__(self, 
                 cluster_id: int,
                 round_num: int,
                 prev_hash: str,
                 model_params: Dict[str, Any],
                 global_model_params: Dict[str, torch.Tensor],
                 transactions: List[Transaction],  # 使用transactions替代clients_info
                 super_nodes: List[Dict]):
        self.timestamp = time.time()
        self.cluster_id = cluster_id
        self.round_num = round_num
        self.prev_hash = prev_hash
        self.model_params = model_params
        self.global_model_params = global_model_params
        self.transactions = transactions
        self.super_nodes = super_nodes
        # 预计算模型参数哈希
        model_params_str = json.dumps(model_params, sort_keys=True)
        self.model_params_hash = hashlib.sha256(model_params_str.encode()).hexdigest()
        self.nonce = 0
        self.hash = self.calculate_hash()

    def calculate_hash(self) -> str:
        """计算子区块的哈希值"""
        block_content = {
            'timestamp': self.timestamp,
            'cluster_id': self.cluster_id,
            'round_num': self.round_num,
            'prev_hash': self.prev_hash,
            'model_params': self.model_params,
            'global_model_params': self.global_model_params,
            'transactions': [tx.to_dict() for tx in self.transactions],
            'super_nodes': self.super_nodes,
            'model_params_hash': self.model_params_hash,
            'nonce': self.nonce
        }
        block_string = json.dumps(block_content, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()