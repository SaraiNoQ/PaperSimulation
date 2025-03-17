#!/usr/bin/env python
import os
import random
import json
import pickle
import argparse
import yaml
from json import JSONEncoder
from tqdm import tqdm

import numpy as np
import torch
import time
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

from fed_baselines.client_base import FedClient
from fed_baselines.server_base import FedServer

from postprocessing.recorder import Recorder
from preprocessing.baselines_dataloader import divide_data, divide_noniid_data
from utils.similarity_cal import WassersteinSimilarity, PearsonSimilarity, JSDistanceSimilarity, perform_spectral_clustering, visualize_clustering_results

from typing import Dict, Any, List
from blockchain.block_structure import BlockChain, ClientInfo
from blockchain.consensus import (
    DPoSElection, HotStuffConsensus, Vote, SuperNode, 
    Transaction, HotStuffMessage, ConsensusBlockChain, Block as ConsensusBlock,
    ModelValidator, SuperNodeConsensus, EnhancedHotStuffConsensus
)

# 用于将python对象序列化为json对象
json_types = (list, dict, str, int, float, bool, type(None))
class PythonObjectEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, json_types):
            return super().default(self, obj)
        return {'_python_object': pickle.dumps(obj).decode('latin-1')}

def fed_args():
    """
    Arguments for running federated learning baselines
    :return: Arguments for federated learning baselines
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, required=True, help='Yaml file for configuration')

    args = parser.parse_args()
    return args

def calculate_reputation(client_id: str, global_model_params: Dict[str, torch.Tensor], 
                       client_model_params: Dict[str, torch.Tensor], 
                       old_reputation: float = 0.5) -> float:
    """计算客户端信誉值
    
    Args:
        client_id: 客户端ID
        global_model_params: 簇内全局模型参数（median）
        client_model_params: 客户端上传的模型参数
        old_reputation: 历史信誉值，默认为0.5
        
    Returns:
        float: 更新后的信誉值
    """
    # 确保所有张量在同一设备上
    device = next(iter(global_model_params.values())).device
    
    # 计算模型参数差异
    total_diff = 0
    total_params = 0
    for key in global_model_params:
        if key in client_model_params:
            # 确保两个张量在同一设备上
            global_param = global_model_params[key].to(device)
            client_param = client_model_params[key].to(device)
            
            # 计算参数差异
            diff = torch.abs(client_param - global_param)
            total_diff += torch.sum(diff).item()
            total_params += diff.numel()
    
    # 计算平均差异
    avg_diff = total_diff / total_params if total_params > 0 else float('inf')
    
    # 计算得分，使用指数衰减
    k = 0.1  # 衰减系数，可以根据需要调整
    score = 100 * np.exp(-k * avg_diff)
    
    # 更新信誉值
    alpha = 0.7  # 平滑因子，可以根据需要调整
    reputation = alpha * old_reputation + (1 - alpha) * (score / 100)  # 将score归一化到[0,1]
    
    return float(reputation)

def serialize_model_params(model_params: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """序列化模型参数"""
    serialized_params = {}
    for key, value in model_params.items():
        # 将张量移动到CPU并转换为列表
        if isinstance(value, torch.Tensor):
            serialized_params[key] = value.cpu().detach().numpy().tolist()
        else:
            serialized_params[key] = value
    return serialized_params

def serialize_super_nodes(nodes):
    return [node.to_dict() for node in nodes] if nodes else []

def run_consensus(hotstuff_consensus: HotStuffConsensus, 
                 consensus_block: ConsensusBlock,  # 使用重命名后的类型
                 serialized_super_nodes: List[Dict], 
                 cluster_id: int) -> bool:
    """运行共识过程"""
    retry_count = 0
    while retry_count < hotstuff_consensus.max_retries:
        try:
            # 重置共识状态
            hotstuff_consensus.reset_phase()
            
            # 启动共识过程
            hotstuff_consensus.start_consensus(consensus_block.hash)
            hotstuff_consensus.start_phase_timer()
            
            # Prepare阶段
            prepare_votes = 0
            for node in serialized_super_nodes:
                if hotstuff_consensus.is_phase_timeout():
                    raise TimeoutError("Prepare phase timeout")
                    
                prepare_msg = HotStuffMessage(
                    sender_id=node['node_id'],
                    phase="prepare",
                    block_hash=consensus_block.hash,
                    view_number=hotstuff_consensus.view_number
                )
                hotstuff_consensus.receive_prepare(prepare_msg)
                prepare_votes += 1
                
                if hotstuff_consensus._has_quorum(hotstuff_consensus.prepare_messages):
                    break
            
            if not hotstuff_consensus._has_quorum(hotstuff_consensus.prepare_messages):
                raise Exception(f"Prepare phase failed: only received {prepare_votes} votes")
            
            # Pre-commit阶段
            hotstuff_consensus.start_phase_timer()
            pre_commit_votes = 0
            for node in serialized_super_nodes:
                if hotstuff_consensus.is_phase_timeout():
                    raise TimeoutError("Pre-commit phase timeout")
                    
                pre_commit_msg = HotStuffMessage(
                    sender_id=node['node_id'],
                    phase="pre-commit",
                    block_hash=consensus_block.hash,
                    view_number=hotstuff_consensus.view_number
                )
                hotstuff_consensus.receive_pre_commit(pre_commit_msg)
                pre_commit_votes += 1
                
                if hotstuff_consensus._has_quorum(hotstuff_consensus.pre_commit_messages):
                    break
            
            if not hotstuff_consensus._has_quorum(hotstuff_consensus.pre_commit_messages):
                raise Exception(f"Pre-commit phase failed: only received {pre_commit_votes} votes")
            
            # Commit阶段
            hotstuff_consensus.start_phase_timer()
            commit_votes = 0
            consensus_reached = False
            for node in serialized_super_nodes:
                if hotstuff_consensus.is_phase_timeout():
                    raise TimeoutError("Commit phase timeout")
                    
                commit_msg = HotStuffMessage(
                    sender_id=node['node_id'],
                    phase="commit",
                    block_hash=consensus_block.hash,
                    view_number=hotstuff_consensus.view_number
                )
                consensus_reached = hotstuff_consensus.receive_commit(commit_msg)
                commit_votes += 1
                
                if consensus_reached:
                    break
            
            if not consensus_reached:
                print(f"簇 {cluster_id} 共识未达成（尝试 {retry_count + 1}/{hotstuff_consensus.max_retries}）：")
                print(f"- 当前阶段: {hotstuff_consensus.current_phase}")
                print(f"- 活跃超级节点数: {len(serialized_super_nodes)}")
                print(f"- 已收到消息数: {len(hotstuff_consensus.commit_messages)}")
                raise Exception(f"Commit phase failed: only received {commit_votes} votes")
            
            return True
            
        except (TimeoutError, Exception) as e:
            print(f"簇 {cluster_id} 共识失败（尝试 {retry_count + 1}/{hotstuff_consensus.max_retries}）: {str(e)}")
            retry_count += 1
            time.sleep(1)  # 等待一秒后重试
    
    return False

def simulate_malicious_update(model_params: Dict[str, torch.Tensor], attack_type: str = "zero") -> Dict[str, torch.Tensor]:
    """
    模拟恶意节点的模型更新
    
    Args:
        model_params: 原始模型参数
        attack_type: 攻击类型，"zero"表示全0参数，"random"表示随机参数
    
    Returns:
        Dict[str, torch.Tensor]: 被污染的模型参数
    """
    malicious_params = {}
    for key, param in model_params.items():
        if attack_type == "zero":
            # 生成全0参数
            malicious_params[key] = torch.zeros_like(param)
        else:  # random
            # 生成随机参数
            malicious_params[key] = torch.randn_like(param)
    return malicious_params

def train_cluster(args):
    """
    单个簇的训练函数
    """
    cluster_id, members, client_dict, cluster_server, cluster_recorder, blockchain, consensus_blockchain, hotstuff_consensus, super_nodes, config, global_round = args
    
    try:
        # 获取设备信息
        device = next(iter(cluster_server.state_dict().values())).device
        
        # 将服务器模型移动到GPU
        cluster_server.model = cluster_server.model.to(device)
        
        # 获取簇内服务器的全局模型参数（训练前）
        old_model_params = {k: v.clone().to(device) for k, v in cluster_server.state_dict().items()}
        
        # 簇内每个节点进行本地训练
        client_states = {}
        client_transactions = []  # 存储客户端交易
        data_num = 0
        
        # 获取恶意节点配置
        malicious_ratio = config["system"].get("malicious_ratio", 0.0)
        attack_type = config["system"].get("attack_type", "zero")
        
        for client_id in members:
            try:
                # 获取簇内服务器的全局模型
                cluster_global_state = cluster_server.state_dict()
                
                # 确保客户端模型在GPU上
                client_dict[client_id].model = client_dict[client_id].model.to(device)
                
                # 更新使用簇内全局模型
                client_dict[client_id].update(cluster_global_state)
                
                # 本地训练得到模型参数、数据量和loss
                state_dict, n_data, loss = client_dict[client_id].train()

                # 模拟恶意节点攻击
                if random.random() < malicious_ratio:
                    print(f"客户端 {client_id} 被检测为恶意节点，执行{attack_type}攻击")
                    state_dict = simulate_malicious_update(state_dict, attack_type)
                    # 恶意节点的loss设置为一个较小的值，以试图欺骗系统
                    loss = 0.1

                # 记录数据量
                data_num += n_data
                
                # 创建交易对象，包含模型更新
                transaction = Transaction(
                    client_id=client_id,
                    model_update=serialize_model_params(state_dict),
                    model_params=state_dict,  # 添加原始模型参数
                    reputation=0.5 if global_round == 0 else next(
                        (tx.reputation for tx in client_transactions if tx.client_id == client_id), 
                        0.5
                    )  # 使用历史信誉值或初始值
                )
                client_transactions.append(transaction)
                
                # 确保状态字典中的所有张量都在正确的设备上
                state_dict = {k: v.to(device) for k, v in state_dict.items()}
                client_states[client_id] = (state_dict, int(n_data), float(loss))
                
                # 将客户端模型移回CPU并清理内存
                client_dict[client_id].model = client_dict[client_id].model.cpu()
                if hasattr(client_dict[client_id], 'optimizer'):
                    client_dict[client_id].optimizer = None
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error training client {client_id} in cluster {cluster_id}: {str(e)}")
                continue

        # 计算超级节点数量（至少5个，且为总节点数的20%）
        num_super_nodes = max(5, int(len(members) * 0.2))
        
        # 执行DPoS选举
        dpos_election = DPoSElection(
            clients_info={
                tx.client_id: tx.reputation 
                for tx in client_transactions
            },
            num_super_nodes=num_super_nodes
        )
        dpos_election.nominate_candidates()
        dpos_election.vote()
        
        # 选举超级节点
        super_nodes = dpos_election.elect_super_nodes()
        super_node_ids = [node.node_id for node in super_nodes]
        
        # 收集所有工作节点的模型参数
        worker_models = {
            client_id: state_dict
            for client_id, (state_dict, _, _) in client_states.items()
            if client_id not in super_node_ids
        }
        
        # 为每个超级节点创建共识实例
        super_node_consensus = {}
        try:
            for super_node in super_nodes:
                node_id = super_node.node_id
                if node_id not in client_states:
                    print(f"警告: 超级节点 {node_id} 的状态未找到，跳过该节点")
                    continue
                    
                model_params = client_states[node_id][0]  # 获取超级节点的模型参数
                super_node_consensus[node_id] = SuperNodeConsensus(node_id, model_params)
                
                # 验证工作节点的模型
                if worker_models:  # 确保有工作节点模型可供验证
                    super_node_consensus[node_id].validate_worker_models(worker_models)
                else:
                    print(f"警告: 没有工作节点模型供超级节点 {node_id} 验证")
            
            # 检查是否有足够的超级节点参与共识
            if not super_node_consensus:
                raise Exception("没有有效的超级节点可以参与共识")
            
            # 创建增强版HotStuff共识实例
            enhanced_consensus = EnhancedHotStuffConsensus(super_node_consensus)
            
        except Exception as e:
            print(f"创建超级节点共识实例时出错: {str(e)}")
            raise e
        
        # 创建初始区块（不包含聚合结果）
        initial_block = consensus_blockchain.create_sub_block(
            cluster_id=int(cluster_id),
            round_num=len(cluster_recorder.res['server']['iid_accuracy']),
            model_params={},  # 暂时为空
            transactions=client_transactions,
            super_nodes=[node.to_dict() for node in super_nodes]
        )
        
        # 运行共识
        consensus_reached, leader_id = enhanced_consensus.run_consensus(initial_block)
        
        if consensus_reached:
            print(f"簇 {cluster_id} 达成共识，Leader: {leader_id}")
            
            # 获取达成共识的区块中记录的可信节点ID
            trusted_nodes = set()
            for node_id, super_node in super_node_consensus.items():
                if super_node.validate_block(initial_block):
                    trusted_nodes.update(super_node.trusted_nodes)
            
            print(f"\n使用可信节点进行聚合:")
            print(f"- 可信节点数量: {len(trusted_nodes)}")
            print(f"- 可信节点列表: {sorted(list(trusted_nodes))}")
            
            # 清空服务器当前的客户端状态
            cluster_server.flush()
            
            # 只记录可信节点的模型参数到服务器
            trusted_data_num = 0
            for client_id, (state_dict, n_data, loss) in client_states.items():
                if client_id in trusted_nodes:
                    # 使用rec方法记录可信节点的信息
                    cluster_server.rec(client_id, state_dict, n_data, loss)
                    trusted_data_num += n_data
            
            # 更新总数据量为可信节点的数据量
            data_num = trusted_data_num
            print(f"- 可信节点总数据量: {data_num}")
            
            # 进行模型聚合（只使用可信节点的参数）
            cluster_server.select_clients()
            cluster_global_state, avg_loss, _ = cluster_server.agg()

            # 更新每个客户端的信誉值
            for tx in client_transactions:
                if tx.client_id in trusted_nodes:
                    # 计算新的信誉值
                    new_reputation = calculate_reputation(
                        tx.client_id,
                        cluster_global_state,  # 簇内全局模型
                        tx.model_params,       # 客户端原始模型参数
                        tx.reputation          # 当前信誉值
                    )
                    # 更新交易中的信誉值
                    tx.update_reputation(new_reputation)

            # 更新区块中的模型参数
            serializable_params = serialize_model_params(cluster_global_state)
            consensus_block = consensus_blockchain.create_sub_block(
                cluster_id=int(cluster_id),
                round_num=len(cluster_recorder.res['server']['iid_accuracy']),
                model_params=serializable_params,
                transactions=client_transactions,
                super_nodes=[node.to_dict() for node in super_nodes]
            )
            
            # 将共识区块添加到链上
            consensus_blockchain.add_block(consensus_block)
            print(f'簇 {cluster_id} 区块创建完成，共识验证通过')

            # 保存子区块内容到JSON文件
            sub_block_content = {
                'basic_info': {
                    'cluster_id': cluster_id,
                    'round_num': consensus_block.round_num,
                    'timestamp': consensus_block.timestamp,
                    'hash': consensus_block.hash,
                    'prev_hash': consensus_block.previous_hash
                },
                'transactions': [
                    {
                        'client_id': tx.client_id,
                        'reputation': tx.reputation,
                        'timestamp': tx.timestamp,
                        'signature': tx.signature
                    }
                    for tx in consensus_block.transactions
                ],
                'super_nodes': [
                    {
                        'node_id': node['node_id'],
                        'reputation': node['reputation'],
                        'total_votes': node['total_votes'],
                        'votes_received': [
                            {
                                'voter_id': vote['voter_id'],
                                'weight': vote['weight'],
                                'timestamp': vote['timestamp'],
                                'signature': vote['signature']
                            }
                            for vote in node['votes_received']
                        ]
                    }
                    for node in consensus_block.super_nodes
                ],
                'model_info': {
                    'num_layers': len(consensus_block.model_params),
                    'total_parameters': sum(np.array(param).size for param in consensus_block.model_params.values()),
                    'layer_shapes': {
                        key: np.array(param).shape 
                        for key, param in consensus_block.model_params.items()
                    }
                }
            }

            # 创建保存路径
            sub_block_dir = os.path.join(
                config["system"]["res_root"],
                'sub_blocks',
                f'cluster_{cluster_id}'
            )
            os.makedirs(sub_block_dir, exist_ok=True)

            # 保存区块内容到JSON文件
            sub_block_filename = os.path.join(
                sub_block_dir,
                f'sub_block_round_{consensus_block.round_num}.json'
            )
            with open(sub_block_filename, 'w') as f:
                json.dump(sub_block_content, f, indent=2, cls=PythonObjectEncoder)

            print(f'子区块内容已保存到: {sub_block_filename}')
            
        else:
            print(f"簇 {cluster_id} 未能达成共识，使用所有节点进行聚合")
            # 清空服务器当前的客户端状态
            cluster_server.flush()
            
            # 记录所有节点的信息和计算总数据量
            data_num = 0
            for client_id, (state_dict, n_data, loss) in client_states.items():
                cluster_server.rec(client_id, state_dict, n_data, loss)
                data_num += n_data
            
            print(f"- 使用所有节点总数据量: {data_num}")
            
            # 使用所有节点的模型参数进行聚合
            cluster_server.select_clients()
            cluster_global_state, avg_loss, _ = cluster_server.agg()
        
        # 簇内测试和记录
        accuracy = cluster_server.test()
        
        # 记录簇内结果
        cluster_recorder.res['server']['iid_accuracy'].append(float(accuracy))
        cluster_recorder.res['server']['train_loss'].append(float(avg_loss))
        
        # 记录每个客户端的结果
        for client_id, (_, _, loss) in client_states.items():
            if client_id not in cluster_recorder.res['clients']:
                cluster_recorder.res['clients'][client_id] = {
                    'train_loss': [],
                    'iid_accuracy': []
                }
            cluster_recorder.res['clients'][client_id]['train_loss'].append(float(loss))
            # 使用簇的准确率作为客户端的准确率
            cluster_recorder.res['clients'][client_id]['iid_accuracy'].append(float(accuracy))

        # 簇内训练完毕，并且完成记录后，清空服务器当前的客户端状态
        cluster_server.flush()

        # 序列化全局模型
        serializable_params = serialize_model_params(cluster_global_state)
        
        # 序列化超级节点列表
        serialized_super_nodes = serialize_super_nodes(super_nodes)
        
        # 创建共识区块
        consensus_block = consensus_blockchain.create_sub_block(
            cluster_id=int(cluster_id),
            round_num=len(cluster_recorder.res['server']['iid_accuracy']),
            model_params=serializable_params,
            transactions=client_transactions,
            super_nodes=serialized_super_nodes
        )
        
        # 运行共识过程
        consensus_success = run_consensus(
            hotstuff_consensus, 
            consensus_block, 
            serialized_super_nodes,
            cluster_id
        )
        
        if not consensus_success:
            raise Exception(f"Failed to reach consensus after {hotstuff_consensus.max_retries} attempts")
        
        # 将共识区块添加到共识链
        consensus_blockchain.add_block(consensus_block)
        print(f'簇 {cluster_id} 区块创建完成，共识验证通过')
        
        # 将服务器模型移回CPU并清理内存
        cluster_server.model = cluster_server.model.cpu()
        torch.cuda.empty_cache()
        
        return {
            'cluster_id': int(cluster_id),
            'global_state': cluster_global_state,
            'avg_loss': float(avg_loss),
            'accuracy': float(accuracy),
            # 'n_members': int(len(members)),
            'n_members': int(data_num),
            'sub_block': consensus_block
        }
        
    except Exception as e:
        print(f"Error in cluster {cluster_id} training: {str(e)}")
        # 确保发生错误时也清理GPU内存
        if hasattr(cluster_server, 'model'):
            cluster_server.model = cluster_server.model.cpu()
        torch.cuda.empty_cache()
        raise e

def train_clusters_sequential(clusters, client_dict, cluster_servers, cluster_recorders, blockchain, consensus_blockchain, hotstuff_consensus, super_nodes, config, global_round):
    """
    顺序训练所有簇
    """
    cluster_results = []
    for cluster_id, members in clusters.items():
        try:
            # 在每个簇开始训练前清理GPU内存
            torch.cuda.empty_cache()
            
            result = train_cluster((cluster_id, members, client_dict, 
                                  cluster_servers[cluster_id], 
                                  cluster_recorders[cluster_id],
                                  blockchain,
                                  consensus_blockchain,
                                  hotstuff_consensus,
                                  super_nodes,
                                  config,
                                  global_round))
            cluster_results.append(result)
            
            # 在每个簇训练后清理GPU内存
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Cluster {cluster_id} training failed: {str(e)}")
            # 确保发生错误时也清理GPU内存
            torch.cuda.empty_cache()
    return cluster_results

def train_clusters_concurrent(clusters, client_dict, cluster_servers, cluster_recorders, blockchain, consensus_blockchain, num_threads, hotstuff_consensus, super_nodes, config, global_round):
    """
    并发训练所有簇
    """
    cluster_results = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # 准备每个簇的训练参数
        cluster_args = [
            (cluster_id, members, client_dict, cluster_servers[cluster_id], 
             cluster_recorders[cluster_id], blockchain, consensus_blockchain, 
             hotstuff_consensus, super_nodes, config, global_round)
            for cluster_id, members in clusters.items()
        ]
        
        # 并行执行簇内训练
        future_to_cluster = {
            executor.submit(train_cluster, args): args[0]
            for args in cluster_args
        }
        
        # 收集训练结果
        for future in concurrent.futures.as_completed(future_to_cluster):
            try:
                result = future.result()
                cluster_results.append(result)
            except Exception as e:
                cluster_id = future_to_cluster[future]
                print(f"Cluster {cluster_id} training failed: {str(e)}")
    
    return cluster_results

def fed_run():
    args = fed_args()
    with open(args.config, "r") as yaml_file:
        try:
            config = yaml.safe_load(yaml_file)
        except yaml.YAMLError as exc:
            print(exc)

    algo_list = ["FedAvg"]
    assert config["client"]["fed_algo"] in algo_list, "The federated learning algorithm is not supported"

    dataset_list = ["MNIST", "CIFAR10", "CIFAR100", "FashionMNIST", "SVHN"]
    assert config["system"]["dataset"] in dataset_list, "The dataset is not supported"

    model_list = ["LeNet", "AlexCifarNet", "CNN", "ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152"]
    assert config["system"]["model"] in model_list, "The model is not supported"

    # 初始化随机数
    np.random.seed(config["system"]["i_seed"])
    torch.manual_seed(config["system"]["i_seed"])
    random.seed(config["system"]["i_seed"])

    client_dict = {}
    recorder = Recorder()

    # trainset_config {'users': ['user_id1', ...], 'user_data': {'user_id1': train_data, ...}, 'num_samples': number}
    if (config["system"]["noniid"]):
        trainset_config, testset = divide_noniid_data(num_client=config["system"]["num_client"], 
                                        imbalance_factor=config["system"]["imbalance_factor"], 
                                        dataset_name=config["system"]["dataset"],
                                        i_seed=config["system"]["i_seed"])
    else:
        trainset_config, testset = divide_data(num_client=config["system"]["num_client"], 
                                            num_local_class=config["system"]["num_local_class"], 
                                            dataset_name=config["system"]["dataset"],
                                            i_seed=config["system"]["i_seed"])
    
    max_acc = 0

    # client_id: f00000-f00xxx
    # 初始化
    for client_id in trainset_config['users']:
        client_dict[client_id] = FedClient(client_id, 
                                       dataset_id=config["system"]["dataset"], 
                                       epoch=config["client"]["num_local_epoch"], 
                                       model_name=config["system"]["model"])
        client_dict[client_id].load_trainset(trainset_config['user_data'][client_id])
    
    fed_server = FedServer(trainset_config['users'], 
                           dataset_id=config["system"]["dataset"], 
                           model_name=config["system"]["model"])
    fed_server.load_testset(testset)
    # 初始化全局模型
    global_state_dict = fed_server.state_dict()

    start_time = time.time()
    print("开始初始训练和聚类分析...")

    # 1. 初始训练，收集所有客户端的模型
    client_models = {}
    for client_id in trainset_config['users']:
        # 更新使用全局模型
        client_dict[client_id].update(global_state_dict)
        # 本地训练得到模型参数
        state_dict, n_data, loss = client_dict[client_id].train()
        client_models[client_id] = state_dict

    # 2. 使用JS散度相似度计算器进行聚类
    n_clients = len(trainset_config['users'])
    client_ids = list(trainset_config['users'])
    
    print("计算模型相似度矩阵...")

    if (config["system"]["similarity_method"] == "wasserstein"):
        wasserstein_calculator = WassersteinSimilarity(n_clients, client_models, client_ids)
        similarity_matrix = wasserstein_calculator.compute_similarity_matrix()
    elif (config["system"]["similarity_method"] == "pearson"):
        pearson_calculator = PearsonSimilarity(n_clients, client_models, client_ids)
        similarity_matrix = pearson_calculator.compute_similarity_matrix()
    elif (config["system"]["similarity_method"] == "js"):
        js_calculator = JSDistanceSimilarity(n_clients, client_models, client_ids)
        similarity_matrix = js_calculator.compute_similarity_matrix()

    # 3. 执行谱聚类
    n_clusters = 3  # 可以根据需要调整
    cluster_labels = perform_spectral_clustering(similarity_matrix, n_clusters)

    cluster_time = time.time() - start_time
    print(f"谱聚类 运算时间: {cluster_time:.4f} 秒")
    
    # 4. 输出聚类结果
    print("\n客户端聚类结果:")
    print("-" * 50)
    clusters = {}
    for i, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(client_ids[i])

    for cluster_id, members in clusters.items():
        print(f"\n簇 {cluster_id}:")
        print(f"成员数量: {len(members)}")
        print("客户端列表:", members)

    # 5. 可视化聚类结果
    visualize_clustering_results(similarity_matrix, cluster_labels, n_clusters, config["system"]["similarity_method"])

    # 保存聚类结果
    clustering_results = {
        'similarity_matrix': similarity_matrix.tolist(),
        'cluster_labels': cluster_labels.tolist(),
        'client_ids': client_ids,
        'clusters': {str(k): v for k, v in clusters.items()}
    }

    # 将结果保存到JSON文件
    clustering_file = os.path.join(
        config["system"]["res_root"],
        f'clustering_results_{config["system"]["dataset"]}_{config["system"]["num_client"]}_{config["system"]["similarity_method"]}_{config["system"]["imbalance_factor"]}_{config["system"]["i_seed"]}.json'
    )
    with open(clustering_file, 'w') as f:
        json.dump(clustering_results, f, indent=2)

    print(f"\n聚类结果已保存到: {clustering_file}")

    # 继续原有的联邦学习训练循环
    print("\n开始分层联邦学习训练...")

    # 为每个簇创建服务器和记录器
    cluster_servers = {}
    cluster_recorders = {}
    cluster_max_acc = {}
    
    # 创建顶层服务器用于簇间聚合
    top_level_server = FedServer(
        list(clusters.keys()),  # 使用簇ID作为客户端ID
        dataset_id=config["system"]["dataset"],
        model_name=config["system"]["model"]
    )
    top_level_server.load_testset(testset)
    top_level_recorder = Recorder()
    top_level_max_acc = 0
    
    for cluster_id in clusters.keys():
        # 为每个簇创建服务器
        cluster_servers[cluster_id] = FedServer(
            clusters[cluster_id],  # 该簇的客户端列表
            dataset_id=config["system"]["dataset"],
            model_name=config["system"]["model"]
        )
        # 加载测试集
        cluster_servers[cluster_id].load_testset(testset)
        # 创建记录器
        cluster_recorders[cluster_id] = Recorder()
        cluster_max_acc[cluster_id] = 0

    # 初始化主区块链实例（用于存储）
    blockchain = BlockChain(difficulty=2)
    # 初始化共识区块链-子区块链实例（用于共识）
    consensus_blockchain = ConsensusBlockChain()
    
    # 初始化DPoS选举（使用所有客户端的初始信誉值）
    initial_dpos_election = DPoSElection(
        clients_info={client_id: 1.0 for client_id in trainset_config['users']}
    )
    initial_dpos_election.nominate_candidates()
    initial_dpos_election.vote()
    
    # 选举初始超级节点
    initial_super_nodes = initial_dpos_election.elect_super_nodes()
    
    # 选择初始leader节点
    initial_leader_id = initial_dpos_election.select_leader()
    initial_super_node_ids = [node.node_id for node in initial_super_nodes]
    
    # 初始化HotStuff共识
    hotstuff_consensus = HotStuffConsensus(initial_leader_id, initial_super_node_ids)
    
    # 并行训练每个簇
    pbar = tqdm(range(config["system"]["num_round"]))
    for global_round in pbar:
        cluster_metrics = {}
        
        # 根据配置决定使用顺序训练还是并发训练
        if config["system"]["concurrent"]:
            num_threads = len(clusters)  # 每个簇一个线程
            cluster_results = train_clusters_concurrent(
                clusters, client_dict, cluster_servers, 
                cluster_recorders, blockchain, consensus_blockchain,
                num_threads, hotstuff_consensus, initial_super_nodes,
                config, global_round
            )
        else:
            cluster_results = train_clusters_sequential(
                clusters, client_dict, cluster_servers, 
                cluster_recorders, blockchain, consensus_blockchain,
                hotstuff_consensus, initial_super_nodes,
                config, global_round
            )
        
        # 处理训练结果
        for result in cluster_results:
            cluster_id = result['cluster_id']
            
            # 更新簇的最大准确率
            if cluster_max_acc[cluster_id] < result['accuracy']:
                cluster_max_acc[cluster_id] = result['accuracy']
            
            # 保存簇的指标用于显示
            cluster_metrics[cluster_id] = {
                'loss': result['avg_loss'],
                'accuracy': result['accuracy'],
                'max_acc': cluster_max_acc[cluster_id]
            }
            
            # 将簇的全局模型提交给顶层服务器
            top_level_server.rec(
                cluster_id,
                result['global_state'],
                result['n_members'],
                result['avg_loss']
            )
        
        # 簇间聚合
        top_level_server.select_clients()
        global_state_dict, global_avg_loss, _ = top_level_server.agg()
        
        # 创建主区块
        # 序列化全局模型参数
        serializable_global_params = serialize_model_params(global_state_dict)
        print('开始创建主区块...')
        try:
            main_block = blockchain.create_main_block(
                round_num=global_round,
                global_model_params=serializable_global_params
            )
            print('主区块创建完成')
        except Exception as e:
            print(f"创建主区块时出错: {str(e)}")
            raise e
        
        # 更新每个簇的全局模型
        for cluster_id in clusters.keys():
            cluster_servers[cluster_id].update(global_state_dict)
        
        # 测试全局模型性能
        global_accuracy = top_level_server.test()
        top_level_server.flush()
        
        # 记录全局结果
        top_level_recorder.res['server']['iid_accuracy'].append(float(global_accuracy))
        top_level_recorder.res['server']['train_loss'].append(float(global_avg_loss))
        
        # 更新全局最大准确率
        if top_level_max_acc < global_accuracy:
            top_level_max_acc = global_accuracy
        
        # 更新进度条显示所有指标
        display_str = f'Global Round: {global_round}'
        for cluster_id, metrics in cluster_metrics.items():
            display_str += f' | Cluster {cluster_id} - '
            display_str += f'Loss: {metrics["loss"]:.4f} '
            display_str += f'Acc: {metrics["accuracy"]:.4f} '
            display_str += f'Max: {metrics["max_acc"]:.4f}'
        display_str += f' | Global - Loss: {global_avg_loss:.4f} Acc: {global_accuracy:.4f} Max: {top_level_max_acc:.4f}'
        pbar.set_description(display_str)
        
        # 保存结果
        if not os.path.exists(config["system"]["res_root"]):
            os.makedirs(config["system"]["res_root"])

        # 保存簇内结果
        for cluster_id, recorder in cluster_recorders.items():
            result_filename = os.path.join(
                config["system"]["res_root"], 
                f'[\'cluster_{cluster_id}_{config["client"]["fed_algo"]}\','
                f'\'cluster_{cluster_id}_{config["system"]["model"]}\',' 
                f'{config["client"]["num_local_epoch"]},'
                f'{config["system"]["num_local_class"]},'
                f'{config["system"]["i_seed"]}]'
            )
            with open(result_filename, "w") as jsfile:
                json.dump(recorder.res, jsfile, cls=PythonObjectEncoder)
        
        # 保存全局结果
        global_result_filename = os.path.join(
            config["system"]["res_root"], 
            f'[\'global_{config["client"]["fed_algo"]}\','
            f'\'global_{config["system"]["model"]}\',' 
            f'{config["client"]["num_local_epoch"]},'
            f'{config["system"]["num_local_class"]},'
            f'{config["system"]["i_seed"]}]'
        )
        with open(global_result_filename, "w") as jsfile:
            json.dump(top_level_recorder.res, jsfile, cls=PythonObjectEncoder)

        # 保存区块链状态
        blockchain_state_filename = os.path.join(
            config["system"]["res_root"],
            f'blockchain_state_round_{global_round}.json'
        )
        with open(blockchain_state_filename, 'w') as f:
            json.dump(blockchain.get_chain_info(), f, indent=2)

    # 验证区块链完整性
    if blockchain.verify_chain():
        print("\n区块链验证成功！")
    else:
        print("\n警告：区块链验证失败！")

def deserialize_model_params(serialized_params: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """反序列化模型参数"""
    model_params = {}
    for key, value in serialized_params.items():
        if isinstance(value, list):  # 如果是列表（序列化的张量）
            model_params[key] = torch.tensor(value)
        else:
            model_params[key] = value
    return model_params

if __name__ == "__main__":
    fed_run()
