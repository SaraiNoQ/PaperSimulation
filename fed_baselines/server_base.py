from utils.models import *
import torch
from torch.utils.data import DataLoader
from utils.fed_utils import assign_dataset, init_model

class FedServer(object):
    def __init__(self, client_list, dataset_id, model_name):
        """
        Initialize the server for federated learning.
        :param client_list: List of the connected clients in networks
        :param dataset_id: Dataset name for the application scenario
        :param model_name: Machine learning model name for the application scenario
        """
        # Initialize the dict and list for system settings
        self.client_state = {}
        self.client_loss = {}
        self.client_n_data = {}
        self.selected_clients = []
        # batch size for testing
        self._batch_size = 200
        self.client_list = client_list

        # Initialize the test dataset
        self.testset = None

        # Initialize the hyper parameter for federated learning in the server
        self.round = 0
        self.n_data = 0
        self._dataset_id = dataset_id

        # Testing on GPU
        gpu = 0
        self._device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() and gpu != -1 else "cpu")

        # Initialize the global machine learning model
        self._num_class, self._image_dim, self._image_channel = assign_dataset(dataset_id)
        self.model_name = model_name
        self.model = init_model(model_name=self.model_name, num_class=self._num_class, image_channel=self._image_channel)

    def load_testset(self, testset):
        """
        Server loads the test dataset.
        :param data: Dataset for testing.
        """
        self.testset = testset

    def state_dict(self):
        """
        Server returns global model dict.
        :return: Global model dict
        """
        return self.model.state_dict()

    def test(self):
        """
        Server tests the model on test dataset.
        """
        test_loader = DataLoader(self.testset, batch_size=self._batch_size, shuffle=True)
        self.model.to(self._device)
        accuracy_collector = 0
        for step, (x, y) in enumerate(test_loader):
            with torch.no_grad():
                b_x = x.to(self._device)  # Tensor on GPU
                b_y = y.to(self._device)  # Tensor on GPU

                test_output = self.model(b_x)
                pred_y = torch.max(test_output, 1)[1].to(self._device).data.squeeze()
                accuracy_collector = accuracy_collector + sum(pred_y == b_y)
        accuracy = accuracy_collector / len(self.testset)

        return accuracy.cpu().numpy()

    def select_clients(self, connection_ratio=1):
        """
        Server selects a fraction of clients.
        :param connection_ratio: connection ratio in the clients
        """
        # select a fraction of clients
        self.selected_clients = []
        self.n_data = 0
        for client_id in self.client_list:
            b = np.random.binomial(np.ones(1).astype(int), connection_ratio)
            if b:
                self.selected_clients.append(client_id)
                self.n_data += self.client_n_data[client_id]

    def agg(self):
        """
        Server aggregates models from connected clients.
        :return: model_state: Updated global model after aggregation
        :return: avg_loss: Averaged loss value
        :return: n_data: Number of the local data points
        """
        # 1. 获取参与聚合的客户端数量
        client_num = len(self.selected_clients)
        # 2. 若没有客户端或数据量为0，直接返回空模型
        if client_num == 0 or self.n_data == 0:
            return self.model.state_dict(), 0, 0

        # 3. 初始化一个空模型用于聚合
        model = init_model(model_name=self.model_name, num_class=self._num_class, image_channel=self._image_channel)
        model_state = model.state_dict() # 获取空模型的初始参数
        avg_loss = 0 # 初始化平均损失

        # Aggregate the local updated models from selected clients
        # 4. 遍历所有选中的客户端进行模型聚合
        for i, name in enumerate(self.selected_clients):
            # 跳过未上传模型的客户端
            if name not in self.client_state:
                continue

            # 遍历模型的每一层参数（如 "conv1.weight", "fc.bias" 等）
            for key in self.client_state[name]:
                # --- 核心逻辑：加权平均 ---
                # 情况1：第一个客户端的参数初始化
                if i == 0:
                    model_state[key] = self.client_state[name][key] * self.client_n_data[name] / self.n_data
                # 情况2：后续客户端的参数累加
                else:
                    model_state[key] = model_state[key] + self.client_state[name][key] * self.client_n_data[
                        name] / self.n_data

            # 计算加权平均损失
            avg_loss = avg_loss + self.client_loss[name] * self.client_n_data[name] / self.n_data

        # 5. 将聚合后的参数加载到全局模型
        self.model.load_state_dict(model_state)
        self.round = self.round + 1 # 更新训练轮次
        n_data = self.n_data # 总数据量

        return model_state, avg_loss, n_data

    def rec(self, name, state_dict, n_data, loss):
        """
        Server receives the local updates from the connected client k.
        :param name: Name of client k
        :param state_dict: Model dict from the client k
        :param n_data: Number of local data points in the client k
        :param loss: Loss of local training in the client k
        """
        self.n_data = self.n_data + n_data
        self.client_state[name] = {}
        self.client_n_data[name] = {}

        self.client_state[name].update(state_dict)
        self.client_n_data[name] = n_data
        self.client_loss[name] = {}
        self.client_loss[name] = loss

    def flush(self):
        """
        Flushing the client information in the server
        """
        self.n_data = 0
        self.client_state = {}
        self.client_n_data = {}
        self.client_loss = {}

    def update(self, model_state_dict):
        """
        Update server's model with new state dictionary.
        :param model_state_dict: New model state dictionary to update with
        """
        self.model = init_model(model_name=self.model_name, num_class=self._num_class, image_channel=self._image_channel)
        self.model.load_state_dict(model_state_dict)
