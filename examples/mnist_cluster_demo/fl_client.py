import torch
import os, sys
from torchvision import datasets, transforms
sys.path.append("C:\\Users\\tchennech\\Documents\\FederateLearningLibrary")
from gl.core.strategy import WorkModeStrategy
from gl.core.trainer_controller import TrainerController
from torch import nn
import torch.nn.functional as F


SERVER_URL = "http://127.0.0.1:9763"
CLIENT_IP = "127.0.0.1"
CLIENT_PORT = 8081
CLIENT_ID = 0

def start_trainer(work_mode, client_id, client_ip, client_port, server_url, data):
    TrainerController(work_mode, data, str(client_id), client_ip, str(client_port), server_url, 5).start()
    #print(os.path.abspath("."))

if __name__ == "__main__":
    mnist_data = datasets.MNIST("./mnist_data", download=True, train=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.13066062,), (0.30810776,))
    ]))

    #start_trainer(WorkModeStrategy.WORKMODE_STANDALONE, CLIENT_ID, mnist_data)

    start_trainer(WorkModeStrategy.WORKMODE_CLUSTER, CLIENT_ID, CLIENT_IP, CLIENT_PORT, SERVER_URL, mnist_data)




