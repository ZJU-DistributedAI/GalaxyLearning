import torch
import os, sys
sys.path.append("C:\\Users\\tchennech\\Documents\\FederateLearningLibrary")
from torchvision import datasets, transforms
from gl.core.strategy import WorkModeStrategy
from gl.core.trainer_controller import TrainerController
from torch import nn
import torch.nn.functional as F

CLIENT_IP = "127.0.0.1"
CLIENT_PORT = 8081
CLIENT_ID = 0
SERVER_URL = "http://127.0.0.1:9763"

def start_trainer(work_mode, client_ip, client_port, client_id, server_url, data):

    TrainerController(work_mode, data, str(client_id), client_ip, str(client_port), server_url, 3).start()

if __name__ == "__main__":


    #CLIENT_ID = int(sys.argv[1])


    mnist_data = datasets.MNIST("./mnist_data", download=True, train=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.13066062,), (0.30810776,))
    ]))

    start_trainer(WorkModeStrategy.WORKMODE_STANDALONE, CLIENT_IP, CLIENT_PORT, CLIENT_ID, SERVER_URL, mnist_data)




