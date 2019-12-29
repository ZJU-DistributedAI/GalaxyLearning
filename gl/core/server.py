
import threading
import time
import os
from concurrent.futures import ThreadPoolExecutor
from gl.core.aggregator import FedAvgAggregator
from gl.core.strategy import WorkModeStrategy, FedrateStrategy
from gl.core import communicate_server

JOB_PATH = os.path.join(os.path.abspath("."), "res", "jobs")
BASE_MODEL_PATH = os.path.join(os.path.abspath("."), "res", "models")

class TianshuFlServer():

    def __init__(self):
        super(TianshuFlServer, self).__init__()


class TianshuFlStandaloneServer(TianshuFlServer):
    def __init__(self, federate_strategy):
        super(TianshuFlStandaloneServer, self).__init__()
        self.executor_pool = ThreadPoolExecutor(2)
        if federate_strategy == FedrateStrategy.FED_AVG:
            self.aggregator = FedAvgAggregator(WorkModeStrategy.WORKMODE_STANDALONE, JOB_PATH, BASE_MODEL_PATH)
        else:
           pass


    def start(self):
        #self.executor_pool.submit(self.aggregator.aggregate)
        self.aggregator.aggregate()




class TianshuFlClusterServer(TianshuFlServer):

    def __init__(self, federate_strategy, ip, port, api_version):
        super(TianshuFlClusterServer, self).__init__()
        self.executor_pool = ThreadPoolExecutor(5)
        if federate_strategy == FedrateStrategy.FED_AVG:
            self.aggregator = FedAvgAggregator(WorkModeStrategy.WORKMODE_CLUSTER, JOB_PATH, BASE_MODEL_PATH)
        else:
            pass
        self.ip = ip
        self.port = port
        self.api_version = api_version
        self.federate_strategy = federate_strategy

    def start(self):
        self.executor_pool.submit(communicate_server.start_communicate_server, self.api_version, self.ip, self.port)
        #self.executor_pool.submit(self.aggregator.aggregate)
        #communicate_server.start_communicate_server(self.api_version, self.ip, self.port)
        if self.federate_strategy == FedrateStrategy.FED_AVG:
            self.aggregator.aggregate()
        else:
            pass





