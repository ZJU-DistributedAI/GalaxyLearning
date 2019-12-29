import torch
import json
import torch.nn.functional as F
import time, os, pickle, requests, importlib, shutil, copy
from flask import url_for
from concurrent.futures import ThreadPoolExecutor
from gl.entity import runtime_config
from gl.core.strategy import RunTimeStrategy
from gl.core import communicate_client
from gl.utils.utils import JobDecoder, JobUtils
from gl.entity.job import Job
from gl.core.strategy import WorkModeStrategy, FedrateStrategy
from gl.core.trainer import TrainStandloneNormalStrategy, TrainMPCNormalStrategy, TrainStandloneDistillationStrategy, TrainMPCDistillationStrategy


JOB_PATH = os.path.join(os.path.abspath("."), "res", "jobs")

class TrainerController(object):
    def __init__(self, work_mode, data, client_id, client_ip, client_port, server_url, concurrent_num=5):
        self.work_mode = work_mode
        self.data = data
        self.client_id = client_id
        self.concurrent_num = concurrent_num
        self.trainer_executor_pool = ThreadPoolExecutor(self.concurrent_num)
        self.job_path = JOB_PATH
        self.fed_step = {}
        self.job_iter_dict = {}
        self.is_finish = True
        self.client_ip = client_ip
        self.client_port = client_port
        self.server_url = server_url



    def start(self):
        if self.work_mode == WorkModeStrategy.WORKMODE_STANDALONE:
            self.trainer_executor_pool.submit(self._trainer_standalone_exec)
            #self._trainer_standalone_exec()
        else:
            response = requests.post("/".join([self.server_url, "register", self.client_ip, '%s' % self.client_port, '%s' % self.client_id]))
            response_json = response.json()
            if response_json['code'] == 200 or response_json['code'] == 201:
                #self.trainer_executor_pool.submit(communicate_client.start_communicate_client, self.client_ip, self.client_port)
                #self.trainer_executor_pool.submit(self._trainer_mpc_exec, self.server_url)
                self.trainer_executor_pool.submit(communicate_client.start_communicate_client, self.client_ip, self.client_port)
                self._trainer_mpc_exec(self.server_url)
            else:
                print("connect to parameter server fail, please check your internet")






    def _trainer_standalone_exec(self):
        job_train_strategy = {}
        while True:
            job_list = JobUtils.list_all_jobs(self.job_path, self.job_iter_dict)
            for job in job_list:
                if job_train_strategy.get(job.get_job_id()) is None:
                    print(job.get_aggregate_strategy())
                    if job.get_aggregate_strategy() == FedrateStrategy.FED_AVG:
                        job_train_strategy[job.get_job_id()] = TrainStandloneNormalStrategy(job, self.data, self.fed_step, self.client_id)
                    else:
                        job_train_strategy[job.get_job_id()] = TrainStandloneDistillationStrategy(job, self.data, self.fed_step, self.client_id)
                self.run(job_train_strategy.get(job.get_job_id()))
            time.sleep(5)




    def _trainer_mpc_exec(self, server_url):
        job_train_strategy = {}
        while True:
            job_list = JobUtils.list_all_jobs(self.job_path, self.job_iter_dict)
            for job in job_list:
                if job_train_strategy.get(job.get_job_id()) is None:
                    if job.get_aggregate_strategy() == FedrateStrategy.FED_AVG:
                        job_train_strategy[job.get_job_id()] = job_train_strategy[job.get_job_id()] = TrainMPCNormalStrategy(job, self.data, self.fed_step, self.client_ip, self.client_port, server_url, self.client_id)
                    else:
                        job_train_strategy[job.get_job_id()] = TrainMPCDistillationStrategy(job, self.data, self.fed_step, self.client_ip, self.client_port, server_url, self.client_id)
                self.run(job_train_strategy.get(job.get_job_id()))
            time.sleep(5)



    def run(self, trainer):
        trainer.train()


