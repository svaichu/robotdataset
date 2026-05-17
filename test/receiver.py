import time

import os
import torch

from data_config import NeuraDataConfig
from robotdataset import RLDataDistributed
# Import extensions to set up environment tasks

class Dummy:
    def get_action(self, observation):
        # Dummy action for testing purposes
        print("Dummy action taken for observation:", observation)
        # return a tensor of zeros as a dummy action
        return torch.zeros((1, 19), device="cuda" if torch.cuda.is_available() else "cpu")
    
class Inference():

    def __init__(self):
        self.policy = Dummy()
        data_config = NeuraDataConfig()
        self.manager = RLDataDistributed(name="agent", data_config=data_config, buffer_size=1000)
        # dist.init_process_group("gloo", rank=0, world_size=2, init_method='tcp://10.5.0.2:8000')
        # self.manager.loadPolicy(self.policy)
        task_strings = ["pick up and lift the cylinder on the table"]
        print("Finished loading policy, starting inference with task")
        data = torch.zeros((1, 20))
        dummy_observation = {
            "state": {
                "policy": None
            },
            "video": {
                "camera": torch.zeros((3, 224, 224)),
            }
        }
        dummy = None
        # out = dist.recv(tensor=data, src=1)
        dump = [None]
        # out = dist.recv_object_list(dump, src=1)
        # dummy = self.manager.recvObject(src=1)
        self.manager.recvBufferFromSim()
        print("Received data")
        self.manager.recvBufferFromSim()
        self.manager.printReplayBufferStructure()
        # dummy = dump[0]
        dummy = self.manager.replay_buffer 
        # print("Received data:", dummy["state"]["policy"].shape)
        print(dummy["state"]["policy"])
        # print("Received data:", dummy["video"]["camera"].shape)
        # print(dummy["video"]["camera"])
        # print(ou;t)


if __name__ == "__main__":
    service = Inference()