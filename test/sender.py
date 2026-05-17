from datetime import datetime
from time import sleep

from data_config import NeuraDataConfig
from robotdataset import RLDataDistributed

import sys
import torch

def pt_file_contents(file_path):
    try:
        data = torch.load(file_path, map_location='cpu')
        return data
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")

class Simulation():

    def __init__(self):
        data_config = NeuraDataConfig()
        self.manager = RLDataDistributed(name = "simulator", data_config=data_config, buffer_size=1000)
        print("buffer initialized")
        print("[INFO] Initialized OnlineBuffer for data management.")
        # dummy_tensor = torch.ones((1, 20))
        # dummy_tensor = dummy_tensor * 4
        # dummy_observation = {
        #     "state": {
        #         "policy": torch.ones((1, 19))
        #     },
        #     "video": {
        #         "camera": torch.ones((3, 224, 224)),
        #     }
        # }
        # dummy = {"observations": dummy_observation}
        # dist.send(tensor=dummy_tensor, dst=0)
        # dist.send_object_list([dummy], dst=0)
        # self.manager.sendObject(dummy, dst=0)
        data_dict = pt_file_contents("/workspace/robotdataset/test/obs_dict_5.pt")
        # print(data_dict)
        for _ in range(15):
            processed_obs = self.manager.process_obs_from_isaac(data_dict)
            self.manager.add_to_replay_buffer(processed_obs)
        self.manager.printReplayBufferStructure()
        print("[INFO] offline data added to replay buffer and ready to send.")
        self.manager.sendBufferToAgent()
        print("[INFO] all sent out")
        self.manager.sendBufferToAgent()
        print("[INFO] all sent out again")
        # self.mana
        

if __name__ == "__main__":
    simulation = Simulation()