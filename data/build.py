from data.tracking import build_tracking_dataset
from data.SoccerNetGSR_Detection import build_gsr_detection_dataloader

def build_dataloader(config: dict):
    dataloader_train_dict = {}
    dataloader_test_dict = {}
    tasks = config["TASKS"]
    for task in tasks:
        if task == "SoccerNetGSR_Detection":
            dataloader_train_dict[task] = build_gsr_detection_dataloader(config=config, split="train")
            dataloader_test_dict[task] = build_gsr_detection_dataloader(config=config, split="test")
        else:
            raise ValueError(f"Task {task} is not supported.")


    return dataloader_train_dict, dataloader_test_dict