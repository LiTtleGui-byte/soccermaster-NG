from data.tracking import build_tracking_dataset

def build_dataset(config: dict):
    train_dataset_dict = {}
    test_dataset_dict = {}
    tasks = config["TASKS"]
    for task in tasks:
        if task == "Tracking":
            train_dataset_dict[task] = build_tracking_dataset(config=config)
            test_dataset_dict[task] = build_tracking_dataset(config=config)
        else:
            raise ValueError(f"Task {task} is not supported.")


    return train_dataset_dict, test_dataset_dict