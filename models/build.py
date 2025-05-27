from models.deformable_detr.deformable_detr import build_deformable_detr_criterion


def build_loss_fn(config: dict):
    loss_fn_dict = {}
    tasks = config["TASKS"]
    for task in tasks:
        if task == "SoccerNetGSR_Detection":
            loss_fn_dict[task] = build_deformable_detr_criterion(config=config)
        else:
            raise ValueError(f"Task {task} is not supported.")


    return loss_fn_dict