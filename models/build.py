from models.deformable_detr.deformable_detr import build_deformable_detr_criterion
from models.SoccerNetGSR_ReID import build_soccer_net_gsr_reid_loss

def build_loss_fn(config: dict):
    loss_fn_dict = {}
    tasks = config["TASKS"]
    for task in tasks:
        if task == "SoccerNetGSR_Detection":
            loss_fn_dict[task] = build_deformable_detr_criterion(config=config)
        elif task == "SoccerNetGSR_ReID":
            loss_fn_dict[task] = build_soccer_net_gsr_reid_loss(config=config)
        else:
            raise ValueError(f"Task {task} is not supported.")


    return loss_fn_dict