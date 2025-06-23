from models.deformable_detr.deformable_detr import build_deformable_detr_criterion, build_detection_metrics
from models.SoccerNetGSR_ReID import build_soccer_net_gsr_reid_loss
from models.VideoCaption import build_video_caption_loss

def build_loss_fn(config: dict):
    loss_fn_dict = {}
    tasks = config["TASKS"]
    for task in tasks:
        if task == "SoccerNetGSR_Detection":
            loss_fn_dict[task] = build_deformable_detr_criterion(config=config)
        elif task == "SoccerNetGSR_ReID":
            loss_fn_dict[task] = build_soccer_net_gsr_reid_loss(config=config)
        elif task == "VideoCaption":
            loss_fn_dict[task] = build_video_caption_loss(config=config)
        else:
            raise ValueError(f"Task {task} is not supported.")

    return loss_fn_dict

def build_metrics_fn(config: dict):
    """
    构建各个任务的metrics计算函数字典
    """
    metrics_fn_dict = {}
    tasks = config["TASKS"]
    for task in tasks:
        if task == "SoccerNetGSR_Detection":
            metrics_fn_dict[task] = build_detection_metrics(config=config)
        elif task == "SoccerNetGSR_ReID":
            # 这里可以添加ReID任务的metrics计算，暂时为None
            metrics_fn_dict[task] = None
        elif task == "VideoCaption":
            metrics_fn_dict[task] = None
        else:
            raise ValueError(f"Task {task} is not supported.")

    return metrics_fn_dict