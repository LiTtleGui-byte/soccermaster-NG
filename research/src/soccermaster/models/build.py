from soccermaster.models.deformable_detr.deformable_detr import build_deformable_detr_criterion, build_detection_metrics
from soccermaster.models.lines_detection import build_lines_detection_loss, build_lines_detection_metrics
from soccermaster.models.keypoints_detection import build_keypoints_detection_loss, build_keypoints_detection_metrics
from soccermaster.models.soccernet_gsr_reid import build_soccer_net_gsr_reid_loss
from soccermaster.models.video_caption import build_video_caption_loss, build_video_caption_metrics
from soccermaster.models.camera import build_camera_loss, build_camera_metrics
from soccermaster.models.caption_classification import build_caption_classification_loss, build_caption_classification_metrics
from soccermaster.models.caption_classification_align import build_caption_classification_loss_align, build_caption_classification_metrics_align

def build_loss_fn(config: dict):
    loss_fn_dict = {}
    # tasks = config["TASKS"]
    datasets_to_heads = config["DATASETS_TO_HEADS"]
    all_heads = []
    for dataset, heads in datasets_to_heads.items():
        all_heads.extend(heads)
    all_heads = list(set(all_heads))
    all_heads.sort()
    
    for head in all_heads:
        if head == "SoccerNetGSR_Detection":
            loss_fn_dict[head] = build_deformable_detr_criterion(config=config)
        elif head == "LinesDetection":
            loss_fn_dict[head] = build_lines_detection_loss(config=config)
        elif head == "KeypointsDetection":
            loss_fn_dict[head] = build_keypoints_detection_loss(config=config)
        elif head == "SoccerNetGSR_ReID":
            loss_fn_dict[head] = build_soccer_net_gsr_reid_loss(config=config)
        elif head == "VideoCaption":
            loss_fn_dict[head] = build_video_caption_loss(config=config)
        elif head == "CaptionClassification":
            loss_fn_dict[head] = build_caption_classification_loss(config=config)
        elif head == "CaptionClassificationAlign":
            loss_fn_dict[head] = build_caption_classification_loss_align(config=config)
        elif head == "CameraRegression":
            loss_fn_dict[head] = build_camera_loss(config=config)
        else:
            raise ValueError(f"Head {head} is not supported.")

    return loss_fn_dict

def build_metrics_fn(config: dict):
    """
    构建各个任务的metrics计算函数字典
    """
    metrics_fn_dict = {}
    # tasks = config["TASKS"]
    datasets_to_heads = config["DATASETS_TO_HEADS"]
    all_heads = []
    for dataset, heads in datasets_to_heads.items():
        all_heads.extend(heads)
    all_heads = list(set(all_heads))
    all_heads.sort()
    for head in all_heads:
        if head == "SoccerNetGSR_Detection":
            metrics_fn_dict[head] = build_detection_metrics(config=config)
        elif head == "LinesDetection":
            metrics_fn_dict[head] = build_lines_detection_metrics(config=config)
        elif head == "KeypointsDetection":
            metrics_fn_dict[head] = build_keypoints_detection_metrics(config=config)
        elif head == "SoccerNetGSR_ReID":
            # 这里可以添加ReID任务的metrics计算，暂时为None
            metrics_fn_dict[head] = None
        elif head == "VideoCaption":
            metrics_fn_dict[head] = build_video_caption_metrics(config=config)
        elif head == "CaptionClassification":
            metrics_fn_dict[head] = build_caption_classification_metrics(config=config)
        elif head == "CaptionClassificationAlign":
            metrics_fn_dict[head] = build_caption_classification_metrics_align(config=config)
        elif head == "CameraRegression":
            metrics_fn_dict[head] = build_camera_metrics(config=config)
        else:
            raise ValueError(f"Head {head} is not supported.")

    return metrics_fn_dict