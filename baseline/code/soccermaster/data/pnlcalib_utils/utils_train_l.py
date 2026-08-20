import torch
import torchvision.transforms as T

from tqdm import tqdm
from time import sleep

from model.metrics import calculate_metrics_l, calculate_metrics_l_with_mask
from data.pnlcalib_utils.utils_heatmap import get_keypoints_from_heatmap_batch_maxpool_l

def train_one_epoch(epoch_index, training_loader, optimizer, loss_fn, model, device, dataset="SoccerNet"):
    model.train(True)
    running_loss = 0.
    samples = 0

    with (tqdm(enumerate(training_loader), unit="batch", total=len(training_loader)) as tepoch):
        for i, data in tepoch:
            tepoch.set_description(f"Epoch {epoch_index}")

            if dataset != "SoccerNet":
                input, target, mask = data[0].to(device), data[1].to(device), data[2].to(device)

                transform = T.Resize((540, 960))
                input = transform(input)

                optimizer.zero_grad()
                outputs = model(input)
                loss = loss_fn(outputs, target, mask.unsqueeze(-1).unsqueeze(-1))

            else:
                input, target = data[0].to(device), data[1].to(device)

                optimizer.zero_grad()
                outputs = model(input)
                loss = loss_fn(outputs, target)


            loss.backward()
            optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            samples += input.size()[0]

            tepoch.set_postfix(loss=running_loss / samples)
            sleep(0.1)


    avg_loss = running_loss / samples

    return avg_loss


def validation_step(validation_loader, loss_fn, model, device, dataset="SoccerNet"):
    running_vloss = 0.0

    acc, prec, rec, f1 = 0, 0, 0, 0
    samples = 0
    model.eval()

    with (torch.no_grad()):
        for i, vdata in tqdm(enumerate(validation_loader), total=len(validation_loader)):

            if dataset != "SoccerNet":
                input, target, mask = vdata[0].to(device), vdata[1].to(device), vdata[2].to(device)

                transform = T.Resize((540, 960))
                input = transform(input)

                voutputs = model(input)
                vloss = loss_fn(voutputs, target, mask.unsqueeze(-1).unsqueeze(-1))

                kp_gt = get_keypoints_from_heatmap_batch_maxpool_l(target[:,:-1,:,:], return_scores=True, max_keypoints=2)
                kp_pred = get_keypoints_from_heatmap_batch_maxpool_l(voutputs[:,:-1,:,:], return_scores=True, max_keypoints=2)
                metrics = calculate_metrics_l_with_mask(kp_gt, kp_pred, mask)

            else:
                input, target = vdata[0].to(device), vdata[1].to(device)
                voutputs = model(input)
                vloss = loss_fn(voutputs, target)

                kp_gt = get_keypoints_from_heatmap_batch_maxpool_l(target[:,:-1,:,:], return_scores=True, max_keypoints=2)
                kp_pred = get_keypoints_from_heatmap_batch_maxpool_l(voutputs[:,:-1,:,:], return_scores=True, max_keypoints=2)
                metrics = calculate_metrics_l(kp_gt, kp_pred)

            running_vloss += vloss
            samples += input.size()[0]

            acc += metrics[0]
            prec += metrics[1]
            rec += metrics[2]
            f1 += metrics[3]

    avg_vloss = running_vloss / samples
    avg_acc = acc / (i+1)
    avg_prec = prec / (i+1)
    avg_rec = rec / (i+1)
    avg_f1 = f1 / (i+1)

    return avg_vloss, avg_acc, avg_prec, avg_rec, avg_f1
