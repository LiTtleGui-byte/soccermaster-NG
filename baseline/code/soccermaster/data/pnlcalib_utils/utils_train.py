import torch
import torchvision.transforms as T

from tqdm import tqdm
from time import sleep

from data.pnlcalib_utils.utils_heatmap import get_keypoints_from_heatmap_batch_maxpool
from model.metrics import calculate_metrics


def train_one_epoch(epoch_index, training_loader, optimizer, loss_fn, model, device):
    model.train(True)
    running_loss = 0.
    samples = 0
    transform = T.Resize((540, 960))

    with (tqdm(enumerate(training_loader), unit="batch", total=len(training_loader)) as tepoch):
        for i, data in tepoch:
            tepoch.set_description(f"Epoch {epoch_index}")
            input, target, mask = data[0].to(device), data[1].to(device), data[2].to(device)
            input = input if input.size()[-1] == 960 else transform(input)

            optimizer.zero_grad()
            outputs = model(input)
            loss = loss_fn(outputs, target) * mask.unsqueeze(-1).unsqueeze(-1)
            loss = loss.mean()
            loss.backward()
            optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            samples += mask.size()[0]

            tepoch.set_postfix(loss=running_loss / samples)
            sleep(0.1)


    avg_loss = running_loss / samples

    return avg_loss


def validation_step(validation_loader, loss_fn, model, device):
    running_vloss = 0.0

    acc, prec, rec, f1 = 0, 0, 0, 0
    samples = 0
    transform = T.Resize((540, 960))
    model.eval()

    with torch.no_grad():
        for i, vdata in tqdm(enumerate(validation_loader), total=len(validation_loader)):
            input, target, mask = vdata[0].to(device), vdata[1].to(device), vdata[2].to(device)
            input = input if input.size()[-1] == 960 else transform(input)

            voutputs = model(input)
            vloss = loss_fn(voutputs, target) *  mask.unsqueeze(-1).unsqueeze(-1)
            vloss = vloss.mean()

            kp_gt = get_keypoints_from_heatmap_batch_maxpool(target[:,:-1,:,:], return_scores=True, max_keypoints=1)
            kp_pred = get_keypoints_from_heatmap_batch_maxpool(voutputs[:,:-1,:,:], return_scores=True, max_keypoints=1)
            metrics = calculate_metrics(kp_gt, kp_pred, mask)

            running_vloss += vloss
            samples += mask.size()[0]

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
