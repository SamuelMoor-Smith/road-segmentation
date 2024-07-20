import sys

import segmentation_models_pytorch as smp
import torch
from segmentation_models_pytorch.utils.meter import AverageValueMeter
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm as tqdm
from utils import show_val_samples


class Epoch:
    def __init__(self, model, loss, metrics, stage_name, device="cpu", verbose=True):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device

        self._to_device()

    def _to_device(self):
        self.model.to(self.device)
        self.loss.to(self.device)

    def _format_logs(self, logs):
        str_logs = ["{} - {:.4}".format(k, v) for k, v in logs.items()]
        s = ", ".join(str_logs)
        return s

    def batch_update(self, x, y):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, dataloader, epoch_num):

        self.on_epoch_start()

        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {metric: AverageValueMeter() for metric in self.metrics}
        with tqdm(
                dataloader,
                desc=self.stage_name,
                file=sys.stdout,
                disable=not self.verbose,
        ) as iterator:
            for it, (x, y, orig_x) in enumerate(iterator):
                x, y = x.to(self.device), y.to(self.device)
                loss, y_pred = self.batch_update(x, y)
                # update loss logs
                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                loss_logs = {'SoftBCE': loss_meter.mean}
                logs.update(loss_logs)
                tp, fp, fn, tn = smp.metrics.get_stats(output=y_pred.cpu().detach(),
                                                       target=y.long().cpu().detach(),
                                                       mode='binary',
                                                       threshold=0.5,
                                                       )
                # update metrics logs
                for metric in self.metrics:
                    metric_fnc = getattr(smp.metrics, metric)
                    metric_value = metric_fnc(tp, fp, fn, tn, reduction="micro")
                    metrics_meters[metric].add(metric_value)
                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                logs.update(metrics_logs)
                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

                if self.stage_name == 'valid' and epoch_num % 10 == 0 and it == 0:
                    orig_x = orig_x.permute(0, 3, 1, 2)
                    show_val_samples(orig_x.detach().cpu().numpy(), y.detach().cpu().numpy(), y_pred.detach().cpu().numpy())
        return logs


class TrainEpoch(Epoch):
    def __init__(self, model, loss, metrics, optimizer, device="cpu", verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name="train",
            device=device,
            verbose=verbose,
        )
        self.optimizer = optimizer
        self.scaler = GradScaler()

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x, y):
        self.optimizer.zero_grad()
        with autocast():
            prediction = self.model.forward(x)
            loss = self.loss(prediction, y)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss, prediction


class ValidEpoch(Epoch):
    def __init__(self, model, loss, metrics, device="cpu", verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name="valid",
            device=device,
            verbose=verbose,
        )

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x, y):
        with torch.no_grad():
            prediction = self.model.forward(x)
            loss = self.loss(prediction, y)
        return loss, prediction
