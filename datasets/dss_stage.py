class DSSStageController(object):
    STAGE_RATIOS = {
        "early": {"gds_ratio": 0.5, "fss_ratio": 0.0, "rs_ratio": 0.5},
        "middle": {"gds_ratio": 0.5, "fss_ratio": 0.25, "rs_ratio": 0.25},
        "late": {"gds_ratio": 0.25, "fss_ratio": 0.5, "rs_ratio": 0.25},
    }

    def __init__(self, args):
        self.stage = "early"
        self.ce_ema = None
        self.total_loss_ema = None
        self.prev_total_loss_ema = None
        self.plateau_count = 0
        self.ce_threshold = float(getattr(args, "dss_ce_threshold", 2.0))
        self.plateau_delta = float(getattr(args, "dss_plateau_delta", 0.05))
        self.plateau_patience = int(getattr(args, "dss_plateau_patience", 3))
        self.ema_momentum = float(getattr(args, "dss_ema_momentum", 0.9))

    def current_ratios(self):
        return dict(self.STAGE_RATIOS[self.stage])

    def _update_ema(self, current, value):
        if current is None:
            return float(value)
        return self.ema_momentum * current + (1.0 - self.ema_momentum) * float(value)

    def update(self, epoch, epoch_ce_loss, epoch_total_loss):
        old_stage = self.stage
        loss_drop = None
        self.ce_ema = self._update_ema(self.ce_ema, epoch_ce_loss)
        self.total_loss_ema = self._update_ema(self.total_loss_ema, epoch_total_loss)

        if self.stage == "early":
            if self.ce_ema < self.ce_threshold:
                self.stage = "middle"
                self.plateau_count = 0
                self.prev_total_loss_ema = self.total_loss_ema
        elif self.stage == "middle":
            if self.prev_total_loss_ema is None:
                self.prev_total_loss_ema = self.total_loss_ema
            loss_drop = (self.prev_total_loss_ema - self.total_loss_ema) / max(self.prev_total_loss_ema, 1e-8)
            if loss_drop < self.plateau_delta:
                self.plateau_count += 1
            else:
                self.plateau_count = 0
            self.prev_total_loss_ema = self.total_loss_ema
            if self.plateau_count >= self.plateau_patience:
                self.stage = "late"

        result = {
            "epoch": epoch,
            "stage": self.stage,
            "old_stage": old_stage,
            "changed": old_stage != self.stage,
            "ce_ema": self.ce_ema,
            "total_loss_ema": self.total_loss_ema,
            "loss_drop": loss_drop,
            "plateau_count": self.plateau_count,
        }
        result.update(self.current_ratios())
        return result
