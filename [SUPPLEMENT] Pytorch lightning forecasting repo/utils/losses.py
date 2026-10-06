import torch


def pinball_loss_expectile(pred, target, tau, dist_side):
    if dist_side == 'up':
        r = (target - pred)
        w = torch.where(r >= 0, tau,
                            1.0 - tau)
        loss = w * r.pow(2)
        loss = torch.mean(loss)
    elif dist_side == 'down':
        r = (pred - target)
        w = torch.where(r >= 0, tau,
                            1.0 - tau)
        loss = w * r.pow(2)
        loss = torch.mean(loss)
    elif dist_side == 'both':
        pred_down = pred[:, :, 0]
        pred_up = pred[:, :, 1]
        r_down = (pred_down - target)
        w_down = torch.where(r_down >= 0, tau,
                            1.0 - tau)
        loss_down = w_down * r_down.pow(2)
        r_up = (target - pred_up)
        w_up = torch.where(r_up >= 0, tau,
                            1.0 - tau)
        loss_up = w_up * r_up.pow(2)
        loss = torch.mean(loss_down + loss_up)
    return loss

def pinball_loss(pred, target, tau, dist_side):
    if dist_side == 'both':
        pred_down = pred[:, :, 0]
        pred_up = pred[:, :, 1]
        loss_up = torch.max((tau * (target - pred_up)), ((tau - 1) * (target - pred_up)))
        loss_down = torch.max((tau * (pred_down - target)), ((tau - 1) * (pred_down - target)))
        loss = torch.mean(loss_up + loss_down)
    elif dist_side == 'up':
        loss = torch.max((tau * (target - pred)), ((tau - 1) * (target - pred)))
        loss = torch.mean(loss)
    elif dist_side == 'down':
        loss = torch.max((tau * (pred - target)), ((tau - 1) * (pred - target)))
        loss = torch.mean(loss)
    return loss