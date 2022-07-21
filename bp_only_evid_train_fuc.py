# -*- coding: utf-8 -*-
"""

"""

import torch

def train_loop(trainLoader, model, device,Lr, metric_fn, loss_fn, is_loss_MAE = 0, is_pretrained = 0):
    if is_pretrained == 0:
        optimizer = torch.optim.Adam(model.parameters(), lr=Lr)
    else:
        for weights in model.encoder.parameters():
            weights.requires_grad = False
        params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.Adam(params, lr=1e-3)
    model.train()
    size = len(trainLoader.dataset)
    sp_mae = 0
    dp_mae = 0
    cnt = 0
    for batch, (data_batch, labels) in enumerate(trainLoader):
        cnt += 1
        data_batch = data_batch.to(device, dtype=torch.float)
        bp_labels = labels.to(device)
        sp, dp = model(data_batch)
        sp_v = sp[:,0]
        sp_mean = sp[:,1]
        sp_alpha = sp[:,2]
        sp_beta = sp[:,3]
        dp_v = dp[:,0]
        dp_mean = dp[:,1]
        dp_alpha = dp[:,2]
        dp_beta = dp[:,3]
        sp_metric = metric_fn(sp_mean, bp_labels[:,0])
        dp_metric = metric_fn(dp_mean, bp_labels[:,1])
        if is_loss_MAE == 1:
            sp_loss = loss_fn(bp_labels[:,0], sp_mean)
            dp_loss = loss_fn(bp_labels[:,1], dp_mean)
            loss = 1 * sp_loss + 1 * dp_loss
        else:
            # loss = loss_fn(bp_labels[:,0], sp_v, sp_mean, sp_alpha, sp_beta, 1.5) + \
            #        loss_fn(bp_labels[:,1], dp_v, dp_mean, dp_alpha, dp_beta, 1.5)
            loss = loss_fn(bp_labels[:,0], sp_v, sp_mean, sp_alpha, sp_beta, 1.5) + \
                   loss_fn(bp_labels[:,1], dp_v, dp_mean, dp_alpha, dp_beta, 1.5)
        assert torch.isnan(sp_mean).sum() == 0, print(loss)
        # print(loss_fn(bp_labels[:,1], dp_v, dp_mean, dp_alpha, dp_beta, 0))
        sp_metric = sp_metric.item()
        dp_metric = dp_metric.item()
        optimizer.zero_grad() # if don't call zero_grad, the grad of each batch will be accumulated
        loss.backward()
        optimizer.step()
        sp_mae += sp_metric
        dp_mae += dp_metric
        # sp_mae = sp_metric
        # dp_mae = dp_metric
        if batch % 100 == 0:
              loss, current = loss.item(), batch * len(data_batch)
              print(f"loss: {loss:>7f}------ sp_metric: {sp_metric:>7f}------ dp_metric: {dp_metric:>7f}  [{current:>5d}/{size:>5d}]")
    sp_mae /= cnt
    dp_mae /= cnt
    print( f"SP MAE: {sp_mae:>8f}------ DP MAE: {dp_mae:>7f}")
    

def test_loop(dataloader, model, epoch, device, deep_evid_learning = 0):
    num_batches = len(dataloader)
    num_batches = 0
    model.eval()
    sp_metric = 0
    dp_metric = 0
    SP_AE = []
    DP_AE = []
    with torch.no_grad():
        for X, y in dataloader:
            num_batches += 1
            X, y = X.to(device), y.to(device)
            sp, dp = model(X)
            for i in range(len(sp)):
                SP_AE.append(abs(sp[i, 1] - y[i,0]))
                DP_AE.append(abs(dp[i, 1] - y[i,1]))
        SP_AE = torch.tensor(SP_AE)
        sp_metric = torch.mean(SP_AE)
        DP_AE = torch.tensor(DP_AE)
        dp_metric = torch.mean(DP_AE)

    # sp_metric = SP_AE[-1]
    # dp_metric = DP_AE[-1]

    print( f"Test SP MAE: {sp_metric:>8f}------ Test DP MAE: {dp_metric:>7f}")
    checkpoint = {
          "model": model.state_dict(),
          "epoch": epoch
      }
    if deep_evid_learning:
        torch.save(checkpoint, './evid_train_model/ckpt_best_%s_sp_%.2f_dp_%.2f.pth' %(str(epoch + 1), 
        sp_metric, dp_metric))
    else:
        torch.save(checkpoint, './train_model/ckpt_best_%s_sp_%.2f_dp_%.2f.pth' %(str(epoch + 1), 
        sp_metric, dp_metric))