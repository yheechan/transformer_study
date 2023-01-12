import torch
import numpy as np

def test(
    model,
    crit,
    test_loader=None,
    src_vocab=None,
    tgt_vocab=None,
    lr_schedular=None,
):

    # check for available gpu
    if torch.cuda.is_available():
        device_num = 0
        print('\nUsing device number: 0')
    else:
        device_num = -1
        print('\nUsing device number: -1')
    
    # Pass model to GPU device if it is necessary
    if device_num >= 0:
        model.cuda(device_num)
        crit.cuda(device_num)

    model.eval()


    tot_test_loss = []
    tot_test_acc = []


    for batch_idx, batch in enumerate(test_loader):

        device = next(model.parameters()).device
        src = (batch.src[0].to(device), batch.src[1])
        tgt = (batch.tgt[0].to(device), batch.tgt[1])

        with torch.no_grad():

            x, y = src, tgt[0]

            y_hat = model(x, y)

            for i in range(y_hat.shape[1]):
                pred = y_hat[:, i, :].argmax(1).flatten()
                acc = (pred == y[:, i]).cpu().numpy().mean() * 100
                tot_test_acc.append(acc)
            
            loss = crit(
                y_hat.contiguous().view(-1, y_hat.size(-1)),
                y.contiguous().view(-1),
            )

            # backward_target = loss.div(y.size(0)).div(engine.config.iteration_per_update)
            backward_target = loss.div(y.size(0)).div(32)
            
            word_cnt = int(tgt[1].sum())
            loss = float(loss / word_cnt)
            tot_test_loss.append(loss)
        
        test_loss = np.mean(tot_test_loss)
        test_acc = np.mean(tot_test_acc)

        return test_loss, test_acc