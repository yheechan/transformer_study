import time
import torch
import numpy as np

def train(
	model,
	crit,
	optimizer,
	train_loader,
	valid_loader,
	src_vocab,
	tgt_vocab,
	n_epochs,
	lr_schedular=None,
):

	best_accuracy = 0

	print('Start training...')
	print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Train Acc':^10} | {'Val Loss':^8} | {'Val Acc':^6} | {'Elapsed':^6}")
	print("-"*80)

	for epoch_i in range(n_epochs):
	
		# ========= Training =========

		#Tracking time
		t0_epoch = time.time()

		#Put the model into training mode
		model.train()

		tot_train_acc = []
		tot_train_loss = []

		for batch_idx, batch in enumerate(train_loader):

			# load data batch
			device = next(model.parameters()).device
			src = (batch.src[0].to(device), batch.src[1])
			tgt = (batch.tgt[0].to(device), batch.tgt[1])

			# Zero out any previously calculated gradients
			optimizer.zero_grad()

			x, y = src, tgt[0]
			# |x| = (batch_size, length)
			# |y| = (batch_size, length)


			y_hat = model(x, y)
			# |y_hat| = (batch_size, length, output_size)
			
			for i in range(y_hat.shape[1]):
				pred = y_hat[:, i, :].argmax(1).flatten()
				acc = (pred == y[:, i]).cpu().numpy().mean() * 100
				tot_train_acc.append(acc)


			loss = crit(
				y_hat.contiguous().view(-1, y_hat.size(-1)),
				y.contiguous().view(-1),
			)

			# backward_target = loss.div(y.size(0)).div(engine.config.iteration_per_update)
			backward_target = loss.div(y.size(0)).div(32)

			# backward_target.backward(retain_graph=True)
			backward_target.backward()
			optimizer.step()

			word_cnt = int(tgt[1].sum())
			loss = float(loss / word_cnt)
			tot_train_loss.append(loss)

			print(batch_idx, loss, np.mean(tot_train_acc))

		# Calculate the average loss over the entire training data
		avg_train_loss = np.mean(tot_train_loss)
		avg_train_acc = np.mean(tot_train_acc)
		break
