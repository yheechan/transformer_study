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
	writer=None,
	title=None,
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
			
			# accuracy for each predicted token
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

			# print(batch_idx, loss, np.mean(tot_train_acc))

		# Calculate the average loss over the entire training data
		train_loss = np.mean(tot_train_loss)
		train_acc = np.mean(tot_train_acc)


		# ========= Training =========

		# After the completion of each training epoch,
		# measure the model's performance on validation set.
		val_loss, val_acc = evalute(
			model,
			crit,
			valid_loader,
			lr_schedular=None,
		)

		if val_acc > best_accuracy:
			best_accuracy = val_acc

		
		# print performance over the entire training data
		time_elapsed = time.time() - t0_epoch
		print(f"{epoch_i + 1:^7} | {train_loss:^12.6f} | {train_acc:^10.6f} | {val_loss:^8.6f} | {val_acc:^6.2f} | {time_elapsed:^6.2f}")

		writer.add_scalars(title + '-Loss',
			{'Train' : train_loss, 'Validation' : val_loss},
			epoch_i + 1)

		writer.add_scalars(title + '-Accuracy',
			{'Train' : train_acc, 'Validation' : val_acc},
			epoch_i + 1)

	writer.flush()

	print('\n')
	print(f"Training complete! Best accuracy: {best_accuracy:.2f}%.")


def evaluate(
	model,
	crit,
	valid_loader,
	lr_schedular=None,
):

	# After the completion of each training epoch,
	# measure the model's performance on validation set.

	
	# Put the model to evaluating mode
	model.eval()

	# Tracking variables
	tot_val_loss = []
	tot_val_acc = []

	
	# For each batch in our validation set...
	for batch_idx, batch in enumerate(valid_loader):

		# load data batch
		device = next(model.parameters()).device
		src = (batch.src[0].to(device), batch.src[1])
		tgt = (batch.tgt[0].to(device), batch.tgt[1])

		with torch.no_grad():
			x, y = src, tgt[0]
			# |x| = (batch_size, length)
			# |y| = (batch_size, length)


			y_hat = model(x, y)
			# |y_hat| = (batch_size, length, output_size)
			

			# accuracy for each predicted token
			for i in range(y_hat.shape[1]):
				pred = y_hat[:, i, :].argmax(1).flatten()
				acc = (pred == y[:, i]).cpu().numpy().mean() * 100
				tot_val_acc.append(acc)


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
			tot_val_loss.append(loss)

			print(batch_idx, loss, np.mean(tot_train_acc))

		# Calculate the average loss over the entire validation data
		val_loss = np.mean(tot_train_loss)
		val_acc = np.mean(tot_train_acc)

		return val_loss, val_acc
