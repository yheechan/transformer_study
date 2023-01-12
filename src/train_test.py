import argparse
import pprint

import torch, gc
import torch.nn as nn
from torch import optim

from data_loader import DataLoader
import data_loader
import trainer
import tester
from models.transformer import Transformer
import model_util as mu

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import timeit


def define_argparser():
	p = argparse.ArgumentParser()

	p.add_argument(
		'--research_subject',
		required=True,
		help='The name of the research subject. (ex: server1)',
	)

	p.add_argument(
		'--research_num',
		required=True,
		help='The number of current test for a subject experiment. (ex: 01)',
	)

	p.add_argument(
		'--train',
		required=True,
		help='Training set file name exept the extention. (ex: train.en --> train)',
	)

	p.add_argument(
        '--valid',
        required=True,
        help='Validation set file name except the extention. (ex: valid.en --> valid)',
    )

	p.add_argument(
        '--test',
        required=True,
        help='Test set file name except the extention. (ex: test.en --> test)',
    )

	p.add_argument(
        '--lang',
        required=True,
        help='Set of extention represents language pair. (ex: en + ko --> enko)',
    )

	p.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Mini batch size for gradient descent. Default=%(default)s',
    )

	p.add_argument(
        '--max_length',
        type=int,
        default=50,
        help='Maximum length of the training sequence. Default=%(default)s',
    )

	'''
	p.add_argument(
		'--use_transformer',
		action='store_true',
		help='Set model architecture as Transformer',
	)
	'''

	p.add_argument(
		'--hidden_size',
		type=int,
		default=256,
		help='Word embedding vector dimension. Default=%(default)s',
	)

	p.add_argument(
		'--n_splits',
		type=int,
		default=8,
		help='Number of heads in architecture as Transformer',
	)

	p.add_argument(
		'--n_layers',
		type=int,
		default=4,
		help='Number of layers in LSTM. Default=%(default)s',
	)

	p.add_argument(
		'--dropout',
		type=float,
		default=.2,
		help='Dropout rate. Default=%(default)s',
	)

	'''
	p.add_argument(
		'--use_adam',
		action='store_true',
		help='Use Adam as optimizer instead of SGD. Oter lr arguments should be changed.',
	)
	'''

	p.add_argument(
		'--lr',
		type=float,
		default=.0001,
		help='Initial learning rate. Default=%(default)s',
	)

	p.add_argument(
		'--n_epochs',
		type=int,
		default=20,
		help='Number of epochs to train. Default=%(default)s',
	)

	config = p.parse_args()

	return config


def get_model(input_size, output_size, config):
	model = Transformer(
		input_size,						# Source vocabulary size
		config.hidden_size,				# Transformer doesn't need word_vec_size,
		output_size,					# Target vocabulary size
		n_splits=config.n_splits,		# Number of head in Multi-head Attention
		n_enc_blocks=config.n_layers,	# number of encoder blocks
		n_dec_blocks=config.n_layers,	# Number of decoder blocks
		dropout_p=config.dropout,		# Dropout rate on each block
	)
	
	return model


def get_crit(output_size, pad_index):
	# Default weight for loss equals to 1, but we don't need to get loss for PAD token
	# Thus, set a weight for PAD to zero.
	loss_weight = torch.ones(output_size)
	loss_weight[pad_index] = 0.0

	# Instead of using Cross-Entropy Loss,
	# we can use Negative Log-Likelihood(NLL) Loss with log-probability.
	print('\n Loss function: Negative Log-Likelihood with log-probability (NLLLoss)')
	crit = nn.NLLLoss(
		weight=loss_weight,
		reduction='sum',
	)

	return crit


def get_optimizer(model, config):
	print('Optimizer: Adam')
	optimizer = optim.Adam(model.parameters(), lr=config.lr, betas=(.9, .98))
	
	return optimizer


def main(config, model_weight=None, opt_weight=None):

	# ********** PRINT CONFIG HELP **********
	def print_config(config):
		pp = pprint.PrettyPrinter(indent=4)
		pp.pprint(vars(config))
	print_config(config)


	# ********** LOAD DATA ACCORDING TO FILE NAME **********
	loader = DataLoader(
        train_fn=config.train,                           # Train file name except extention, which is language.
        valid_fn=config.valid,                           # Validation file name except extension.
        test_fn=config.test,                           	 # Test file name except extension.
        exts=(config.lang[:2], config.lang[-2:]),    	 # Source and target language.
        batch_size=config.batch_size,
        device=-1,                              # Lazy loading
        max_length=config.max_length,           # Loger sequence will be excluded.
        dsl=False,                              # Turn-off Dual-supervised Learning mode.
    )

	input_size, output_size = len(loader.src.vocab), len(loader.tgt.vocab)
	print('\ninput_size: ', input_size)
	print('output_size: ', output_size)


	# ********** GET MODEL | LOSS FUNCTION & MOVE TO DEVICE **********
	model = get_model(input_size, output_size, config)

	crit = get_crit(output_size, data_loader.PAD)

	'''
	if model_weight is not None:
		model.load_state_dict(model_weight)
	'''

	# check for available gpu
	if torch.cuda.is_available():
		device_num = 0
		print('\nUsing device number: 0')
	else:
		device_num = -1
		print('\nUsing device number: -1')

	# Clear memory cache
	gc.collect()
	torch.cuda.empty_cache()

	# Pass model to GPU device if it is necessary
	if device_num >= 0:
		model.cuda(device_num)
		crit.cuda(device_num)
	

	# ********** GET OPTIMIZER **********
	optimizer = get_optimizer(model, config)

	'''
	if opt_weight is not None and config.use_adam:
		optimizer.load_state_dict(opt_weight)
	'''
	
	lr_schedular = None


	# ********** INSTANTIATE TENSORBOARD **********
	subject_title = config.research_subject

	timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
	writer = SummaryWriter('../tensorboard/'+subject_title+'/tests')

	title = subject_title + '_' + config.research_num

	start_time = timeit.default_timer()


	# ********** TRAIN MODEL **********
	trainer.train(
		model,
		crit,
		optimizer,
		train_loader=loader.train_iter,
		valid_loader=loader.valid_iter,
		src_vocab=loader.src.vocab,
		tgt_vocab=loader.tgt.vocab,
		n_epochs=config.n_epochs,
		lr_schedular=lr_schedular,
		writer=writer,
		title=title,
	)

	end_time = (timeit.default_timer() - start_time) / 60.0

	print('total taken time: ', end_time)


	# ********** SAVE & BRING MODEL **********
	mu.saveModel(subject_title, title, model)
	# mu.graphModel(train_dataloader, model, writer, device)

	model = mu.getModel(subject_title, title)


	# ********** TEST MODEL **********
	test_loss, test_acc = tester.test(
		model,
		crit,
		test_loader=loader.test_iter,
		src_vocab=loader.src.vocab,
		tgt_vocab=loader.tgt.vocab,
		lr_schedular=lr_schedular,
	)

	print('test loss: ', test_loss)
	print('test acc: ', test_acc)



if __name__ == '__main__':
	config = define_argparser()
	main(config)
