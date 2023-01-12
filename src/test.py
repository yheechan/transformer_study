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
		required=False,
		help='Training set file name exept the extention. (ex: train.en --> train)',
	)

	p.add_argument(
        '--valid',
        required=False,
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

	'''
	p.add_argument(
		'--use_adam',
		action='store_true',
		help='Use Adam as optimizer instead of SGD. Oter lr arguments should be changed.',
	)
	'''

	config = p.parse_args()

	return config


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



def main(config, model_weight=None, opt_weight=None):

	# ********** PRINT CONFIG HELP **********
	def print_config(config):
		pp = pprint.PrettyPrinter(indent=4)
		pp.pprint(vars(config))
	print_config(config)


	# ********** BRING MODEL **********
	subject_title = config.research_subject
	title = subject_title + '_' + config.research_num

	model = mu.getModel(subject_title, title)

	output_size = model.output_size


	# ********** LOAD DATA ACCORDING TO FILE NAME **********
	loader = DataLoader(
        test_fn=config.test,                           	 # Test file name except extension.
        exts=(config.lang[:2], config.lang[-2:]),    	 # Source and target language.
        batch_size=config.batch_size,
        device=-1,                              # Lazy loading
        max_length=config.max_length,           # Loger sequence will be excluded.
        dsl=False,                              # Turn-off Dual-supervised Learning mode.
    )


	# ********** GET LOSS FUNCTION **********
	crit = get_crit(output_size, data_loader.PAD)


	# ********** TEST MODEL **********
	test_loss, test_acc = tester.test(
		model,
		crit,
		test_loader=loader.test_iter,
		src_vocab=loader.src.vocab,
		tgt_vocab=loader.tgt.vocab,
	)


	print('test loss: ', test_loss)
	print('test acc: ', test_acc)


if __name__ == '__main__':
	config = define_argparser()
	main(config)
