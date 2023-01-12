import os

import torchtext
version = list(map(int, torchtext.__version__.split('.')))

if version[0] <= 0 and version[1] < 9:
	from torchtext import data
	print('using data from torchtext')
else:
	from torchtext.legacy import data
	print('using data from torchtext.legacy')

PAD, BOS, EOS = 1, 2, 3

class DataLoader():

	def __init__(
		self,
		train_fn=None,
		valid_fn=None,
		test_fn=None,
		exts=None,
		batch_size=64,
		device='cpu',
		max_vocab=99999999,
		max_length=50,
		fix_length=None,
		use_bos=True,
		use_eos=True,
		shuffle=False,
		dsl=False
	):
		super(DataLoader, self).__init__()

		# torchtext.legacy Field
		# Defines a datatype together with instructions for converting to Tensor
		# Parameters >
		# sequential - Whether the datatype represents sequential data.
		#			   If False, no tokenization is applied. Default: True.
		# use_vocab  - Whether to use a Vocab object.
		#			   If False, the data in this field should already be numerical. Default: True.
		# init_token - A token that will be prepended to every example using this field,
		#			   or None for no initial token. Default None.
		# eos_token  - A token that will be appended to every example using this field,
		#			   or None for no end-of-sentence token. Default: None.
		# fix_length - A fixed length that all examples using this field will be padded to,
		#			   or None for flexible sequence lengths. Default: None.
		# include_lengths - Whether to return a tuple of a padded minibatch and a list containing
		#					the lengths of each examples, or just a padded minibatch. Default: False.
		# batch_first 	  - Whether to preduce tensors with the batch dimension first. Default: False.
		self.src = data.Field(
            sequential=True,
            use_vocab=True,
            batch_first=True,
            include_lengths=True,
            fix_length=fix_length,
            init_token='<BOS>' if dsl else None,
            eos_token='<EOS>' if dsl else None,
        )

		self.tgt = data.Field(
			sequential=True,
			use_vocab=True,
			batch_first=True,
			include_lengths=True,
			fix_length=fix_length,
			init_token='<BOS>' if use_bos else None,
			eos_token='<EOS>' if use_eos else None,
		)


		if train_fn is not None and valid_fn is not None and test_fn is not None and exts is not None:
			train = TranslationDataset(
				path=train_fn,
				exts=exts,
				fields=[('src', self.src), ('tgt', self.tgt)],
				max_length=max_length
			)

			valid = TranslationDataset(
				path=valid_fn,
				exts=exts,
				fields=[('src', self.src), ('tgt', self.tgt)],
				max_length=max_length,
			)

			test = TranslationDataset(
				path=test_fn,
				exts=exts,
				fields=[('src', self.src), ('tgt', self.tgt)],
				max_length=max_length
			)

			# torchtext.legacy BucketIterator
			# Defines an iterator that batches examples of similar lengths together.

			self.train_iter = data.BucketIterator(
				dataset=train,
				batch_size=batch_size,
				device='cuda:%d' % device if device >= 0 else 'cpu',
				shuffle=shuffle,
				sort_key=lambda x: len(x.tgt) + (max_length * len(x.src)),
				sort_within_batch=True,
			)

			self.valid_iter = data.BucketIterator(
				dataset=valid,
				batch_size=batch_size,
				device='cuda:%d' % device if device >= 0 else 'cpu',
				shuffle=shuffle,
				sort_key=lambda x: len(x.tgt) + (max_length * len(x.src)),
				sort_within_batch=True,
			)

			self.test_iter = data.BucketIterator(
				dataset=test,
				batch_size=batch_size,
				device='cuda:%d' % device if device >= 0 else 'cpu',
				shuffle=shuffle,
				sort_key=lambda x: len(x.tgt) + (max_length * len(x.src)),
				sort_within_batch=True,
			)

			# torchtext.legacy build_vocab()
			# Construct the Vocab object for this field from one or more dataset.
			self.src.build_vocab(train, max_size=max_vocab)
			self.tgt.build_vocab(train, max_size=max_vocab)
	
	def load_vocab(self, src_vocab, tgt_vocab):
		self.src.vocab = src_vocab
		self.tgt.vocab = tgt_vocab


class TranslationDataset(data.Dataset):
	"""Defines a dataset for machine translation."""

	@staticmethod
	def sort_key(ex):
		return data.interleave_keys(len(ex.src), len(ex.trg))
	
	def __init__(self, path, exts, fields, max_length=None, **kwargs):
		"""Create a TranslationDataset given paths and fields.
			Arguments:
			path: Common prefix of paths to the data files for both languages.
			exts: A tuple containing the extension to path for each language.
			fields: A tuple containing the fields that will be used for data
					in each language.
			Remaining keyword arguments: Passed to the constructor of
			data.Dataset.
		"""

		if not isinstance(fields[0], (tuple, list)):
			fields = [('src', fields[0]), ('trg', fields[1])]

		if not path.endswith('.'):
			path += '.'

		src_path, trg_path = tuple(os.path.expanduser(path + x) for x in exts)

		src_path = './data/' + src_path
		trg_path = './data/' + trg_path

		examples = []

		with open(src_path, encoding='utf-8') as src_file, open(trg_path, encoding='utf-8') as trg_file:
			for src_line, trg_line in zip(src_file, trg_file):

				src_line, trg_line = src_line.strip(), trg_line.strip()

				if max_length and max_length < max(len(src_line.split()), len(trg_line.split())):
					continue
				if src_line != '' and trg_line != '':
					# torchtext.legacy Examples
					# Example Defines a single training or test example
					# Stores each column of the example as an attribute
					examples += [data.Example.fromlist([src_line, trg_line], fields)]

		super().__init__(examples, fields, **kwargs)


if __name__ == '__main__':
	import sys

	loader = DataLoader(
		sys.argv[1],
		sys.argv[2],
		(sys.argv[3], sys.argv[4]),
		batch_size=128
	)

	print(len(loader.src.vocab))
	print(len(loader.tgt.vocab))

	for batch_index, batch in enumerate(loader.train_iter):
		print(batch.src)
		print(batch.tgt)
	
		if batch_index > 1: break
