import torch
import numpy as np
from bisect import bisect

from fairseq.data import data_utils


class WordNoising(object):
    def __init__(self, dictionary):
        self.dictionary = dictionary

    def noising(self, x , noising_prob = 0.0):
        raise NotImplementedError()



class WordDropout(WordNoising):
    def __init__(self, dictionary):
        super().__init__(dictionary)
        self.default_prob = 0.1

    def noising(self, x, noising_prob = None):
        if noising_prob is None:
            noising_prob = self.default_prob
        if noising_prob == 0:
            return x
        new_dict = self.dictionary.symbols[4:]
        freq = self.dictionary.count[4:]
        freq = np.array(freq)/sum(freq)
        freq_cum = np.cumsum(freq)

        assert 0 < noising_prob < 1 
        new_s = x[0]
        has_eos = new_s[-1] == self.dictionary.eos()
        words = new_s.tolist()
        word_length = len(words)
        if has_eos:
            word_length -=1

        for i in range(word_length):
            draw = np.random.binomial(1,noising_prob)
            if draw:
                new_id = new_dict[self.sample_index(freq_cum)]
                new_id = self.dictionary.indices[new_id]
                words[i] = new_id

        return torch.LongTensor(words)

    def sample_index(self, ps):
        return bisect(ps,np.random.random()*ps[-1])


class NoisingData(torch.utils.data.Dataset):
    def __init__(self,src_dataset,src_dict,seed, noiser = WordDropout, dropout = 0.1):
        self.src_dataset = src_dataset
        self.src_dict = src_dict
        self.seed = seed
        self.noiser = noiser(dictionary = src_dict )
        self.dropout = dropout

    def __getitem__(self, index):
        src_tokens = self.src_dataset[index]
        src_tokens = src_tokens.unsqueeze(0)
        #print("before dropout")
        #print(src_tokens)

        with data_utils.numpy_seed(self.seed + index):
            noisy_src_tokens = self.noiser.noising(src_tokens, self.dropout)
        #print("after dropout")
        #print(noisy_src_tokens)

        return noisy_src_tokens


    def __len__(self):
        return len(self.src_dataset)


    @property
    def supports_prefetch(self):
        return self.src_dataset.supports_prefetch

    def prefetch(self, indices):
        if self.src_dataset.supports_prefetch:
            self.src_dataset.prefetch(indices)
