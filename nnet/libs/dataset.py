import random
import torch as th
import numpy as np

from torch.utils.data.dataloader import default_collate
import torch.utils.data as dat
from torch.nn.utils.rnn import pad_sequence
from .audio import WaveReader

def make_dataloader(train=True,
                    data_kwargs=None,
                    num_workers=4,
                    chunk_size=32000,
                    batch_size=16):
    dataset = Dataset(**data_kwargs)
    return DataLoader(dataset,
                      train=train,
                      chunk_size=chunk_size,
                      batch_size=batch_size,
                      num_workers=num_workers)


class Dataset(object):
    """
    Per Utterance Loader
    mix and ref shoulde be multi-channel sinal,
    while aux shoulde be single-channel. 
    """
    def __init__(self, mix_scp="", ref_scp=None, aux_scp=None, sample_rate=8000):
        self.mix = WaveReader(mix_scp, sample_rate=sample_rate)
        self.ref = WaveReader(ref_scp, sample_rate=sample_rate)
        self.aux = WaveReader(aux_scp, sample_rate=sample_rate)
        self.sample_rate = sample_rate
        self.spk_list = self._load_spk(r"/Share/hjy/project/td-speakerbeam/data/wsj0_2mix_extr_tr.spk")

    def _load_spk(self, spk_list_path):
        if spk_list_path is None:
            return []
        lines = open(spk_list_path).readlines()
        new_lines = []
        for line in lines:
            new_lines.append(line.strip())

        return new_lines 

    def __len__(self):
        return len(self.mix)

    def _select_channel(self, data, channel_num=1):
        """
        data: C x N
        select channel for input data, return the selected data
        """
        if data.ndim < 2:
            raise RuntimeError("The input shoulde be multi-channel array!")

        return data[:channel_num, :]

    def __getitem__(self, index):
        key = self.mix.index_keys[index]
        mix1 = np.squeeze(self._select_channel(self.mix[key], 2)[0])
        mix2 = np.squeeze(self._select_channel(self.mix[key], 2)[1])
        ref = np.squeeze(self._select_channel(self.ref[key], 1))
        
        aux = self.aux[key]
        spk_idx = self.spk_list.index(key.split('_')[-1][0:3]) 
        
        return {
            "mix1": mix1.astype(np.float32),
            "mix2": mix2.astype(np.float32),
            "ref": ref.astype(np.float32),
            "aux": aux.astype(np.float32),
            "aux_len": len(aux),
            "spk_idx": spk_idx,
            "key": key
        }


class ChunkSplitter(object):
    """
    Split utterance into small chunks
    """
    def __init__(self, chunk_size, train=True, least=16000):
        self.chunk_size = chunk_size
        self.least = least
        self.train = train

    def _make_chunk(self, eg, s):
        """
        Make a chunk instance, which contains:
            "mix": ndarray,
            "ref": [ndarray...]
        """
        chunk = dict()
        chunk["mix1"] = eg["mix1"][s:s + self.chunk_size]
        chunk["mix2"] = eg["mix2"][s:s + self.chunk_size]
        chunk["ref"] = eg["ref"][s:s + self.chunk_size]
        chunk["aux"] = eg["aux"]
        chunk["aux_len"] = eg["aux_len"]
        chunk["valid_len"] = int(self.chunk_size)
        chunk["spk_idx"] = eg["spk_idx"]
        return chunk

    def split(self, eg):
        N = eg["mix1"].size
        # too short, throw away
        if N < self.least:
            return []
        chunks = []
        # padding zeros
        if N < self.chunk_size:
            P = self.chunk_size - N
            chunk = dict()
            chunk["mix1"] = np.pad(eg["mix1"], (0, P), "constant")
            chunk["mix2"] = np.pad(eg["mix2"], (0, P), "constant")
            chunk["ref"] = np.pad(eg["ref"], (0, P), "constant")
            chunk["aux"] = eg["aux"]
            chunk["aux_len"] = eg["aux_len"]
            chunk["valid_len"] = int(N)
            chunk["spk_idx"] = eg["spk_idx"] 
            chunks.append(chunk)
        else:
            # random select start point for training
            s = random.randint(0, N % self.least) if self.train else 0
            while True:
                if s + self.chunk_size > N:
                    break
                chunk = self._make_chunk(eg, s)
                chunks.append(chunk)
                s += self.least
        return chunks


class DataLoader(object):
    """
    Online dataloader for chunk-level PIT
    """
    def __init__(self,
                 dataset,
                 num_workers=4,
                 chunk_size=32000,
                 batch_size=4,
                 train=True):
        self.batch_size = batch_size
        self.train = train
        self.splitter = ChunkSplitter(chunk_size,
                                      train=train,
                                      least=chunk_size // 2)
        # just return batch of egs, support multiple workers
        self.eg_loader = dat.DataLoader(dataset,
                                        batch_size=batch_size // 2,
                                        num_workers=num_workers,
                                        shuffle=train,
                                        collate_fn=self._collate)

    def _collate(self, batch):
        """
        Online split utterances
        """
        chunk = []
        for eg in batch:
            chunk += self.splitter.split(eg)
        return chunk

    def _pad_aux(self, chunk_list):
        lens_list = []
        for chunk_item in chunk_list:
            lens_list.append(chunk_item['aux_len'])
        max_len = np.max(lens_list)
        
        
        for idx in range(len(chunk_list)):
            P = max_len - len(chunk_list[idx]["aux"])
            chunk_list[idx]["aux"] = np.pad(chunk_list[idx]["aux"], (0, P), "constant")

        return chunk_list

    def _merge(self, chunk_list):
        """
        Merge chunk list into mini-batch
        """
        N = len(chunk_list)
        if self.train:
            random.shuffle(chunk_list)
        blist = []
        for s in range(0, N - self.batch_size + 1, self.batch_size):
            batch = default_collate(self._pad_aux(chunk_list[s:s + self.batch_size]))
            blist.append(batch)
        rn = N % self.batch_size
        return blist, chunk_list[-rn:] if rn else []

    def __iter__(self):
        chunk_list = []
        for chunks in self.eg_loader:
            chunk_list += chunks
            batch, chunk_list = self._merge(chunk_list)
            for obj in batch:
                yield obj