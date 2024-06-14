import math
import numpy as np
from torch.utils.data import Dataset
import bisect

class MyTrainSet(Dataset):
    def __init__(self, dataset, total_steps = None, ratio=[0.25,0.5], num_epoch=None, delta=0.85, quantiles=[20,85], reverse=False):
        self.dataset = dataset
        self.ratio = ratio
        self.total_steps = total_steps
        self.num_epoch = num_epoch
        self.delta = delta
        self.scores = np.ones([len(self.dataset)])
        self.weights = np.ones(len(self.dataset))
        self.save_num = 0
        self.quantiles = quantiles
        self.reverse = reverse

    def __setscore__(self, indices, values):
        self.scores[indices] = values

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        weight = self.weights[index]
        return data, index, weight

    def __bucketize__(self,quantile_thresholds,leq=False):
        select_func = bisect.bisect_left if leq else bisect.bisect_right
        well_learned_samples = [[] for i in range(len(quantile_thresholds)+1)]
        for id,value in enumerate(self.scores):
            well_learned_samples[select_func(quantile_thresholds,value)].append(id)
        return well_learned_samples

    def prune(self, leq=False):
        # prune samples that are well learned, rebalence the weight by scaling up remaining
        # well learned samples' learning rate to keep estimation about the same
        # for the next version, also consider new class balance

        if self.reverse:
            self.scores = -np.abs(self.scores)
        quantile_thresholds = [np.percentile(self.scores,q,axis=0) for q in self.quantiles]
        bucktized_samples = self.__bucketize__(quantile_thresholds,leq)
        print([len(l) for l in bucktized_samples])
        pruned_samples = []
        pruned_samples.extend(bucktized_samples[-1])
        well_learned_samples = bucktized_samples[:-1]
        print([len(l) for l in well_learned_samples])
        selected_q = [np.random.choice(well_learned_samples[i], \
            int(self.ratio[i]*len(well_learned_samples[i])),replace=False) for i in range(len(well_learned_samples))]

        self.reset_weights()
        for i,selected in enumerate(selected_q):
            if len(selected)>0:
                self.weights[selected]=1./self.ratio[i]
                pruned_samples.extend(selected)
        print('Cut {} samples for next iteration'.format(len(self.dataset)-len(pruned_samples)))
        self.save_num += len(self.dataset)-len(pruned_samples)
        np.random.shuffle(pruned_samples)
        return pruned_samples

    def pruning_sampler(self):
        return InfoBatchSampler(self, self.num_epoch, self.delta, total_steps=self.total_steps)

    def no_prune(self):
        samples = list(range(len(self.dataset)))
        np.random.shuffle(samples)
        return samples

    def mean_score(self):
        return self.scores.mean()

    def normal_sampler_no_prune(self):
        return InfoBatchSampler(self.no_prune)

    def get_weights(self,indexes):
        return self.weights[indexes]

    def total_save(self):
        return self.save_num

    def reset_weights(self):
        self.weights = np.ones(len(self.dataset))



class InfoBatchSampler():
    def __init__(self, infobatch_dataset, num_epoch = math.inf, delta = 1, total_steps=math.inf):
        self.infobatch_dataset = infobatch_dataset
        self.seq = None
        self.stop_prune = min(num_epoch * delta, total_steps*delta)
        self.seed = 0
        self.reset()

    def reset(self):
        np.random.seed(self.seed)
        self.seed+=1
        if self.seed>self.stop_prune:
            if self.seed <= self.stop_prune+1:
                self.infobatch_dataset.reset_weights()
            self.seq = self.infobatch_dataset.no_prune()
        else:
            self.seq = self.infobatch_dataset.prune(self.seed>1)
        self.ite = iter(self.seq)
        self.new_length = len(self.seq)

    def __next__(self):
        try:
            nxt = next(self.ite)
            return nxt
        except StopIteration:
            self.reset()
            raise StopIteration

    def __len__(self):
        return len(self.seq)

    def __iter__(self):
        self.ite = iter(self.seq)
        return self