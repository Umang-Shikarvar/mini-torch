import math
import random

class DataLoader:

    def __init__(self, dataset, batch_size = 1, shuffle = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._reset()
        self.current_index = 0

 
    def _reset(self): # to reset the iterator after each epoch
        self.indices = list(range(len(self.dataset)))

        if self.shuffle:
            random.shuffle(self.indices)
        
        self.current_index = 0
        
    def __iter__(self):
        self._reset()
        return self
    
    def __next__(self):
        if self.current_index >= len(self.indices):
            raise StopIteration
        
        start = self.current_index
        end = min(start + self.batch_size, len(self.indices)) 
        batch_indices = self.indices[start:end]

        batch = [self.dataset[i] for i in batch_indices]
        self.current_index = end

        return zip(*batch)
    
    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)
    
    def __repr__(self):
        return f"DataLoader(batch_size={self.batch_size}, shuffle={self.shuffle}, dataset_size={len(self.dataset)})"
    
    