class Dataset:
    def __init__(self):
        pass

    def __len__(self):
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def __getitem__(self, index):
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]