import os

class DataLoader():
    def __init__(self):
        self.path = 'data/train/data.txt'
        self.data_loader = self.load_data()
        
    def load_data(self):
        dataset = []

        with open(os.path.realpath(self.path), 'r', encoding='utf-8') as f:
            for line in f:
                dataset.append(line.strip())
        return dataset