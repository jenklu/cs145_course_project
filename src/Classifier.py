from abc import ABC, abstractmethod

class Classifier(ABC):
    """ Classifier template class. """
    
    def __init__(self, type):
        self.type = type
        
    @abstractmethod
    def train(self, reviews, users, businesses):
        pass
    
    @abstractmethod
    def classify(self, queries):
        """ Should return a pandas Series """
        pass
    
    def write(self, output):
        #with open(f"{self.type}_submission", 'w+') as f:
        #   f.write(output)
        output.to_csv(path=f"../output/{self.type}_submission.csv", header=True, index_label="index")
