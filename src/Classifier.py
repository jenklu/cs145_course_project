class Classifier():
    def __init__(self, type):
        self.type = type
    # override this method
    def train(self, reviews, users, businesses):
        pass
    # override this method - should return a pandas Series
    def classify(self, queries):
        pass
    def write(self, output):
        #with open(f"{self.type}_submission", 'w+') as f:
        #   f.write(output)
        output.to_csv(path=f"../output/{self.type}_submission.csv", header=True, index_label="index")
