class Average():
    def __init__(self):
        super().__init__()
        self.count=0
        self.sum=0
    def update(self,value,N=1):
        self.count+=1
        self.sum+=value*N
    def reset(self):
        self.count=0
        self.sum=0
    def avg(self):
        return self.sum/float(self.count)
