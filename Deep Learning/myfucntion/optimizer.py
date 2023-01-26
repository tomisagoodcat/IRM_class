import numpy as np 
class AdaGrad:
    def __init__(self,learning_rate=0.01):
        self.lr=learning_rate
        self.h=None
    def update(self,params,grads):
        if self.h is None:
            self.h={}
            for key,val in params.items():
                self.h[key]=np.zeros_like(val)
        for key in params.keys():
            self.h[key]+=grads[key]*grads[key]
            params[key]-=(grads[key]*self.lr)/(np.sqrt(self.h[key])+1e-7)# 加入1e-7,避免出现self.h[key]为0的情况

class Momentum:
    
    """Momentum SGD"""

    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None
        
    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():                                
                self.v[key] = np.zeros_like(val)
                
        for key in params.keys():
            self.v[key] = self.momentum*self.v[key] - self.lr*grads[key] 
            params[key] += self.v[key]