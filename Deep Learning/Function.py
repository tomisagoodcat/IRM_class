import numpy as np 

def identify(self,a):
        """恒等函数

        Args:
            a (_type_): _description_

        Returns:
            _type_: _description_
        """
        return a
def softmax(a:np.array)->np.array:
        
        c=np.max(a)
        return np.exp(a-c)/np.sum(np.exp(a-c))
    
def sigmoid(x:np.array)->np.array:
        return 1/(1+np.exp(-x))