import numpy as np 
import myfucntion.Function as my 

class Relu:
    def __init__(self):
        self.mask=None
    def forward(self,x):
        """ relu 激活函数正向传播函数，其中mask为一个数组，标识了输入数组对应index是否<0，
        如果<0 对应为ture

        Args:
            x (_np.array_): 输入的一组向量
        """
        self.mask=(x<=0)
        out=x.copy()
        out[self.mask]=0# 
        return out 
    def backward(self,dout):
        dout[self.mask]=0
        dx=dout
        return dx 
    
class Sigmoid:
    def __init__(self):
        self.x=None
    def sigmoid(self,x):
        return 1/(1-np.exp(-x))
    def forward(self,x):
        self.x=x
        y=self.sigmoid(x)
        return y 
    def backward(self,dout):
        dx=self.sigmoid(self.x)(1-self.sigmoid(self.x))
        return dx 
    
class Affine:
    def __init__(self,W,B):
        self.X=None
        self.W=W
        self.B=B
        self.dw=None
        self.db=None
    def forward(self,X,W,B):
        self.X=X
        self.W=W
        out=X@W+B
        return out
    def backward(self,dout):
        dx=dout@self.W.t
        self.dw=self.X.t@dout
        self.db=np.sum(dout,axis=0)

class SoftmaxWithLoss:
    def __init__(self):
        self.x=None
        self.t=None
        
    def forward(self,x,t):
        self.x=x
        self.t=t
        y=my.softmax(x)
        er=my.cross_entropy_e(y,t)
        return y,er 
    
    def backward(self,dout=1):
        """在进行softmaxWithLoss层的反向传播时，需要将传播的值除批处理（min-batch）大小，传递给前面层的才是单个数据的误差

        Args:
            dout (int, optional): _description_. Defaults to 1.

        Returns:
            _type_: _description_
        """
        batch_size=self.t.shape[0]
        dx=(my.softmax(self.x)-self.t)/batch_size
        return dx 