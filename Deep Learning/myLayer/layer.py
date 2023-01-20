from common.layers import *
class TwoLayerNet:
    def __init__(self,input_size=784,hidden_size=50,output_size=10,weight_init_std=0.01):
        #self.params={}
        #self.params['w1']=np.random.randn(input_size,hidden_size)
        #self.params['b1']=np.zeros(hidden_size)
        #self.params['w2']=np.random.randn(hidden_size,output_size)
        #self.params['b2']=np.zeros(output_size)
                # 初始化权重
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size) 
        self.params['b2'] = np.zeros(output_size)
        
        self.affine1=Affine(self.params['W1'],self.params['b1'])
        self.relu1=Relu()
        self.affine2=Affine(self.params['W2'],self.params['b2'])
        self.lastLayer=SoftmaxWithLoss()
        
        
    
    def predict(self,x):
        x=self.affine1.forward(x)
        x=self.relu1.forward(x)
        x=self.affine2.forward(x)
        return x
        
    
    #def loss(self,x,t):
        #out=self.predict(x)
        #y_p,loss=self.lastLayer.forward(out,t)
       # return y_p,loss
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)
    
    def gradient(self,x,t):
        grads={}
        self.loss(x,t)
        dout=1
        dx=self.lastLayer.backward(dout)
        dx=self.affine2.backward(dx)
        dx=self.relu1.backward(dx)
        dx=self.affine1.backward(dx) 
        grads['W1']=self.affine1.dW
        grads['W2']=self.affine2.dW
        grads['b1']=self.affine1.db
        grads['b2']=self.affine2.db
        return grads
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy