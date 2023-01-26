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

class TwoLayerNet_Op(TwoLayerNet):
    """_summary_

    Args:
        TwoLayerNet (_type_): _description_
    """
    def __init__(self,input_size=784,hidden_size=50,output_size=10):
        """_summary_

        Args:
            input_size (int, optional): _description_. Defaults to 784.
            hidden_size (int, optional): _description_. Defaults to 50.
            output_size (int, optional): _description_. Defaults to 10.
        """
   
                # 初始化权重
        self.params = {}
        self.params['W1'] =   np.random.randn(input_size, hidden_size)*(2/np.sqrt(input_size))
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] =  np.random.randn(hidden_size, output_size) *(2/np.sqrt(hidden_size))
        self.params['b2'] = np.zeros(output_size)
        
        self.affine1=Affine(self.params['W1'],self.params['b1'])
        self.relu1=Relu()
        self.affine2=Affine(self.params['W2'],self.params['b2'])
        self.lastLayer=SoftmaxWithLoss()
        
class ThreeLayerNet_Op(TwoLayerNet):
        def __init__(self,input_size=784,hidden_size=50,m_hiddent_size=50,output_size=10):
            """_summary_第三层网络激活函数使用sigmoid，权参初始std使用Xavier（1/np.sqrt(前面节点数目)

            Args:
                input_size (int, optional): _description_. Defaults to 784.
                hidden_size (int, optional): _description_. Defaults to 50.
                m_hiddent_size (int, optional): _description_. Defaults to 50.
                output_size (int, optional): _description_. Defaults to 10.
            """
 
                # 初始化权重
            self.params = {}
            self.params['W1'] =   np.random.randn(input_size, hidden_size)*(2/np.sqrt(input_size))
            self.params['b1'] = np.zeros(hidden_size)
            self.params['W2'] =  np.random.randn(hidden_size, m_hiddent_size) *(1/np.sqrt(hidden_size))
            self.params['b2'] = np.zeros(m_hiddent_size)
            self.params['W3'] =  np.random.randn(m_hiddent_size, output_size) *(2/np.sqrt(m_hiddent_size))
            self.params['b3'] = np.zeros(output_size)            
            self.affine1=Affine(self.params['W1'],self.params['b1'])#第一层全连接层
            self.relu1=Relu()#第2层激活层
            self.affine2=Affine(self.params['W2'],self.params['b2'])#第3全连接层
            self.relu2=Sigmoid()#第4层激活层
            self.affine3=Affine(self.params['W3'],self.params['b3'])#第5连接层
            self.lastLayer=SoftmaxWithLoss()
            
        def predict(self,x):
            x=self.affine1.forward(x)
            x=self.relu1.forward(x)
            x=self.affine2.forward(x)
            x=self.relu2.forward(x)
            x=self.affine3.forward(x)
            return x
        
        def gradient(self,x,t):
            grads={}
            self.loss(x,t)
            dout=1
            dx=self.lastLayer.backward(dout)
            dx=self.affine3.backward(dx)
            dx=self.relu2.backward(dx)
            dx=self.affine2.backward(dx)
            dx=self.relu1.backward(dx)
            dx=self.affine1.backward(dx) 
            grads['W1']=self.affine1.dW
            grads['W2']=self.affine2.dW
            grads['W3']=self.affine3.dW
            grads['b1']=self.affine1.db
            grads['b2']=self.affine2.db
            grads['b3']=self.affine3.db
            return grads            