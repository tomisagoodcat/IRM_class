from common.layers import *
from collections import OrderedDict
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
    """继承基础的两层网络，但替换权参初始方法，使用标准差=He生产权参

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
            self.params['W2'] =  np.random.randn(hidden_size, m_hiddent_size) *(2/np.sqrt(hidden_size))
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
        

     
class MultiLayerNet(TwoLayerNet):   
    """全连接的多层神经网络

    Parameters
    ----------
    input_size : 输入大小（MNIST的情况下为784）
    hidden_size_list : 隐藏层的神经元数量的列表（e.g. [100, 100, 100]）
    output_size : 输出大小（MNIST的情况下为10）
    activation : 'relu' or 'sigmoid'
    weight_init_std : 指定权重的标准差（e.g. 0.01）
        指定'relu'或'he'的情况下设定“He的初始值”
        指定'sigmoid'或'xavier'的情况下设定“Xavier的初始值”
    weight_decay_lambda : Weight Decay（L2范数）的强度
    """
    def __init__(self,input_size=784,hidden_size_list=[100,100,100,100,100,100],output_size=10,activation='relu',weight_init_std='relu',weight_decay_lambda=0):
        self.hidden_size_list=hidden_size_list
        self.params={}
        self.input_size=input_size
        self.output_size=output_size
        self.weight_init_std=weight_init_std
        self.activation=activation
        self.hidden_layer_num=len(hidden_size_list)
        self.weight_decay_lambda=weight_decay_lambda
        #调用函数生成初始权参
        self.__init_weight()
        #生成网络
        self.layers=OrderedDict()
        self.layers = OrderedDict()
        for idx in range(1, self.hidden_layer_num+1):
            self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)],
                                                      self.params['b' + str(idx)])
            self.layers['Activation_function' + str(idx)] = Relu()
        #最后输出加loss层
        last_num=self.hidden_layer_num+1
        self.layers['Affine'+str(last_num)]=Affine(self.params['W'+str(last_num)],self.params['b'+str(last_num)])
        self.lastlayer=SoftmaxWithLoss()
    
    def __init_weight(self, weight_init_std='relu'):
        """设定权重的初始值

        Parameters
        ----------
        weight_init_std : 指定权重的标准差（e.g. 0.01）
            指定'relu'或'he'的情况下设定“He的初始值”
            指定'sigmoid'或'xavier'的情况下设定“Xavier的初始值”
        """
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        for idx in range(1, len(all_size_list)):
            scale = weight_init_std
            if str(weight_init_std).lower() in ('relu', 'he'):
                scale = np.sqrt(2.0 / all_size_list[idx - 1])  # 使用ReLU的情况下推荐的初始值
            elif str(weight_init_std).lower() in ('sigmoid', 'xavier'):
                scale = np.sqrt(1.0 / all_size_list[idx - 1])  # 使用sigmoid的情况下推荐的初始值

            self.params['W' + str(idx)] = scale * np.random.randn(all_size_list[idx-1], all_size_list[idx])
            self.params['b' + str(idx)] = np.zeros(all_size_list[idx])

    def predict(self,x):
        for layer in self.layers.values():
            x=layer.forward(x)
        return x
                
                
    def loss(self,x,t):
        y=self.predict(x)
        weight_decay = 0
        for idx in range(1,self.hidden_layer_num+1):
            weight_decay=np.sum(self.params['W'+str(idx)]**2)
        loss=self.lastlayer.forward(y,t)+self.weight_decay_lambda*weight_decay*0.5
        return loss
        
        
    def gradient(self,x,t):
        loss=self.loss(x,t)
        
        loss_dout=1#loss的导数为1
        dout=self.lastlayer.backward(loss_dout) 
        layers=list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout=layer.backward(dout)
        grads={}
        for idx in range(1,self.hidden_layer_num+2):
            grads["W"+str(idx)]=self.layers["Affine"+str(idx)].dW+self.weight_decay_lambda*0.5*self.layers["Affine"+str(idx)].dW
            grads["b"+str(idx)]=self.layers["Affine"+str(idx)].db
        

        return grads
            
 
            
           
class Dropout:
    """
    http://arxiv.org/abs/1207.0580
    """
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask
          
class MultiLayerNet_dropout(MultiLayerNet):  
    def __init__(self,input_size=784,hidden_size_list=[100,100,100,100,100,100],output_size=10,activation='relu',weight_init_std='relu',weight_decay_lambda=0,dropout_ratio=0.5):
     
        self.hidden_size_list=hidden_size_list
        self.params={}
        self.input_size=input_size
        self.output_size=output_size
        self.weight_init_std=weight_init_std
        self.activation=activation
        self.hidden_layer_num=len(hidden_size_list)
        self.weight_decay_lambda=weight_decay_lambda
        #调用函数生成初始权参
        #我们在python中从某父类继承子类时，可以在子类中对父类的数据域和方法操作，但是当该数据域或方法为私有时（有双下划线作为前缀），应注意调用格式如下：
            #子类调用父类私有数据域：self._父类名+私有数据域名
            #子类调用父类私有方法：self._父类名+私有方法名
        self._MultiLayerNet__init_weight()
        #self.__init_weight()
        #生成网络
        self.layers=OrderedDict()
        self.layers = OrderedDict()
        for idx in range(1, self.hidden_layer_num+1):
            self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)],
                                                      self.params['b' + str(idx)])
            self.layers['Activation_function' + str(idx)] = Relu()
            self.layers['Dropout' + str(idx)] = Dropout()
        #最后输出加loss层
        last_num=self.hidden_layer_num+1
        self.layers['Affine'+str(last_num)]=Affine(self.params['W'+str(last_num)],self.params['b'+str(last_num)])
        self.lastlayer=SoftmaxWithLoss()

