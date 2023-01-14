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

def cross_entropy_e(y,y_p):
    """更新之前交叉熵方法，扩展为多个测试集相加后平均情况，但也可以接受单个测试数据,在此直接基于numpy.sum实现多个维度全运算

    Args:
        y (_type_): 真值，可以是一维、二维数组，但最终计算时都必须转换为矩阵
        y_p (_type_): 预测值，可以是一维、二维数组，但最终计算时都必须转换为矩阵
    """
    if y.ndim==1:#当输入为一维数组（单个数据、图片）需将其转换为二维度矩阵，以统一运算
        y=y.reshape(1,y.size)#1行，k列的矩阵
        y_p=y_p.reshape(1,y_p.size)
        
        
    delta=1e-7# 为了防止计算结果溢出
    n=y.shape[0]
    e=-np.sum(y*np.log(y_p+delta))/n
    return e
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 监督数据是one-hot-vector的情况下，转换为正确解标签的索引
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


def softmax_loss(X, t):
    y = softmax(X)
    return cross_entropy_error(y, t)
def mean_sqared_e(y,y_p):
    """均方误差函数

    Args:
        y (_type_): 真值
        y_p (_type_):预测值
    Returns:
        _type_: _description_
    """
    e=(np.sum((y-y_p)**2))/2
    return e