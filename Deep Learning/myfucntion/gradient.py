import numpy as np  
def mynumerical_gradient(f,x_s):
    """梯度求解
    注意，使用的是数值微分法，是基于微小值不断求导。即基于最原始的 df/dx=lim (f(x+h)-f(x-h))/h h->0 的形式
    Args:
        f (str): 所要求梯度的函数
        x_s (np.array): 计算梯度点的向量例如 M(x=0,y=0)

    Returns:
        np.array: M点对应函数f的梯度值
    """
    delta=1e-4
    d_x=np.zeros_like(x_s)# 定义偏导数结果向量
    #对每一个自变量向量中自变量进行求导，并放入偏导数结果向量中
    for i in range(x_s.size):
        #根据计算公式，在此，x_s[i]对应自变量发生了改变（增加微小值），但其他自变量不变，带入方程求解的结果
        # f(x_s[x+detla,y,z]-f(x,y,z))
        origial_x=x_s[i]#保留原始的x_s[i]值
       
        x_s[i]=origial_x+delta
        #print("x+delta:",x_s[i])
        r1=f(x_s)
        #print("f(x+delta,y)计算结果：",r1)
      
        x_s[i]=origial_x-delta
        print("x-delta:",x_s[i])
        r2=f(x_s)
        #print("f(x-delta,y)计算结果：",r2)
        
        d_x[i]=(r1-r2)/(2*delta)
        x_s[i]=origial_x#还原 x_s[i]
        #print("f1-f2/h计算结果",d_x[i])
        
    return d_x
def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 还原值
        it.iternext()   
        
    return grad

def gradient_descent(f,init_x,lr=0.01,step_num=100):
    """梯度下降法

    Args:
        f (_type_): 误差函数
        init_x (_type_): 权重初始值
        lr (float, optional): 学习率.
        step_num (int, optional): 循环次数. Defaults to 100.

    Returns:
        _type_: 更新后的新权重，实现误差函数达到极小值
    """
    x=init_x
    for i in range(step_num):
        grad=numerical_gradient(f,x)
        x=x-lr*grad
    return x 
 