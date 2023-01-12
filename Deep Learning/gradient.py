def numerical_gradient(f,x_s):
    """梯度求解
    注意，使用的是数值微分法，是基于微小值不断求导。即基于最原始的 df/dx=lim (f(x+h)-f(x-h))/h h->0 的形式
    Args:
        f (str): 所要求梯度的函数
        x_s (np.array): 计算梯度点的向量例如 M(x=0,y=0)

    Returns:
        np.array: M点对应函数f的梯度值
    """
    delta=1e-4
    print(delta)
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