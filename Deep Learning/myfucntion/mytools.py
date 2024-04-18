import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
 
class My_funcation:
    def loss_fig(train_list,test_list,type='loss'):  
            fig = plt.gcf()
            fig.set_size_inches(10, 5)
            plt.xlabel('Epochs', fontsize=15)
            if type == 'loss':  
                label1='Train loss'
                label2='Test loss'
                title='Training and Test loss'
            elif type=='r2':
                label1='Train R2'
                label2='Test R2'
                title='Training and Test R2'

            plt.ylabel('Loss', fontsize=15)
            plt.plot(train_list, 'blue', label=label1)
            plt.plot(test_list, 'red', label=label2)
            plt.legend(loc='best')
            plt.title(title, fontsize=15)
            plt.show()