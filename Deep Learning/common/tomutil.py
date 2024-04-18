import numpy as np 
import matplotlib.pyplot as plt
def loss_acc_fig(train_loss, test_loss, train_acc, test_acc):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))  # 创建一个图形，其中包含1行2列的子图
    
    # 绘制损失子图
    axes[0].plot(train_loss, 'blue', label='Train Loss')
    axes[0].plot(test_loss, 'red', linestyle='--',label='Test Loss')
    axes[0].set_title('Training and Test Loss', fontsize=15)
    axes[0].set_xlabel('Epochs', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].legend(loc='best')
    
    # 绘制准确率子图
    axes[1].plot(train_acc, 'blue', label='Train Accuracy')
    axes[1].plot(test_acc, 'red',linestyle='--', label='Test Accuracy')
    axes[1].set_title('Training and Test Accuracy', fontsize=15)
    axes[1].set_xlabel('Epochs', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].legend(loc='best')
    
    plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
    plt.show()