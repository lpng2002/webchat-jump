# webchat-jump
python练手opencv线性拟合打造跳一跳脚本

本脚本适用于 win7系统+android手机+Mi5s直接使用（1920*1080）

脚本思路
-
1、主要采用python+opencv的方法，使用颜色区间识别小人的位置。

2、使用边缘识别的方法确定目标方块的顶点。

![Image text](https://raw.github.com/lpng2002/webchat-jump/master/tyt/img88f.png)

3、根据方块移动的方向固定不变的规律，画出两条直线，此直线是必定经过方块中心的，然后算出目标中心坐标和跳跃距离。

![Image text](https://raw.github.com/lpng2002/webchat-jump/master/tyt/imga.png)
![Image text](https://raw.github.com/lpng2002/webchat-jump/master/tyt/imgb.png)

4、抓取跳跃前后的图片，截取跳跃后小人正下方固定区域图片（下图黄色矩形区域为匹配到的结果），匹配跳跃前的图片，得到小人实际落点和实际跳跃距离（实际落点为下图黄色矩形上方一点点） ，计算出实际跳跃距离对应的按压系数 。

![Image text](https://raw.github.com/lpng2002/webchat-jump/master/tyt/img88e.png)
![Image text](https://raw.github.com/lpng2002/webchat-jump/master/tyt/img89e.png)

5、通过采集多次实际跳跃距离的按压系数 ，使用matplotlib库求出按压系数与跳跃距离间的抛物线拟合曲线。

import numpy as np

import matplotlib.pyplot as plt

x = [。。。]   #实际跳跃距离的集合

y = [。。。]   #实际按压系数的集合

def polyfit(x, y, degree):

    results = {}
    
    coeffs = np.polyfit(x, y, degree)
    
    results['polynomial'] = coeffs.tolist()
    
    p = np.poly1d(coeffs)
    
    yhat = p(x)                         # or [p(z) for z in x]
    
    ybar = np.sum(y)/len(y)          # or sum(y)/len(y)
    
    ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
    
    sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
    
    results['determination'] = ssreg / sstot #准确率
    
    return results
    
z1 = polyfit(x, y, 2)

print(z1)

![Image text](https://raw.github.com/lpng2002/webchat-jump/master/tyt/F1.png)

6、优化脚本函数，确认最终的脚本，开始起跳吧。初步估计肯定能上千，只是我不敢跳那么高，刷分不重要，中途都手动断掉了，制作此脚本只为练手和学习，加深对python中各种库的使用熟练程度，欢迎多多交流！
