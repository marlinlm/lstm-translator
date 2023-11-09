import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os

if __name__=="__main__":
    loss_out_file="../loss.log"
    x=[]
    y=[]
    idx = 0
    with open(loss_out_file, 'r', encoding='utf-8') as ff:
        for line in ff.readlines():
            token = line.split(',')
            y.append(float(token[3][:-1]))
            x.append(idx)
            idx += 1

    plot1=plt.plot(x,y,'*',label='sigmoid(x)')
    
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.title('Logit Function')
    plt.legend(loc=4)
    plt.grid(True)

