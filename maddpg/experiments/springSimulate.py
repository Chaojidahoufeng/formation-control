import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.integrate as si
 
'''
模拟弹簧的强迫振动，是个二阶微分方程
dx^2/dt^2 + u/m*dx/dt + c/m*x = H/m*sinpt
第二项是阻尼，第三项是弹簧恢复力与重力等的合力加速度，右边是铅直干扰力的加速度
这里不考虑铅直干扰力
'''
m = 5 #质量
u = 2 #阻力的比例系数
c = 0.52 #弹簧劲度系数
 
#因为odeint解一阶的微分方程比如容易，所以用 dx/dt = v 改写为方程组,垂直画图，所以用y来表示x
def func(start,t,m,u,c):
    y,v = start   #y距离，v为dx/dt
    dxdt = v  #相当于距离对时间的变化  即微分方程1阶的项
    dvdt = -u/m*v - c/m*y  #速度对时间的变化
    return dxdt,dvdt
 
y0 = [0,-5] #在0时刻，位置为1
t = np.arange(0, 20, 0.05) #0.1间隔，取0-20，方便后面用时间
track = si.odeint(func, y0, t,args=(m,u,c))
xdata = [0 for i in range(len(track))] #横坐标为0
ydata = [track[i, 0] for i in range(len(track))] #纵坐标为对时间的变化
print(ydata)
#画图
fig,ax = plt.subplots()
ax.grid()#建立网格
line2, = ax.plot([], [], 'o-', lw=2)
time_text = ax.text(-1, 4, '', transform=ax.transAxes)#显示文字
def init():
    global line1,ax
    ax.set_xlim(-2,2)
    ax.set_ylim(-10,5)
    # 移位置 设为原点相交
    ax.xaxis.set_ticks_position('bottom')  # 设置为底部
    ax.spines['bottom'].set_position(('data', 0))  # 获取底部轴设置其位置，表示设置底部轴移动到竖轴的0坐标位置
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data', 0))
    #弹簧
    line1 = ax.plot([0,0],[0,3],'o-',lw=2)
    return ax
 
def update(i):
    line2.set_data([0,0],[0,ydata[i]])
    time_text.set_text('time = ' + str(0.05 *i))
    return line2,time_text
 
ani = FuncAnimation(fig, update, range(1, len(xdata)), init_func=init, interval=50)
ani.save("forced_vibration.gif",writer='pillow')