from ase.io import read, Trajectory
import numpy as np
import matplotlib.pyplot as plt 

log = open('active.log')
log_lines = log.readlines()
x_range = np.arange(int(log_lines[-1].split()[2]))
xy_values = []
for i in range(len(log_lines)):
    if 'R2' in log_lines[i].split()[-2]:
        xy_values += [[int(log_lines[i].split()[2]), float(log_lines[i].split()[-1])]]

train_set = input("What is your train set's name? : ")
ediff = input("What is the ediff value? : ")
fdiff = input("What is the fdiff value? : ")
ediff_tot = input("What is the ediff_tot value? : ")

try:
    train_params = open('train.py')
    train_params = train_params.readlines()
    for i in range(len(train_params)):
        if 'kernel_kw' in train_params[i].split('\n')[0]:
            kernel = [train_params[14].split('\n')[0]]
except:
    filename = input('What is your filename(or path) to train? (ex: train.py) : ')
    train_params = open(filename)
    train_params = train_params.readlines()
    for i in range(len(train_params)):
        if 'kernel_kw' in train_params[i].split('\n')[0]:
            kernel = [train_params[14].split('\n')[0]]
try:
    predict = open('predict.log')
    score = float(predict.readlines()[-3].split()[-1])
except:
    filename = input('What is your filename(or path) to test? (ex: predict.py) : ')
    predict = open(filename)
    score = float(predict.readlines()[-3].split()[-1])
    

x, y = zip(*xy_values)
plt.xticks
plt.plot(x, y, marker='o', color='black', alpha=0.7, label=f'train_set')
plt.annotate(f'max={max(y)}', xy=(x[y.index(max(y))], max(y)),
             xytext=(x[y.index(max(y))], max(y)+0.005),
             ha='center', va='top', color='black')

x_position = int(log_lines[-1].split()[2])
y_position = min(y) 
adj = int(log_lines[-1].split()[2])*0.03
plt.text(x_position-(adj), y_position, s=f'score = {score:.2f}\ntrain set = {train_set}\nediff = {ediff}\nfdiff = {fdiff}\nediff_tot = {ediff_tot}', 
         fontsize=11, ha='right', va='bottom', bbox=dict(facecolor='white', alpha=0.7))
#plt.xticks(x_range)
plt.xlim(-20,int(log_lines[-1].split()[2]))
plt.ylim(min(y)-0.01, max(y)+0.01)
plt.xlabel('Trainig data' ,fontsize=13)
plt.ylabel('$R^2$', fontsize=13)
plt.title(f'{kernel[0]}', fontsize=12)
plt.savefig(f"{train_set}.png")
plt.show()
    
