from ase.io import read, Trajectory
import numpy as np
import matplotlib.pyplot as plt 
import subprocess

ls_result = subprocess.run(['ls'], stdout=subprocess.PIPE, text=True)
print(f'\n"""\n{ls_result.stdout}\n"""\n')

try:
    log = open('active.log')
except:
    log_name = input(" > What is your log file's name? (ex: active-16.log) : ")
    log = open(log_name)
log_lines = log.readlines()
x_range = np.arange(int(log_lines[-1].split()[2]))
xy_values = []
for i in range(len(log_lines)):
    if 'R2' in log_lines[i].split()[-2]:
        xy_values += [[int(log_lines[i].split()[2]), float(log_lines[i].split()[-1])]]

command = 'cat'
try:
    filename = 'train.py'
    train_params = open(filename)
except:
    filename = input(' > What is your filename(or path) to train? (ex: train.py) : ')
    train_params = open(filename)
result = subprocess.run([command, filename], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
output = result.stdout
print(f'\n"""\n{output}\n"""\n')
train_params = train_params.readlines()
kernel_tag = False
for i in range(len(train_params)):
    if 'kernel_kw' in train_params[i].split('\n')[0]:
        kernel = [train_params[i].split('\n')[0]]
        kernel_tag = True
        break
if not kernel_tag:
    kernel = ['kernel_kw = dict(lmax=3, nmax=3, exponent=4, cutoff=6.0)']

train_set = input(" > What is your train set's name? (ex: Si_16, if filename is 'Si_16.traj') : ")
ediff = input(" > What is the ediff value? : ")
fdiff = input(" > What is the fdiff value? : ")
ediff_tot = input(" > What is the ediff_tot value? : ")

try:
    filename = 'predict.log'
    predict = open(filename)
except:
    filename = input(' > What is your filename(or path) to test? (ex: predict.py) : ')
    predict = open(filename)
score = float(predict.readlines()[-3].split()[-1])

a = Trajectory(f'{train_set}.traj')
cell_ls = []
for i in range(len(a)):
    cell = a[i].get_cell()
    cell_ls += [cell[2][2]]
print(f'\nmax(z-axis of cell)    : {max(cell_ls):2f}')
print(f'min(z-axis of cell)    : {min(cell_ls):2f}')
print(f'average(z-axis of cell): {np.average(cell_ls):2f}\n')
    
x, y = zip(*xy_values)
plt.xticks
plt.plot(x, y, marker='o', color='black', alpha=0.7, label=f'train_set')
plt.annotate(f'max={max(y)}', xy=(x[y.index(max(y))], max(y)),
             xytext=(x[y.index(max(y))], max(y)+0.005),
             ha='center', va='top', color='black')

x_position = int(log_lines[-1].split()[2])
y_position = min(y) 
adj = int(log_lines[-1].split()[2])*0.03
plt.text(x_position-(adj), y_position, s=f'score = {score:.2f}\ntrain set = {train_set}\nmean(z-axis) = {np.average(cell_ls):.2f}\nediff = {ediff}\nfdiff = {fdiff}\nediff_tot = {ediff_tot}', 
         fontsize=11, ha='right', va='bottom', bbox=dict(facecolor='white', alpha=0.7))
#plt.xticks(x_range)
plt.xlim(-20,int(log_lines[-1].split()[2]))
plt.ylim(min(y)-0.01, max(y)+0.01)
plt.xlabel('Trainig data' ,fontsize=13)
plt.ylabel('$R^2$', fontsize=13)
plt.title(f'{kernel[0]}', fontsize=12)
plt.savefig(f"{train_set}.png")
plt.show()
    
