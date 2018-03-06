import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

import numpy as np
from src.fcnet import FullyConnectedNet
from src.utils.solver import Solver
from src.utils.data_utils import get_CIFAR10_data
from src.utils.data_utils import get_FER2013_data



regs = [0,0.1,0.2,0.5,0.8]
data_dict = get_FER2013_data()

losses = []
accuracies = []
trainAccs = []
f1s = []
classAccs = []

for regularisation in regs:
    model = FullyConnectedNet([120], input_dim = 48*48*1, dropout=0, reg=regularisation, dtype=np.float64, seed=237)
    number_epochs = 50
    solver = Solver(model,data_dict,optim_config={'learning_rate':1e-4},lr_decay=0.95,num_epochs=number_epochs,batch_size=200,print_every=5000,num_train_samples=40000)
    results = solver.train()
    losses.append(solver.loss_history)
    accuracies.append(solver.val_acc_history)
    trainAccs.append(solver.train_acc_history)
    f1s.append(results["F1"])
    classAccs.append(results["recall"])
#print(losses, accuracies)


fontP = FontProperties()
fontP.set_size('small')

fig = plt.figure(figsize=(3,6))
plt.subplot(2,2,1)
plt.title("Training Loss")
for i, loss in enumerate(losses):
    plt.plot(loss,'-o',ms=0.1,label="regularisation: " + str(regs[i]))
plt.ylim(0, 6)
plt.legend(prop=fontP,loc='upper right')


plt.subplot(2,2,2)
plt.title("Classification Rate (Validation Set)")
for i, accuracy in enumerate(accuracies):
    plt.plot(accuracy,'-o',ms=1, label="regularisation: " + str(regs[i]))
plt.plot([0.5]* len(solver.val_acc_history),'k--')
plt.legend(prop=fontP,loc='upper left')
plt.xlim(0,number_epochs)

plt.subplot(2,2,3)
plt.title("Classification Rate (Training Set)")
for i, accuracy in enumerate(trainAccs):
    plt.plot(accuracy,'-o',ms=1,label="regularisation: " + str(regs[i]))
plt.plot([0.5]* len(solver.train_acc_history),'k--')
plt.legend(prop=fontP,loc='upper left')
plt.xlim(0,number_epochs)

plt.subplot(2,2,4)
plt.title("F1 Per Class (Validation Set)")
xTicks = [str(reg) for reg in regs]
colours = ['g', 'b', 'r', 'w', 'c', 'm', 'y', 'k', "#505050", "#DD1000"]
for i, f1 in enumerate(f1s):
    for j, f in enumerate(f1):
        plt.bar(1.5*i+0.1*j, f, width=0.1, color=colours[j], label=j)
plt.xlabel('Regularisation')
plt.xticks([1.5*i+0.5 for i in range(len(f1s))], xTicks)


#plt.plot(solver.loss_history,'-o',label='train')


#plt.subplot(2,2,4)
#plt.title("Classification Rate Per Class (Validation Set)")
#xTicks = [str(rate) for rate in learningRates]
#colours = ['g', 'b', 'r', 'w', 'c', 'm', 'y', 'k', "#505050", "#DD1000"]
#for i, classAcc in enumerate(classAccs):
#    for j, acc in enumerate(classAcc):
#        plt.bar(1.5*i+0.1*j, acc, width=0.1, color=colours[j], label=j)
#plt.xlabel('Learning Rate')
#plt.xticks([1.5*i+0.5 for i in range(len(f1s))], xTicks)

plt.gcf().set_size_inches(15,12)

# plt.show()
fig.savefig("src/optimising/outputs/learningReg.png")
