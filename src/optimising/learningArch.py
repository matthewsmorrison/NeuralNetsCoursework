import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

from src.fcnet import FullyConnectedNet
from src.utils.solver import Solver
from src.utils.data_utils import get_CIFAR10_data
from src.utils.data_utils import get_FER2013_data
from src.utils.data_utils import get_FER2013_data_normalisation



archs = [[400,200], [400]]#, 1e-4, 1e-6, 1e-8]
data_dict = get_FER2013_data_normalisation()

losses = []
accuracies = []
trainAccs = []
f1s = []
classAccs = []
trainAccs = []


for arch in archs:
    model = FullyConnectedNet(arch, input_dim = 48*48*1, num_classes=7, dropout=0, reg=0.5, dtype=np.float64, seed=237)
    number_epochs = 120
    solver = Solver(model,data_dict,optim_config={'learning_rate':1e-4},lr_decay=0.9,num_epochs=number_epochs,batch_size=200,print_every=5000,num_train_samples=40000)
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
<<<<<<< HEAD
    plt.plot(loss,'-o',ms=0.1,label="learning rate: " + str(archs[i]))
#plt.ylim(0, 6)
=======
    plt.plot(loss,'-o',ms=0.1,label="architecture: " + str(archs[i]))
plt.ylim(0, 6)
>>>>>>> a16d8d2a80fd44e7b33267ae434d41e741fb9923
plt.legend(prop=fontP,loc='upper right')


plt.subplot(2,2,2)
plt.title("Classification Rate (Validation Set)")
for i, accuracy in enumerate(accuracies):
<<<<<<< HEAD
    plt.plot(accuracy,'-o',ms=1, label="learning rate: " + str(archs[i]))
=======
    plt.plot(accuracy,'-o',ms=1, label="architecture: " + str(archs[i]))
>>>>>>> a16d8d2a80fd44e7b33267ae434d41e741fb9923
plt.plot([0.5]* len(solver.val_acc_history),'k--')
plt.legend(prop=fontP,loc='upper left')
plt.xlim(0,number_epochs)
plt.ylim(0,0.6)

plt.subplot(2,2,3)
plt.title("Classification Rate (Training Set)")
for i, accuracy in enumerate(trainAccs):
<<<<<<< HEAD
    plt.plot(accuracy,'-o',ms=1,label="learning rate: " + str(archs[i]))
plt.plot([0.5]* len(solver.train_acc_history),'k--')
plt.legend(prop=fontP,loc='upper left')
plt.xlim(0,number_epochs)

plt.subplot(2,2,4)
plt.title("F1 Per Class (Validation Set)")
xTicks = [str(rate) for rate in archs]
colours = ['g', 'b', 'r', 'w', 'c', 'm', 'y', 'k', "#505050", "#DD1000"]
for i, f1 in enumerate(f1s):
    for j, f in enumerate(f1):
        plt.bar(1.5*i+0.1*j, f, width=0.1, color=colours[j], label=j)
plt.xlabel('Architecture')
plt.xticks([1.5*i+0.5 for i in range(len(f1s))], xTicks)


#plt.plot(solver.loss_history,'-o',label='train')


#plt.subplot(2,2,4)
#plt.title("Classification Rate Per Class (Validation Set)")
#xTicks = [str(rate) for rate in archs]
#colours = ['g', 'b', 'r', 'w', 'c', 'm', 'y', 'k', "#505050", "#DD1000"]
#for i, classAcc in enumerate(classAccs):
#    for j, acc in enumerate(classAcc):
#        plt.bar(1.5*i+0.1*j, acc, width=0.1, color=colours[j], label=j)
#plt.xlabel('Learning Rate')
#plt.xticks([1.5*i+0.5 for i in range(len(f1s))], xTicks)

#plt.gcf().set_size_inches(15,12)
=======
    plt.plot(accuracy,'-o',ms=1,label="architecture: " + str(archs[i]))
plt.plot([0.5]* len(solver.train_acc_history),'k--')
plt.legend(prop=fontP,loc='upper left')
plt.xlim(0,number_epochs)
plt.ylim(0,1)

plt.subplot(2,2,4)
plt.title("F1 Per Class (Validation Set)")
xTicks = [str(architecture) for architecture in archs]
colours = ['g', 'b', 'r', '#101010', 'c', 'm', 'y', 'k', "#505050", "#DD1000"]
for i, f1 in enumerate(f1s):
    for j, f in enumerate(f1):
        plt.bar(1.5*i+(j/7), f, width=(1/7), color=colours[j], label=j)
plt.xlabel('Architecture')
plt.xticks([1.5*i+0.5 for i in range(len(f1s))], xTicks)

plt.gcf().set_size_inches(15,12)
fig.savefig("src/optimising/outputs/testarchitecture.png")
>>>>>>> a16d8d2a80fd44e7b33267ae434d41e741fb9923

plt.show()
fig.savefig("learningArch.png")
