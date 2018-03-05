import matplotlib.pyplot as plt
import numpy as np
from src.fcnet import FullyConnectedNet
from src.utils.solver import Solver
from src.utils.data_utils import get_CIFAR10_data


rateDecays = [0.99, 0.95, 0.9, 0.3]#, 1e-4, 1e-6, 1e-8]
data_dict = get_CIFAR10_data()

losses = []
accuracies = []
f1s = []
classAccs = []

for rateDecay in rateDecays:
    model = FullyConnectedNet([120], dropout=0, reg=0, dtype=np.float64, seed=237)
    number_epochs = 12
    solver = Solver(model,data_dict,optim_config={'learning_rate':1e-4},lr_decay=rateDecay,num_epochs=number_epochs,batch_size=200,print_every=5000,num_train_samples=40000)
    results = solver.train()
    losses.append(solver.loss_history)
    accuracies.append(solver.val_acc_history)
    f1s.append(results["F1"])
    classAccs.append(results["recall"])
#print(losses, accuracies)

plt.subplot(2,2,1)
plt.title("Training loss")
for i, loss in enumerate(losses):
    plt.plot(loss,'-o',ms=0.1,label="learning rate: " + str(rateDecays[i]))
plt.ylim(0, 5)
plt.legend(loc='lower right')


plt.subplot(2,2,2)
plt.title("Classification rate")
for i, accuracy in enumerate(accuracies):
    plt.plot(accuracy,'-o',label="learning rate: " + str(rateDecays[i]))
plt.plot([0.5]* len(solver.val_acc_history),'k--')
plt.legend(loc='lower right')
plt.xlim(0,number_epochs)

plt.subplot(2,2,3)
plt.title("F1 per class")
xTicks = [str(rate) for rate in rateDecays]
colours = ['g', 'b', 'r', 'w', 'c', 'm', 'y', 'k', "#505050", "#DD1000"]
for i, f1 in enumerate(f1s):
    for j, f in enumerate(f1):
        plt.bar(1.5*i+0.1*j, f, width=0.1, color=colours[j], label=j)
plt.xlabel('Decay Rate')
plt.xticks([1.5*i+0.5 for i in range(len(f1s))], xTicks)
#plt.plot(solver.loss_history,'-o',label='train')


plt.subplot(2,2,4)
plt.title("Classification rate per class")
xTicks = [str(rate) for rate in rateDecays]
colours = ['g', 'b', 'r', 'w', 'c', 'm', 'y', 'k', "#505050", "#DD1000"]
for i, classAcc in enumerate(classAccs):
    for j, acc in enumerate(classAcc):
        plt.bar(1.5*i+0.1*j, acc, width=0.1, color=colours[j], label=j)
plt.xlabel('Decay Rate')
plt.xticks([1.5*i+0.5 for i in range(len(f1s))], xTicks)

plt.gcf().set_size_inches(15,12)
plt.show()

# learning rate, rate decay, momentum, regularisation, hidden layers, neurons per layer
