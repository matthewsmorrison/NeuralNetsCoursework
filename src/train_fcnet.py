import numpy as np

from src.fcnet import FullyConnectedNet
from src.utils.solver import Solver
from src.utils.data_utils import get_CIFAR10_data
import matplotlib.pyplot as plt

"""
TODO: Use a Solver instance to train a TwoLayerNet that achieves at least 50%
accuracy on the validation set.
"""
###########################################################################
#                           BEGIN OF YOUR CODE                            #
###########################################################################
data_dict = get_CIFAR10_data()

model = FullyConnectedNet([120],reg=0.5, dtype=np.float64)
number_epochs = 15
solver = Solver(model,data_dict,optim_config={'learning_rate':1e-4},lr_decay = 1,num_epochs=number_epochs,batch_size=200,print_every=5000,num_train_samples=40000)
solver.train()

plt.subplot(2,1,1)
plt.title("Training loss")
plt.plot(solver.loss_history,'-o',label='train')
plt.plot(solver.val_acc_history,'-o',label='val')
plt.plot([0.5]* len(solver.val_acc_history),'k--')
plt.xlabel('Epoch')
plt.xlim(0,number_epochs)
plt.legend(loc='lower right')
plt.gcf().set_size_inches(15,12)
plt.show()
##############################################################################
#                             END OF YOUR CODE                               #
##############################################################################
