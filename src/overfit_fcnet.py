import numpy as np
import matplotlib.pyplot as plt
from src.fcnet import FullyConnectedNet
from src.utils.solver import Solver
from src.utils.data_utils import get_CIFAR10_data

"""
TODO: Overfit the network with 50 samples of CIFAR-10
"""
###########################################################################
#                           BEGIN OF YOUR CODE                            #
###########################################################################
data_dict = get_CIFAR10_data(num_training=50)

model = FullyConnectedNet([100],reg=0, dtype=np.float64)

solver = Solver(model,data_dict,optim_config={'learning_rate':1e-4},num_epochs=20,batch_size=50,print_every=50)
solver.train()

plt.subplot(2,1,1)
plt.title("Training loss")
plt.plot(solver.loss_history,'-o',label='train')
plt.plot(solver.val_acc_history,'-o',label='val')
plt.plot([0.5]* len(solver.val_acc_history),'k--')
plt.xlabel('Epoch')
plt.xlim(0,20)
plt.legend(loc='lower right')
plt.gcf().set_size_inches(15,12)
plt.show()

##############################################################################
#                             END OF YOUR CODE                               #
##############################################################################
