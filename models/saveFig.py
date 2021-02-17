import matplotlib.pyplot as plt
def saveFig(model_name, parameter, data_train, data_test, version):
    fig, axs = plt.subplots(2)
    #fig = plt.figure()
    fig.canvas = plt.FigureCanvasBase(fig) 
    axs[0].plot(data_train)
    axs[1].plot(data_test)
    fig.suptitle('model_{}_{}'.format(model_name,parameter), fontsize=16)
    plt.xlabel('epochs', fontsize=16)
    axs[0].set(ylabel = 'train_' + parameter)
    axs[1].set(ylabel = 'test_' + parameter)
    fig.savefig('graphs/v{}_model_{}_{}.jpg'.format(version,model_name,parameter))