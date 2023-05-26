import numpy as np


def attack_tests(sample_list, attack_func, database_x, database_y, cnn_model, pca_model, dnn_model):
    '''
    Run attack and return accuracy as attack varies 
    '''
    cnn_acc = np.zeros(len(sample_list))
    pca_acc = np.zeros(len(sample_list))
    dnn_acc = np.zeros(len(sample_list))

    for i, val in enumerate(sample_list):
        cnn_acc[i] = cnn_model.evaluate(attack_func(database_x, 0, val), database_y)[1]
        pca_acc[i] = pca_model.evaluate(attack_func(database_x, 0, val), database_y)[1]
        dnn_acc[i] = dnn_model.evaluate(attack_func(database_x, 0, val), database_y)[1]  

    return cnn_acc, pca_acc, dnn_acc



# Visualizing blur 

# fig = plt.figure(figsize=(5, 5))

# for i in range(5):
#     for j in range(5):
#         fig.add_subplot(i, j)
#         plt.imshow(gaussian_blur(x_train[j], i))
# plt.show()

# f, axes = plt.subplots(5, 5)
    
# for i, row in enumerate(axes):
#     for j, ax in enumerate(row):
#         ax.imshow(gaussian_blur(x_train[j], 2))

# plt.show()