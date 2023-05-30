#global imports
import matplotlib as plt
import numpy as np
import tensorflow.keras.datasets.mnist as mnist
from tensorflow.keras import utils
from tensorflow.keras import models
import matplotlib.pyplot  as plt

#local imports
from models import CNN_model, PCA_model
from attack_tests import *
from img_manipulations import *
from database_manipulations import *
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method




#CONSTANTS
MAX_BLUR = 2


#Load the main MNIST Dataset 
(x_train, y_train), (x_test, y_test) = mnist.load_data()  
x_train = x_train / 255  
x_test = x_test / 255

#get categorical versions of the labels
y_train_cat = utils.to_categorical(y_train, 10)
y_test_cat = utils.to_categorical(y_test, 10)

#special version of dataset for rotations
rx_train, ry_train = prep_rotations(x_train, y_train)
rx_test, ry_test = prep_rotations(x_test, y_test)

#get categorical versions of the rotation-ready labels
ry_train_cat = utils.to_categorical(ry_train, 10)
ry_test_cat = utils.to_categorical(ry_test, 10)


# pca_model = PCA_model(250, 10)

""" PCA Model Training Code (250 components)
pca_model = PCA_model(250, 10)
pca_model.fit(x_train, y_train_cat)
pca_model.save("pca_weights")
"""

""" DNN Model Training Code
dnn_model = PCA_model(28*28, 10, True, False)
dnn_model.fit(x_train, y_train_cat)
dnn_model.save("dnn_weights")
"""

""" CNN Model Training Code
cnn_model.fit(x_train.reshape(-1, 28, 28, 1), y_train_cat)
cnn_model.save_weights('cnn_weights')
"""

# fig, ax = plt.subplots(5, 6)
# eps = np.linspace(0, 0.5, 6)
# for i in range(5):
#     for j in range(6):
#         img = constant_noise(x_test[i], eps[j]) 
#         ax[i][j].imshow(img)
#         ax[i][j].axis('off')

# fig.suptitle('Random constant noise of inf-norm in [0, 0.5]')

# cnn = CNN_model()
# cnn.load_weights("cnn_weights")

# fig, ax = plt.subplots(5, 6)
# eps = np.linspace(0, 0.5, 6)
# for i in range(5):
#     for j in range(6):
#         img = fast_gradient_method(cnn, x_test[i].reshape(-1, 28, 28, 1), eps[j], 
#                                             np.inf, y=np.array([y_test[i]]).astype(int)).numpy().reshape(28, 28)
#         ax[i][j].imshow(img)
#         ax[i][j].axis('off')

# fig.suptitle('FGSM noise of inf-norm in [0, 0.5]')


#training the supersoldier models



# rotation_steps = list(range(0, 181, 10))
# accuracy_matrix = np.array([[0.9799576997756958, 0.9768455028533936, 0.9631519913673401, 0.9294161796569824, 0.8653056025505066, 0.7889953851699829, 0.7194074392318726, 0.6447155475616455, 0.6056267619132996, 0.5565791130065918, 0.5106436014175415, 0.47728121280670166, 0.45524710416793823, 0.4427984654903412, 0.42549481987953186, 0.4225071668624878, 0.432964026927948, 0.4369475841522217, 0.4268641769886017],
# [0.9188348054885864, 0.9123615026473999, 0.8893315196037292, 0.8539773225784302, 0.7962155938148499, 0.7354661822319031, 0.6641354560852051, 0.6078675389289856, 0.555085301399231, 0.5045437812805176, 0.47703224420547485, 0.4425494968891144, 0.42063987255096436, 0.3956180810928345, 0.3804307281970978, 0.3704718053340912, 0.36337608098983765, 0.3508029282093048, 0.35914352536201477],
# [0.9808290600776672, 0.9814515113830566, 0.9757251143455505, 0.9625295400619507, 0.9399974942207336, 0.8870907425880432, 0.8315697908401489, 0.7621062994003296, 0.702228307723999, 0.652309238910675, 0.6088634133338928, 0.565915584564209, 0.5395244359970093, 0.5113905072212219, 0.495207279920578, 0.48699116706848145, 0.482883095741272, 0.4779036343097687, 0.4893563985824585],
# [0.9210755825042725, 0.9204531311988831, 0.9221959710121155, 0.9106187224388123, 0.8958048224449158, 0.8687912225723267, 0.8170048594474792, 0.77467942237854, 0.7175401449203491, 0.6592804789543152, 0.6218100190162659, 0.5904394388198853, 0.5654176473617554, 0.5464956760406494, 0.5176148414611816, 0.5135067701339722, 0.5055396556854248, 0.48736461997032166, 0.48724013566970825],
# [0.9778413772583008, 0.9783393740653992, 0.9758496284484863, 0.9747292399406433, 0.9698742628097534, 0.958421528339386, 0.9332752227783203, 0.8864682912826538, 0.8308228850364685, 0.78302001953125, 0.7291173934936523, 0.6834308505058289, 0.647703230381012, 0.6285322904586792, 0.6012697815895081, 0.5936760902404785, 0.5806050300598145, 0.5773683786392212, 0.5625544786453247],
# [0.9632765054702759, 0.9637744426727295, 0.9640234112739563, 0.9660151600837708, 0.9627785682678223, 0.9589194655418396, 0.9504543542861938, 0.9261795282363892, 0.8824847340583801, 0.8406572937965393, 0.784015953540802, 0.7380804419517517, 0.6950080990791321, 0.6699863076210022, 0.6561682820320129, 0.6389891505241394, 0.6196937561035156, 0.6103572845458984, 0.6018921732902527],
# [0.9629030227661133, 0.9624050855636597, 0.9616581797599792, 0.9645213484764099, 0.9662641882896423, 0.9625295400619507, 0.960911214351654, 0.9498319625854492, 0.9317814111709595, 0.8966761827468872, 0.86368727684021, 0.8186231851577759, 0.7790364623069763, 0.7525208592414856, 0.7251338362693787, 0.7114403247833252, 0.6887837648391724, 0.6765840649604797, 0.6560438275337219],
# [0.9604132771492004, 0.9610357284545898, 0.9590439200401306, 0.9601643085479736, 0.9589194655418396, 0.9617826342582703, 0.961907148361206, 0.9577990770339966, 0.948960542678833, 0.926428496837616, 0.8982945084571838, 0.8562180995941162, 0.8216108679771423, 0.7973359823226929, 0.772314190864563, 0.7400721907615662, 0.7241379022598267, 0.7099464535713196, 0.6860450506210327],
# [0.9558072686195374, 0.9564297199249268, 0.9570521712303162, 0.9569276571273804, 0.9565542340278625, 0.9568032026290894, 0.9559317827224731, 0.958421528339386, 0.950080931186676, 0.9391261339187622, 0.9246856570243835, 0.8946844339370728, 0.8640607595443726, 0.827835202217102, 0.805427610874176, 0.7740570306777954, 0.7586206793785095, 0.7360886335372925, 0.7187849879264832],
# [0.9570521712303162, 0.9565542340278625, 0.9561807513237, 0.9570521712303162, 0.9591684341430664, 0.9564297199249268, 0.957301139831543, 0.957301139831543, 0.9591684341430664, 0.9561807513237, 0.9448524713516235, 0.926552951335907, 0.8984190225601196, 0.8699116110801697, 0.8410307765007019, 0.8084152936935425, 0.7800323367118835, 0.7575002908706665, 0.7368355393409729],
# [0.8949334025382996, 0.8930661082267761, 0.8920702338218689, 0.8928171396255493, 0.8940619826316833, 0.9009087681770325, 0.905141294002533, 0.9046433568000793, 0.9068840742111206, 0.9126104712486267, 0.9114900827407837, 0.9020291566848755, 0.8935640454292297, 0.8654301166534424, 0.8441429138183594, 0.8096601366996765, 0.7894933223724365, 0.7655919194221497, 0.7424374222755432],
# [0.9432341456413269, 0.9429851770401001, 0.9432341456413269, 0.9416158199310303, 0.9362629055976868, 0.9396240711212158, 0.9396240711212158, 0.9386281371116638, 0.940619945526123, 0.9402464628219604, 0.9386281371116638, 0.9335241913795471, 0.926552951335907, 0.9078800082206726, 0.88733971118927, 0.8621934652328491, 0.8401593565940857, 0.8066724538803101, 0.7811527252197266],
# [0.9136063456535339, 0.9149757027626038, 0.9185858368873596, 0.91584712266922, 0.9178389310836792, 0.9198306798934937, 0.9219469428062439, 0.9233162999153137, 0.9285447597503662, 0.9240632653236389, 0.9259305596351624, 0.926428496837616, 0.9229428768157959, 0.9131084084510803, 0.8961782455444336, 0.8793725967407227, 0.8528569936752319, 0.8247230052947998, 0.7944728136062622],
# [0.780654788017273, 0.784140408039093, 0.7870036363601685, 0.7906137108802795, 0.7988298535346985, 0.7949707508087158, 0.8025644421577454, 0.8117763996124268, 0.8156355023384094, 0.8130213022232056, 0.8156355023384094, 0.8116519451141357, 0.816133439540863, 0.8048051595687866, 0.7865056395530701, 0.7755508422851562, 0.752022922039032, 0.7256317734718323, 0.7063363790512085],
# [0.8809909224510193, 0.8793725967407227, 0.8752645254135132, 0.868293285369873, 0.8621934652328491, 0.8514876365661621, 0.8432714939117432, 0.8412797451019287, 0.8334370851516724, 0.8326901793479919, 0.837420642375946, 0.8298269510269165, 0.8334370851516724, 0.8300759196281433, 0.823229193687439, 0.8204904794692993, 0.7987053394317627, 0.782895565032959, 0.771069347858429],
# [0.9401220083236694, 0.9396240711212158, 0.9376322627067566, 0.9386281371116638, 0.9387526512145996, 0.9421137571334839, 0.9402464628219604, 0.9414913654327393, 0.9398730397224426, 0.9388771057128906, 0.9404954314231873, 0.9403709769248962, 0.9409934282302856, 0.940619945526123, 0.9423627257347107, 0.9347690939903259, 0.9295406341552734, 0.91933274269104, 0.9017801284790039],
# [0.9306610226631165, 0.9321548342704773, 0.930038571357727, 0.9341466426849365, 0.9338976740837097, 0.9325283169746399, 0.9321548342704773, 0.9335241913795471, 0.9305365085601807, 0.9351425170898438, 0.9362629055976868, 0.9329017996788025, 0.9356404542922974, 0.9376322627067566, 0.9333997368812561, 0.9316568970680237, 0.9311589598655701, 0.9244366884231567, 0.9223204255104065],
# [0.9122370481491089, 0.9131084084510803, 0.9109921455383301, 0.9141043424606323, 0.9133573770523071, 0.9187102913856506, 0.9189593195915222, 0.9179633855819702, 0.9245612025260925, 0.9218224883079529, 0.9213245511054993, 0.9194572567939758, 0.9216979742050171, 0.916967511177063, 0.9153491854667664, 0.9184613227844238, 0.91584712266922, 0.9202041625976562, 0.9179633855819702],
# [0.9083779454231262, 0.9078800082206726, 0.907382071018219, 0.9065106511116028, 0.9072575569152832, 0.9082534313201904, 0.9022781252861023, 0.905141294002533, 0.9040209054946899, 0.9045188426971436, 0.9055147767066956, 0.9032739996910095, 0.9081289768218994, 0.9080044627189636, 0.9082534313201904, 0.9102452397346497, 0.9096227884292603, 0.9112411141395569, 0.9129839539527893]])

# fig, ax = plt.subplots()
# c = ax.pcolormesh(rotation_steps, rotation_steps, accuracy_matrix.T, cmap='RdBu')
# ax.set_title("Accuracy on rotated images, depending on rotation of the training dataset (CNN)")
# plt.xlabel("Rotation max of the training dataset")
# plt.ylabel("Rotation max of the testing sample")
# plt.show()




#list of all the training data set rotation values
# training_database= rotate_database(rx_train, 0, rotation_steps[1])



#loading the models from memory

# cnn = CNN_model()
# cnn.load_weights("cnn_weights")

# super_cnn = CNN_model()
# super_cnn.load_weights("super_cnn_weights")


# # model_list = [pca_model, dnn_model, cnn_model]

# #BOILERPLATE code for generating and plotting the effect of an attack
# n_samples = 8
# attack_function = uniform_noise_database

# arguments = np.array(  [[i/10] for i in range(8)] )
# # arguments[:, 0] = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5]
# # arguments[:, 1] = arguments[:, 0]

# #x_axis = arguments[:, 0]
# x_axis = [i/10 for i in range(8)]

# model_list = [cnn, super_cnn]
# x_axis = [i/20 for i in range(30)]


# # result = run_attacks(x_test, y_test_cat, model_list, attack_function, arguments)


# plt.plot(x_axis, result[0], label="CNN Model")
# plt.plot(x_axis, result[1], label="Super CNN Model")
# plt.legend(loc="upper right")
# plt.xlabel("Noise in Test Database")
# plt.ylabel("Accuracy of models")



# plt.show()




# pca_model = PCA_model.load("pca125_weights")
# # pca_model.fit(x_train, y_train_cat)
# # pca_model.save("pca125_weights")

# dnn_model = PCA_model(784, 10, True, False)
# dnn_model.fit(x_train, y_train_cat)

# cnn_model = CNN_model()
# # cnn_model.fit(x_train, y_train_cat)
# cnn_model.load_weights("cnn_weights_3_epochs")

# epsilons = np.linspace(0, 0.5, 20)
# pca_accs = np.zeros(20)
# cnn_accs = np.zeros(20)
# dnn_accs = np.zeros(20)

# for i, eps in np.ndenumerate(epsilons):
#     # fgsm_x = fast_gradient_method(cnn_model, x_test.reshape(-1, 28, 28, 1), eps, np.inf, y=y_test.astype(int)).numpy()
#     fgsm_x = constant_noise_database(x_test, eps)
#     pca_accs[i] = pca_model.evaluate(fgsm_x, y_test_cat)[1]
#     cnn_accs[i] = cnn_model.evaluate(fgsm_x, y_test_cat)[1]
#     dnn_accs[i] = dnn_model.evaluate(fgsm_x, y_test_cat)[1]
    
# plt.plot(epsilons, pca_accs, label="PCA model accuracy")
# plt.plot(epsilons, cnn_accs, label="CNN model accuracy")
# plt.plot(epsilons, dnn_accs, label="DNN model accuracy")
# plt.legend()
# plt.xlabel("Perturbation (epsilon) of each pixel")
# plt.ylabel("Model accuracy")
# plt.title("Model Accuracies on Images with Randomly Perturbed (fixed epsilon) Pixels")
# plt.show()





# fgsm_x = fast_gradient_method(cnn_model, x_test.reshape(-1, 28, 28, 1), 0.1, np.inf, y=y_test.astype(int)).numpy()
# print(fgsm_x.shape)




# fgsm_x = fast_gradient_method(cnn, x_test.reshape(-1, 28, 28, 1), 0.07, np.inf).numpy()
# print(cnn.evaluate(fgsm_x, y_test_cat))

# fig, ax = plt.subplots(5, 5)
# eps = np.linspace(0, 0.5, 25)

# for i in range(5):
#     for j in range(5):
#         fgsm_x = fast_gradient_method(cnn, x_test[0].reshape(-1, 28, 28, 1), eps[i * 5 + j], np.inf, y=np.array([y_test[0]]).astype(int)).numpy().reshape(28, 28)
#         ax[i][j].imshow(fgsm_x)
#         ax[i][j].axis('off')

# plt.show()



# print(cnn.evaluate(x_test, y_test_cat))
# super_cnn = CNN_model()
# super_cnn.load_weights("super_cnn_weights")


# # model_list = [pca_model, dnn_model, cnn_model]

# #BOILERPLATE code for generating and plotting the effect of an attack
# n_samples = 8
# attack_function = uniform_noise_database

# arguments = np.array(  [[i/10] for i in range(8)] )
# # arguments[:, 0] = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5]
# # arguments[:, 1] = arguments[:, 0]

# #x_axis = arguments[:, 0]
# x_axis = [i/10 for i in range(8)]

# model_list = [cnn, super_cnn]
# x_axis = [i/20 for i in range(30)]


# # result = run_attacks(x_test, y_test_cat, model_list, attack_function, arguments)


# plt.plot(x_axis, result[0], label="CNN Model")
# plt.plot(x_axis, result[1], label="Super CNN Model")
# plt.legend(loc="upper right")
# plt.xlabel("Noise in Test Database")
# plt.ylabel("Accuracy of models")



# plt.show()
# lattice = attack_lattice(CNN_model, (rx_train, ry_train_cat), (rx_test, ry_test_cat), 
#                          gaussian_blur_database, rotate_database, np.arange(0, 2, 0.1), np.arange(0, 180, 10))
# np.savetxt("cnn_gaussian_against_rotation.txt", lattice)
# lattice = np.loadtxt("cnn_gaussian_against_rotation.txt")
# plt.pcolormesh(np.arange(0, 180, 10), np.arange(0, 2, 0.1), lattice, cmap="RdBu")
# plt.xlabel("Training Data Max Rotation Angle")
# plt.ylabel("Testing Data Max Blur Level")
# plt.title("Gaussian Blur against Rotation (CNN model)")
# plt.show()

cnn = CNN_model()
cnn.load_weights("cnn_weights_3_epochs")
# images = fgsm_database_cnn(cnn, x_test[:5], y_test[:5], 0.1)

# plt.imshow(images[3])
# plt.show()

# lattice = attack_lattice_fgsm_cnn((x_train, y_train, y_train_cat), (x_test, y_test, y_test_cat), np.linspace(0, 0.5, 20))
# np.savetxt("FGSM_cnn_lattice.txt", lattice)
# epsilons = np.linspace(0, 0.5, 20)
# average_confidence_original = np.ones(20)
# average_confidences_fgsm = np.zeros(20)
# targets = np.random.randint(0, 10, y_test.shape[0])

# average_confidence_original *= compute_average_confidence_over_true_answer(cnn, x_test, targets)
# for i, eps in np.ndenumerate(epsilons):
#     x_fgsm = fast_gradient_method(cnn, x_test.reshape(-1, 28, 28, 1), eps, 
#                                 np.inf, y=targets, targeted=True).numpy()
#     average_confidences_fgsm[i] = compute_average_confidence_over_true_answer(cnn, x_fgsm, targets)
    
# np.savetxt("average_confs.txt", average_confidences_fgsm)
# plt.plot(epsilons, average_confidences_fgsm, label="FGSM generated images")
# plt.plot(epsilons, average_confidence_original, label="Original images")
# plt.xlabel("Epsilon used to Perturbe each Pixel")
# plt.ylabel("Average Confidence in Targets")
# plt.title("Demonstration of Targeted FGSM")
# plt.legend()
# plt.show()