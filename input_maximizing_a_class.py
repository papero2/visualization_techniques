## FIND IDEAL INPUT
import numpy as np
import matplotlib.pyplot as plt
import keras
import skimage
import skimage.viewer
from keras import backend as K
from keras import layers

# USER OPTIONS
model_name="mnist.model"
iterations=20

# VARIABLES
model = keras.models.load_model(model_name)
print(model.summary())
input_shape=model.input.shape[1]
nc=model.output_shape[1]
dense2_name=model.layers[7].name
final_output=np.zeros((input_shape*nc, input_shape*iterations, model.input.shape[3]))

def chollet():
    B=28
    for k in range(nc):
        step=1
        X=np.random.random((1, input_shape, input_shape, model.input.shape[3]))*0.1+0.5
        ideal=0
        objective = model.output[0,k]
        c=K.gradients(objective, model.input)[0]
        c /= (K.sqrt(K.mean(K.square(c))) + 1e-5)
        get= K.function([model.input, K.learning_phase()],[objective, c])
        get2=K.function([model.input, K.learning_phase()],[model.layers[7].output[0,k]])
        A=0
        
        for j in range(iterations):
            loss_value, grads_value=get([X, 1])
            final_output[input_shape*k:input_shape*(k+1),A:A+B]=X[0]
            X += grads_value*step
            A+=B
            ideal= get2([X, 1])
            print("score ", ideal)
        out=model.predict(X)
        print(out[0])
    plt.imshow(final_output[:,:,0], vmin=0, vmax=1)
    plt.show()
    
chollet()

