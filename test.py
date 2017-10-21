import keras

# This returns a tensor
input_list = []

for i in range(0,15):
    input_list.append(keras.layers.Input(shape=(23,)))

hidden_layer1_list = []

for i in range(2,13):
    hidden_layer1_list.append(keras.layers.Dense(512,activation='relu')(keras.layers.concatenate( [input_list[i-2], input_list[i-1], input_list[i], input_list[i+1], input_list[i+2]])))

hidden_layer2_list = []
hidden_layer2_list.append(keras.layers.Dense(512,activation='relu')(keras.layers.concatenate([hidden_layer1_list[0],hidden_layer1_list[2],hidden_layer1_list[4]])))
hidden_layer2_list.append(keras.layers.Dense(512,activation='relu')(keras.layers.concatenate([hidden_layer1_list[3],hidden_layer1_list[5],hidden_layer1_list[7]])))
hidden_layer2_list.append(keras.layers.Dense(512,activation='relu')(keras.layers.concatenate([hidden_layer1_list[6],hidden_layer1_list[8],hidden_layer1_list[10]])))


hidden_layer3 = keras.layers.Dense(512,activation='relu')(keras.layers.concatenate([hidden_layer2_list[0],hidden_layer2_list[1],hidden_layer2_list[2]]))
hidden_layer4 = keras.layers.Dense(512, activation='relu')(hidden_layer3)
hidden_layer5 = keras.layers.Dense(512, activation='softmax')(hidden_layer4)

# predictions = keras.layers.Dense(1, activation='softmax')(x)

# This creates a model that includes
# the Input layer and three Dense layers
model = keras.models.Model(inputs=input_list, outputs=hidden_layer5)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# model.fit(data, labels)  # starts training
print(model.summary())
keras.utils.plot_model(model, to_file='model.png')
print("done")

