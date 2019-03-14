import tensorflow as tf
from sklearn.metrics import confusion_matrix
import numpy as np
from numpy import genfromtxt
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Treina o classificador conforme os dados extraidos das imagens')
parser.add_argument('base_dados',help='Indicar nome do arquivo que possui os dados')
args = parser.parse_args()



def carregar_dados(nome_arquivo):
    datain = genfromtxt(nome_arquivo + '.txt', delimiter=' ')  # ,dtype='unicode')

    entrada = datain[:, 2:len(datain[0]) - 2]
    entrada = np.array(entrada,dtype=float)
    entrada = normalize(entrada)

    ##retira ultima coluna dos dados correspondentes ao valor da irradiacao
    saida = datain[:, len(datain[0]) - 1]



    ##extrai outliers (valores maiores que 1000)
    nova_entrada = []
    nova_saida = []

    for k in range(0, len(datain)):
        # if k % 7 and dataout[k]<1000:
        if saida[k] < 1000:
            nova_entrada.append(entrada[k])
            nova_saida.append(saida[k])


    ##separa os dados para o classificador em conjuntos de treinamento e conjuntos de teste de acordo com a porcentagem definida (test_size)
    x_train, x_test, y_train, y_test = train_test_split(nova_entrada, nova_saida, test_size=0.3, random_state=14)
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)

    return x_train,x_test,y_train,y_test


def mostrar_resultado(history,predito,real):
    # #
    # ##graph training stage
    # #axes[0, 0].set_title('')
    # axes[0, 0].set_ylabel('loss')
    # axes[0, 0].set_xlabel('epoch')
    # axes[0, 0].legend(['train', 'test'], loc='upper left')
    # axes[0, 0].plot(history.history['loss'])
    # axes[0, 0].plot(history.history['val_loss'])
    # ##prediction vs real
    # axes[1, 1].set_title('Result')
    # axes[1, 1].set_ylabel('total irradiation')
    # axes[1, 1].set_xlabel('sample')
    # axes[1, 1].legend(['predicted', 'real'], loc='upper left')
    # axes[1, 1].plot(np.arange(len(real)), real, 'g')
    # axes[1, 1].plot(np.arange(len(predicted)), predicted, 'r')
    # plt.show()
    plt.subplot(2, 1, 1)
    plt.title('')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])

    plt.subplot(2,1,2)
    plt.title('Result')
    plt.ylabel('total irradiation')
    plt.xlabel('sample')
    plt.legend(['predicted', 'real'], loc='upper left')
    error =  np.abs(predito[:,0]-real)
    plt.plot(np.arange(len(error)),error, 'g')
    #plt.plot(np.arange(len(predicted)), predicted, 'r',alpha=0.5)
    # axes[1, 1].set_ylabel('total irradiation')
    # axes[1, 1].set_xlabel('sample')
    # axes[1, 1].legend(['predicted', 'real'], loc='upper left')
    # axes[1, 1].plot(np.arange(len(real)), real, 'g')
    # axes[1, 1].plot(np.arange(len(predicted)), predicted, 'r')
    plt.show()

def build_model(camadas,formato_entrada):
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(camadas, activation=tf.nn.relu,
                       input_shape=(formato_entrada,)),
    tf.keras.layers.Dense(camadas, activation=tf.nn.relu),
    tf.keras.layers.Dense(camadas, activation=tf.nn.relu),
    tf.keras.layers.Dense(camadas, activation=tf.nn.relu),
    tf.keras.layers.Dense(camadas, activation=tf.nn.relu),
    tf.keras.layers.Dense(camadas, activation=tf.nn.relu),
    #tf.keras.layers.Dense(camadas, activation=tf.nn.relu),
    # tf.keras.layers.Dense(camadas, activation=tf.nn.relu),
    #tf.keras.layers.Dense(camadas, activation=tf.nn.relu),
    #tf.keras.layers.Dense(camadas, activation=tf.nn.relu),
    tf.keras.layers.Dense(1)
  ])

  optimizer = tf.train.RMSPropOptimizer(0.0003)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mse','mae'])
  return model




if __name__ == '__main__':
    x_treino,x_teste,y_treino,y_teste = carregar_dados(args.base_dados)
    modelo = build_model(camadas=110,formato_entrada=x_treino.shape[1])
    print("Test Samples: " + str(len(y_teste)))

    history = modelo.fit(x_treino, y_treino, validation_split=0.3, epochs=200, shuffle=True)
    resultado = modelo.predict(x_teste)
    [loss, mse, mae] = modelo.evaluate(x_teste, y_teste, verbose=0)
    mostrar_resultado(history, resultado, y_teste)

