import tensorflow as tf
from sklearn.metrics import confusion_matrix
import numpy as np
from numpy import genfromtxt
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

from tensorflow.keras.models import model_from_yaml
import argparse
import os

parser = argparse.ArgumentParser(description='Treina o classificador conforme os dados extraidos das imagens')
parser.add_argument('modelo',help='Indicar nome do modelo treinado')
parser.add_argument('pesos',help='Indicar nome do arquivo de pesos do modelo treinado')
parser.add_argument('arquivo',help='Indicar nome do arquivo com os dados dos dias a serem avaliados')
args = parser.parse_args()



def carregar_dados(nome_arquivo):
    datain = genfromtxt(nome_arquivo, delimiter=' ')  # ,dtype='unicode')

    entrada = datain[:, 3:len(datain[0]) - 2]
    entrada = np.array(entrada,dtype=float)
    entrada = normalize(entrada)

    ##retira ultima coluna dos dados correspondentes ao valor da irradiacao
    saida = datain[:, len(datain[0]) - 1]

    print(saida[10])

    ##extrai outliers (valores maiores que 1000)
    nova_entrada = []
    nova_saida = []

    for k in range(0, len(datain)):
        # if k % 7 and dataout[k]<1000:
        if saida[k] < 1000:
            nova_entrada.append(entrada[k])
            nova_saida.append(saida[k])



    x_test = np.asarray(nova_entrada)
    y_test = np.asarray(nova_saida)

    return x_test,y_test


def mostrar_resultado(predito,real):
    # #


    plt.subplot(2,1,1)
    plt.title('Result')
    plt.ylabel('total irradiation')
    plt.xlabel('sample')
    plt.legend(['predicted', 'real'], loc='upper left')
    error =  np.abs(predito[:,0]-real)
    plt.plot(np.arange(len(error)),predito, 'r')
    plt.plot(np.arange(len(error)), real, 'g')

    plt.subplot(2, 1, 2)
    plt.title('Result')
    plt.ylabel('total irradiation')
    plt.xlabel('sample')
    plt.legend(['predicted', 'real'], loc='upper left')
    error = np.abs(predito[:, 0] - real)
    plt.plot(np.arange(len(error)), error, 'b')
    #plt.plot(np.arange(len(predicted)), predicted, 'r',alpha=0.5)
    # axes[1, 1].set_ylabel('total irradiation')
    # axes[1, 1].set_xlabel('sample')
    # axes[1, 1].legend(['predicted', 'real'], loc='upper left')
    # axes[1, 1].plot(np.arange(len(real)), real, 'g')
    # axes[1, 1].plot(np.arange(len(predicted)), predicted, 'r')
    plt.show()


def carregar_modelo(modelo, pesos,entrada,saida):
    # carrega o modelo de rede em formato .yaml
    # yaml_file = open(modelo, 'r')
    # loaded_model_yaml = yaml_file.read()
    # yaml_file.close()
    # loaded_model = model_from_yaml(loaded_model_yaml)
    # # carrega os pesos do modelo treinado em .h5
    # loaded_model.load_weights(pesos)
    # print("Loaded model from disk")
    #
    # # evaluate loaded model on test data
    # loaded_model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
    # predito = loaded_model.

    with open(modelo) as ff:
        model_yaml = ff.read()
        model = model_from_yaml(model_yaml)
    model.compile(loss='mse',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    model.load_weights(pesos)
    predito = model.predict(entrada)
    mostrar_resultado(predito,saida)




if __name__ == '__main__':
    x_teste,y_teste = carregar_dados(args.arquivo)

    print("Test Samples: " + str(len(y_teste)))
    carregar_modelo(args.modelo,args.pesos,x_teste,y_teste)



