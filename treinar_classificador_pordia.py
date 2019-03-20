import tensorflow as tf
from sklearn.metrics import confusion_matrix
import numpy as np
from numpy import genfromtxt
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from tensorflow.keras.models import model_from_yaml
import argparse
import os

parser = argparse.ArgumentParser(description='Treina o classificador conforme os dados extraidos das imagens')
parser.add_argument('base_dados',help='Indicar nome do arquivo que possui os dados')
parser.add_argument('nome_modelo',help='Indicar nome do modelo que sera salvo apos o treinamento')
parser.add_argument('data_teste',help='Indicar a data a ser avaliada pelo classificador apos o treinamento')
args = parser.parse_args()



def carregar_dados(nome_arquivo ,data_teste):

    datain = genfromtxt(nome_arquivo + '.txt', delimiter=' ')  # ,dtype='unicode')
    datain2 = genfromtxt(nome_arquivo + '.txt', delimiter=' ' ,dtype='unicode', usecols=(2))


    entrada = datain[:, 3:len(datain[0]) - 2]
    entrada = np.array(entrada,dtype=float)
    entrada = normalize(entrada)

    ##retira ultima coluna dos dados correspondentes ao valor da irradiacao
    saida = datain[:, len(datain[0]) - 1]

    print(saida[10])

    ##extrai outliers (valores maiores que 1000)
    nova_entrada = []
    nova_saida = []
    teste_entrada =  []
    teste_saida = []
    for k in range(0, len(datain)):
        ##retira outlier e separa o dia de teste dos dias de treinamento
        if saida[k] < 1000 and (str(datain2[k])!=str(data_teste)):
            nova_entrada.append(entrada[k])
            nova_saida.append(saida[k])
        else:
            teste_entrada.append(entrada[k])
            teste_saida.append(saida[k])


    ##separa os dados para o classificador em conjuntos de treinamento e conjuntos de teste de acordo com a porcentagem definida (test_size)
    x_train, x_test, y_train, y_test = train_test_split(nova_entrada, nova_saida, test_size=0.3, random_state=14)


    ##usa os demais dias para treinar o modelo
    x_train = np.asarray(nova_entrada)
    y_train = np.asarray(nova_saida)

    ##insere os dados do dia selecionado para teste
    x_test = np.asarray(teste_entrada)
    y_test = np.asarray(teste_saida)



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
    plt.plot(np.arange(len(error)),predito[:,0], 'g')
    plt.plot(np.arange(len(error)), real, 'y')
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



def treinar_modelo(modelo, x_treino, y_treino,nome_modelo):
    history = modelo.fit(x_treino, y_treino, validation_split=0.3, epochs=200, shuffle=True)
    resultado = modelo.predict(x_teste)
    [loss, mse, mae] = modelo.evaluate(x_teste, y_teste, verbose=0)
    mostrar_resultado(history, resultado, y_teste)
    # transformar o objeto modelo para codificacao YAML
    model_yaml = modelo.to_yaml()
    with open("Redes_Treinadas/"+nome_modelo + ".yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)
    # serialize weights to HDF5
    modelo.save_weights("Redes_Treinadas/"+nome_modelo+".h5")
    print("Modelo salvo com sucesso!")


if __name__ == '__main__':
    ##verificar se existe a pasta "Modelo" para Salvar modelos treinados
    if not os.path.exists("Redes_Treinadas"):
        os.makedirs("Redes_Treinadas")


    x_treino,x_teste,y_treino,y_teste = carregar_dados(args.base_dados, args.data_teste)
    modelo = build_model(camadas=110,formato_entrada=x_treino.shape[1])
    print("Train Samples: " + str(len(y_treino)))
    print("Test Samples: " + str(len(y_teste)))
    treinar_modelo(modelo,x_treino, y_treino, args.nome_modelo)



