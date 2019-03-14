import numpy as np
from sklearn.ensemble import RandomForestRegressor
from numpy import genfromtxt
import cv2
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Extrai histograma dado um conjunto de imagens salvando ao final o valor de irradiancia total')
parser.add_argument('arquivo_dados',help='inserir o nome de arquivo de dados, na pasta "Dados_ImagemPiranometro", o qual contem nome da imagem e valor dos respectivos sensores')
parser.add_argument('pasta_de_imagens',type=str,help='indicar a pasta externa aonde se encontra as imagens listadas no "arquivo_dados"')
parser.add_argument('nome_arquivo_saida',type=str,help='inserir o nome do arquivo, o qual contem histograma + valor de sensores, que sera salvo na pasta "Base_de_dados" ')
args = parser.parse_args()



def extrairHistograma(nome_arquivo,pasta_destino,dados):
    new_data = []
    for i in range(0,len(dados)):
        img = cv2.imread(str(pasta_destino)+dados[i][0])
        print(i)
        ##extrai informacoes correspodentes a histograma vermelho, verde e azul e faz a concatenacao dos vetores
        hist_B = cv2.calcHist([img],[0],None,[256],[0,256])
        hist_G = cv2.calcHist([img], [1], None, [256], [0, 256])
        hist_R = cv2.calcHist([img], [2], None, [256], [0, 256])
        rgb =np.asarray(np.concatenate((hist_B,hist_G,hist_R)))
        data = np.empty(771,dtype=object)

        ##salva os dados conforme dia/hora/RGB/valor_piranometro
        data[0]=dados[i][0]
        data[1]=dados[i][1]
        data[2:770] = rgb[:,0]
        data[770] = dados[i][3]
        ##----------------------------
        new_data.append(data)

    new_data = np.array(new_data)

    np.savetxt('Base_de_dados/'+nome_arquivo+'.txt', new_data, delimiter=" ", fmt="%s")


#retorna dados filtrados: imagem com seu valor de piranômetro contidos na pasta Dados_ImagemPiranometro
def ler_dados(name):
    dados_entrada = genfromtxt('Dados_ImagemPiranometro/'+name,delimiter=";",dtype='unicode')
    ##retira cabecalho do arquivo dos dados
    dados_entrada = dados_entrada[1:]

    return dados_entrada

##passar no args
if __name__ == '__main__':
    dados = ler_dados(args.arquivo_dados)
    extrairHistograma(args.nome_arquivo_saida,args.pasta_de_imagens,dados)




