import argparse
import time
import numpy as np
import os



parser = argparse.ArgumentParser(description='Cruza as imagens de uma pasta com os dados do piranometro, indexando a cada imagem o valor do piranometro')
parser.add_argument('dado_piranometro',help='Indicar o nome do arquivo desejado na pasta "Dados_Piranometro"')
parser.add_argument('pasta_de_imagens',type=str,help='Indicar a pasta externa aonde se encontra as imagens a serem cruzadas com os valores do piranometro')
parser.add_argument('nome_arquivo_saida',type=str,help='inserir o nome do arquivo, o qual contem histograma + valor de sensores, que sera salvo na pasta "Base_de_dados" ')
args = parser.parse_args()



def listar_imagensar(arquivo_imagens):
    image_folder = arquivo_imagens
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    print(len(images))
    images = np.sort(images)
    return images



def carregar_dados_piranometro(dado_piranometro):
    data = np.genfromtxt(dado_piranometro, dtype='unicode', delimiter=';')
    return data


def filtrar_ImagemPiranometro(nome_arquivo,dados_pir,images) :


    datafile = open('Dados_ImagemPiranometro/' + nome_arquivo + ".csv", 'w')
    datafile.write(
        "Img" + ";" + "Dia" + ";" + "Hora" + ";" + "Pir" + ";" + "Sar" + ";" + "Tar" + ";" + "Tex" + ";" + "Taq" + ";" + "Urt")
    datafile.write("\n")

    counter = 1
    check = 0
    #
    x = 1
    for img in images:
        substringhora = img[11:13] + "_" + img[14:16] + "_" + img[17:19]
        substringdata = img[8:10] + "_" + img[5:7] + "_" + img[0:4]
        date_image = time.strptime(substringdata, "%d_%m_%Y")
        hour_image = time.strptime(substringhora, "%H_%M_%S")
        print(img)

        for i in range(x, len(dados_pir)):
            # print(data[i])
            # print(i)
            if ((time.strptime(dados_pir[i][1], "%H_%M_%S") == hour_image) and (
                time.strptime(dados_pir[i][0], "%d_%m_%Y") == date_image)):
                datafile.write(
                    img + ";" + substringdata + ";" + substringhora + ";" + dados_pir[i][5] + ";" + dados_pir[i][6] + ";" +
                    dados_pir[i][7] + ";" + dados_pir[i][8] + ";" + dados_pir[i][9] + ";" + dados_pir[i][10])
                datafile.write("\n")
                x = i + 10
                print(img)
                break;


    datafile.close()


if __name__ == '__main__':
    if not os.path.exists("Dados_ImagemPiranometro"):
        os.makedirs("Dados_ImagemPiranometro")

    imagens = listar_imagensar(args.pasta_de_imagens)
    dados_pir = carregar_dados_piranometro(args.dado_piranometro)
    filtrar_ImagemPiranometro(args.nome_arquivo_saida,dados_pir,imagens)