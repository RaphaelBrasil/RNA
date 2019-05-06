import numpy as np
import pandas
from sklearn import model_selection
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC


def linhas(x):  # Retorna a quantidade de linhas em uma matriz
    return x.shape[0]


def colunas(x):  # Retorna a quantidade de colunas em ums matriz
    return x.shape[1]


def calc_erro(t_iteration):
    return Y_treino[t_iteration] - predict(t_iteration, 1)  # O u[0] é chamado para pegar o elemento dentro do array


def calc_erro_teste(t_iteration):
    return Y_teste[t_iteration] - predict(t_iteration, 0)


def degrau(t_iteration, treino):
    if treino == 1:
        return X_treino[t_iteration].dot(w_pesos_sinapticos)
    else:
        return X_teste[t_iteration].dot(w_pesos_sinapticos)


def predict(t_iteration, treino):
    return np.where(degrau(t_iteration, treino) >= 0.0, 1, 0)


def selecao_base(tipo, Y):
    if tipo == 0: # Setosa vs Outras
        Y[Y == 'Iris-setosa'] = 1
        Y[Y == 'Iris-virginica'] = 0
        Y[Y == 'Iris-versicolor'] = 0
        return Y
    if tipo == 1:# Virginica vs Outras
        Y[Y == 'Iris-setosa'] = 0
        Y[Y == 'Iris-virginica'] = 1
        Y[Y == 'Iris-versicolor'] = 0
        return Y
    if tipo == 2:# Versicolor vs Outras
        Y[Y == 'Iris-setosa'] = 0
        Y[Y == 'Iris-virginica'] = 0
        Y[Y == 'Iris-versicolor'] = 1
        return Y




# Carregar os dados
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)




# Embaralha e divide os dados em treinamento e teste
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
X_treino, X_teste, Y_treino, Y_teste = model_selection.train_test_split(X, Y, test_size=validation_size)


tipo = 2 # Define qual classe será vs Outras


# Adiciona os valores de X0
X_treino = np.insert(X_treino, 0, -1, axis = 1)

X_teste = np.insert(X_teste, 0, -1, axis = 1)


Y_treino = selecao_base(tipo, Y_treino).T # Troca os nomes em string para 0 ou 1

Y_teste = selecao_base(tipo, Y_teste).T # Troca os nomes em string para 0 ou 1

n_taxa_de_aprendizado = 0.1

qt_epocas = 20

qt_realizacoes = 20

w_pesos_sinapticos = 2 * np.random.random((colunas(X_treino), 1)) - 1



print('Pesos sinápticos iniciais randômicos: ')
print(linhas(X_treino)) # Retorna a quantidade de elementos em um array de arrays
print(w_pesos_sinapticos)

for realizacoes in range(qt_realizacoes):

    for t_epoca in range(qt_epocas):

        qt_erros = 0

        for t_iteration in range(linhas(X_treino)):

            u = degrau(t_iteration, 1)  # Faz o somatório de WiXi (Σ)

            erro = calc_erro(t_iteration)  # Faz o calculo de D - Y (A função de ativação é chamada aqui (Predict))

            for i in range(len(w_pesos_sinapticos)):
                w_pesos_sinapticos[i] = [w_pesos_sinapticos[i] + n_taxa_de_aprendizado * erro * X_treino[t_iteration, i]] # Função de aprendizagem para cada Wi

            if erro != 0:

                qt_erros = qt_erros + 1

        if qt_erros == 0:
            print('Saiu com erro == 0!')
            print('Na epóca: ')
            print(t_epoca)
            break

print('Pesos Ajustados: ')
print(w_pesos_sinapticos)


#TESTE

acerto = 0
predicao = np.array(-1)  # Somente para iniciar a variavel, depois o -1 é excluido

for t_iteration in range(linhas(X_teste)):

    # u = classificador(t_iteration, 0)  # Faz o somatório de WiXi (Σ) Predict

    predicao = np.append(predicao, predict(t_iteration, 0))


predicao = np.delete(predicao, 0)
print('Predição: ')
print(predicao)
print('Desejado: ')
print(Y_teste)

# Cálculo da acurácia
acuracia = 0
for interacao in range(Y_teste.size):
    if predicao[interacao] == Y_teste[interacao]:
        acuracia = acuracia + 1

# Matriz de confusão
matriz = np.zeros(shape=(2, 2))
for interacao in range(Y_teste.size):
    if predicao[interacao] == Y_teste[interacao]:
        if predicao[interacao] == 0:
            matriz[1, 1] += 1
        else:
            matriz[0, 0] += 1
    if predicao[interacao] != Y_teste[interacao]:
        if predicao[interacao] == 0:
            matriz[1, 0] += 1
        else:
            matriz[0, 1] += 1


print('--------------------------')
print('Acurácia: ')
print(acuracia/Y_teste.size)

print('Matriz de confusão: ')
print(matriz)

print('--------------------------')


