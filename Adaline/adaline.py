import random
import numpy as np
from sklearn import model_selection
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D as a3d


def linhas(x):  # Retorna a quantidade de linhas em uma matriz
    return x.shape[0]


def colunas(x):  # Retorna a quantidade de colunas em ums matriz
    return x.shape[1]


def createWeights(length):
    w = np.random.rand(1, length)[0]
    return w


def artificial2D():
    x = np.linspace(0, 1, 20)
    y = -2 * x + 2
    for i in range(len(y)):
        y[i] += np.random.uniform(-0.1, 0.1)

    data = []
    for i, j in zip(x, y):
        data.append([i, j])
    data = np.asarray(data)
    data = normaliza(data)
    data = insertbias(data)

    return np.asarray(data)


def artificial3D():
    data = np.ones((500, 3))
    for i in range(500):
        x1 = np.random.uniform(-0.7, 0.7)*10
        x2 = np.random.uniform(-0.7, 0.7)*10
        data[i][0] = x1
        data[i][1] = x2
        data[i][2] = (3 * x1 + 4 * x2 + 5 + (random.random() * 10))
    data = normaliza(data)
    data = insertbias(data)
    return np.asarray(data)


def insertbias(dataset):
    new = []
    for i in range(len(dataset)):
        new.append(np.insert(dataset[i], 0, -1))
    return np.asarray(new)


def normaliza(X):
    for i in range(X.shape[1]):
        max_ = max(X[:, i])
        min_ = min(X[:, i])

        for j in range(X.shape[0]):
            X[j, i] = (X[j, i] - min_) / (max_ - min_)

    return X


def plot2D(dataset, pesos):
    y =[]
    for i in dataset:
        y.append(np.dot(pesos, i[0:len(i) - 1]))

    plt.scatter(dataset[:, 1], dataset[:, -1], s=5, c='teal')
    plt.plot(dataset[:, 1], y, color='darkgrey')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Adaline - Base 2D')
    plt.show()


def plot3D(dataset, pesos):
    zModel = []
    for i in dataset:
        zModel.append(np.dot(pesos, i[0:len(i) - 1]))

    zModel = np.asarray(zModel)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = list(dataset[:, 1])
    y = list(dataset[:, 2])
    z = list(dataset[:, -1])
    x2, y2 = np.asarray(x), np.asarray(y).T
    ax.plot_trisurf(x2, y2, zModel.T, linewidth=0.2, antialiased=True, color="teal")
    ax.scatter(x, y, z, c='darkgrey', marker='o')
    plt.title('ADALINE - 3D')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


def plotErro(error):
    plt.plot(range(1, len(error) + 1), error)
    plt.xlabel('Iteração')
    plt.ylabel('Erros')
    plt.show()

# Carregar os dados


base = 1
array = np.asarray([])
if base == 0:
    array = artificial2D()
else:
    array = artificial3D()

X = array

validation_size = 0.20

X_treino, X_teste = model_selection.train_test_split(X, test_size=validation_size)

n_taxa_de_aprendizado = 0.1

qt_epocas = 20

qt_realizacoes = 20

w_pesos_sinapticos = np.random.rand(1, colunas(X_treino) - 1)[0]

print('Pesos sinápticos iniciais randômicos: ')
print(w_pesos_sinapticos)


acuracia = 0
best_hit = 0
hit_vet = []
erro_treinamento = []
mse = []
rmse = []

for realizacoes in range(qt_realizacoes):

    erro_epoca = 0

    for t_epoca in range(qt_epocas):

        erro = []

        for t_iteration in range(linhas(X_treino)):

            y = np.dot(w_pesos_sinapticos, X_treino[t_iteration, :colunas(X_treino) - 1].T) # Faz o somatório de WiXi (Σ)

            d = X_treino[t_iteration, colunas(X_treino) - 1]

            e = (d - y)  # Faz o calculo de D - Y

            w_pesos_sinapticos = w_pesos_sinapticos + n_taxa_de_aprendizado * e * X_treino[t_iteration, :colunas(X_treino) - 1].T

            erro = np.append(erro, e*e)

        erro_epoca = erro.mean()
        erro_treinamento.append(erro_epoca)

    # TESTE

    mse.append(np.mean(erro_treinamento))
    rmse.append(np.sqrt(mse[realizacoes]))
    print('Para a realização:', realizacoes + 1, ' temos, MSE:', mse[realizacoes], 'RMSE: ', rmse[realizacoes])

print('Pesos Ajustados: ')
print(w_pesos_sinapticos)

plotErro(erro_treinamento)





#TESTE
erro_teste =[]
auxmin = 1000
min = 0

for i in X_teste:
    y = np.dot(w_pesos_sinapticos, i[0:len(i) - 1])
    e = i[len(i) - 1] - y
    erro_teste.append(e*e)
erro_teste.append(np.mean(erro_teste))

print('')
print('Desvio Padrão: ', np.std(erro_teste))

if base == 0:
    plot2D(X, w_pesos_sinapticos.T)
elif base == 1:
    plot3D(X, w_pesos_sinapticos.T)
