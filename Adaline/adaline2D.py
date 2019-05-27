import numpy as np
from sklearn import model_selection
import matplotlib.pyplot as plt


def linhas(x):  # Retorna a quantidade de linhas em uma matriz
    return x.shape[0]


def colunas(x):  # Retorna a quantidade de colunas em ums matriz
    return x.shape[1]


def calc_erro(linha, X_treino):
    return 0.5 * (Y_treino[linha] - predict(linha, X_treino))**2  # O u[0] é chamado para pegar o elemento dentro do array


def predict(linha, X_treino):
    return np.where(somatorio(linha, X_treino)) # Linha/ Base


def somatorio(linha, X_treino):
    return X_treino[linha].dot(w_pesos_sinapticos)


def plotData2d(dataset, w):
    y = []
    for i in dataset:
        y.append(np.dot(w, i[0:len(i) - 1]))
        print(y)

    plt.scatter(dataset[:, 1], dataset[:, -1], s=3)
    #plt.plot(dataset[:, 1], y, color='r')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('ADALINE - Iteration with minimum MSE')
    plt.show()


def artificial1gen():
    x = np.linspace(0, 1, 20)
    y = 3 * x + 1
    for i in range(len(y)):
        y[i] += np.random.uniform(-0.2, 0.2)

    data = []
    for i, j in zip(x, y):
        data.append([i, j])
    data = np.asarray(data)
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


def testdata():
    a = artificial1gen()
    plt.scatter(a[:, 1], a[:, -1], s=3)
    plt.show()

# Carregar os dados

testdata()
dataset = artificial1gen()




# Embaralha e divide os dados em treinamento e teste
array = dataset
X = array


plot = 1  # Define se terá plot ou não

Y = array[:, 2]

validation_size = 0.20

X_treino, X_teste, Y_treino, Y_teste = model_selection.train_test_split(X, Y, test_size=validation_size)


Y_treino = Y_treino.T # Troca os nomes em string para 0 ou 1

Y_teste = Y_teste.T # Troca os nomes em string para 0 ou 1


n_taxa_de_aprendizado = 0.1

qt_epocas = 20

qt_realizacoes = 20

w_pesos_sinapticos = 2 * np.random.random((colunas(X_treino), 1)) - 1



print('Pesos sinápticos iniciais randômicos: ')
print(linhas(X_treino)) # Retorna a quantidade de elementos em um array de arrays
print(w_pesos_sinapticos)

acuracia = 0
best_hit = 0
hit_vet = []

for realizacoes in range(qt_realizacoes):

    for t_epoca in range(qt_epocas):

        qt_erros = 0

        for t_iteration in range(linhas(X_treino)):

            u = somatorio(t_iteration, X_treino)  # Faz o somatório de WiXi (Σ)


            erro = calc_erro(t_iteration, X_treino)  # Faz o calculo de D - Y (A função de ativação é chamada aqui)

            for i in range(len(w_pesos_sinapticos)):
                w_pesos_sinapticos[i] = [w_pesos_sinapticos[i] + n_taxa_de_aprendizado * erro * X_treino[t_iteration, i]]  # Função de aprendizagem para cada Wi

            if erro != 0:
                qt_erros = qt_erros + 1

        if qt_erros == 0:
            print('Saiu com erro == 0!')
            print('Na epóca: ')
            print(t_epoca)
            break

    predicao = []

    for t_iteration in range(linhas(X_teste)):
        predicao = np.append(predicao, predict(t_iteration, X_teste))

    # Cálculo dos hits
    hit = 0
    for interacao in range(Y_teste.size):
        if predicao[interacao] == Y_teste[interacao]:
            hit = hit + 1

    hit_vet = np.append(hit_vet, ((hit * 100)) / Y_teste.size)

    matriz = np.zeros(shape=(2, 2))

    if hit > best_hit:
        best_hit = hit
        # Seleciona a melhor Matriz de confusão
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

    acuracia = acuracia + hit

print('Pesos Ajustados: ')
print(w_pesos_sinapticos)

# TESTE

print('Predição: ')
print(predicao)
print('Desejado: ')

print('--------------------------')
print('Acurácia: ')
print((acuracia / (Y_teste.size * qt_realizacoes) * 100))
print(hit_vet.mean())

print('Melhor Matriz de confusão: ')
print(matriz)

print('Desvio Padrão: ')
print(hit_vet.std())

print('--------------------------')

plot =1
if plot == 1:
    weights = w_pesos_sinapticos
    #TODO
    points = np.linspace(0, 1, 20)
    c_points = []
    for p in points:
        data = []
        if len(weights) == 3:
            data = [-0.2, p, 0.2]
        c_points.append(data)

    c_points = np.array(c_points)
    predict = np.dot(weights.T, c_points.T)
    plt.scatter(points, predict, c='r', s=3)


plt.show()