import numpy as np
import pandas
from sklearn import model_selection
import matplotlib.pyplot as plt

def linhas(x):  # Retorna a quantidade de linhas em uma matriz
    return x.shape[0]


def colunas(x):  # Retorna a quantidade de colunas em ums matriz
    return x.shape[1]


def calc_erro(linha, X_treino):
    return Y_treino[linha] - predict(linha, X_treino)  # O u[0] é chamado para pegar o elemento dentro do array


def somatorio(linha, X_treino):
    return X_treino[linha].dot(w_pesos_sinapticos)


def predict(linha, X_treino):
    return np.where(somatorio(linha, X_treino) >= 0.0, 1, 0) # Linha/ Base


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

def normaliza(X):
    for i in range(X.shape[1]):
        max_ = max(X[:, i])
        min_ = min(X[:, i])

        for j in range(X.shape[0]):
            X[j, i] = (X[j, i] - min_) / (max_ - min_)

    return X


# Carregar os dados
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)




# Embaralha e divide os dados em treinamento e teste
array = dataset.values
X = array[:, [0, 2]] #X = array[:, 0:4] # X = array[:,0:4] # X = array[:, [0, 2]]  # Quantidade de atributos


X = normaliza(X)

tipo = 2  # Define qual classe será vs Outras

plot = 1  # Define se terá plot ou não

Y = array[:, 4]
Y = selecao_base(tipo, Y)
validation_size = 0.20
X_treino, X_teste, Y_treino, Y_teste = model_selection.train_test_split(X, Y, test_size=validation_size)



# Adiciona os valores de X0
X_treino = np.insert(X_treino, 0, -1, axis = 1)

X_teste = np.insert(X_teste, 0, -1, axis = 1)


Y_treino = Y_treino.T # Troca os nomes em string para 0 ou 1

Y_teste = Y_teste.T # Troca os nomes em string para 0 ou 1

n_taxa_de_aprendizado = 0.1

qt_epocas = 200

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
                w_pesos_sinapticos[i] = [w_pesos_sinapticos[i] + n_taxa_de_aprendizado * erro * X_treino[t_iteration, i]] # Função de aprendizagem para cada Wi

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

    hit_vet = np.append(hit_vet, ((hit*100))/Y_teste.size)

    if hit > best_hit:
        best_hit = hit
        # Seleciona a melhor Matriz de confusão
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


    acuracia = acuracia + hit

print('Pesos Ajustados: ')
print(w_pesos_sinapticos)


#TESTE

print('Predição: ')
print(predicao)
print('Desejado: ')

print('--------------------------')
print('Acurácia: ')
print((acuracia/(Y_teste.size * qt_realizacoes)* 100))
print(hit_vet.mean())

print('Melhor Matriz de confusão: ')
print(matriz)

print('Desvio Padrão: ')
print(hit_vet.std())

print('--------------------------')


if plot == 1:
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.009),
                         np.arange(y_min, y_max, 0.009))

    base = np.c_[xx.ravel(), yy.ravel()]
    base = np.insert(base, 0, -1, axis = 1)


    f, axarr = plt.subplots(1, 1, sharex='col', sharey='row', figsize=(10, 8))

    Z = []

    for i in range(linhas(base)):
        Z = np.append(Z, predict(i, base))

    Z = Z.reshape(xx.shape)

    axarr = plt.contourf(xx, yy, Z, alpha=0.7)
    axarr = plt.scatter(X[:, 0], X[:, 1], c=Y,
                                  s=20, edgecolor='k')

    if tipo == 0:
        axarr = plt.title('Setosa vs Outras')
    if tipo == 1:
        axarr = plt.title('Virginica vs Outras')
    if tipo == 2:
        axarr = plt.title('Versicolor vs Outras')

    plt.show()