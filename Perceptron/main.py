import numpy as np

def linhas(x): # Retorna a quantidade de linhas em uma matriz
    return x.shape[0]

def colunas(x): # Retorna a quantidade de colunas em ums matriz
    return x.shape[1]

def calc_erro(t_iteration, u): # Retorna a quantidade de colunas em ums matriz
    return y_saida_treinamento[t_iteration] - funcao_de_ativacao(u[0])

def funcao_de_ativacao(u):
    if(u >= 0):
        return 1
    return 0


x_entrada_treinamento = np.array([[-1,2,2], [-1,4,4]])

y_saida_treinamento = np.array([[1,0]]).T

n_taxa_de_aprendizado = 0.1

qt_epocas = 80

w_pesos_sinapticos = 2 * np.random.random((colunas(x_entrada_treinamento), 1)) - 1


print('Pesos sinápticos iniciais randômicos: ')
print(linhas(x_entrada_treinamento)) # Retorna a quantidade de elementos em um array de arrays
print(w_pesos_sinapticos)

for t_epoca in range(qt_epocas):

    qt_erros = 0

    print('Epóca: ')
    print(t_epoca)

    for t_iteration in range(linhas(x_entrada_treinamento)):

        u = x_entrada_treinamento[t_iteration].dot(w_pesos_sinapticos) #Faz o somatório de WiXi (Σ)

        erro = calc_erro(t_iteration, u) # Faz o calculo do erro, ou seja D - Y

        for i in range(len(w_pesos_sinapticos)):
            w_pesos_sinapticos[i] = [w_pesos_sinapticos[i] + n_taxa_de_aprendizado * erro * x_entrada_treinamento[t_iteration, i]] # Função de aprendizagem para cada Wi

        print('Iteração: ')
        print(t_iteration)
        #print(w_pesos_sinapticos)

        if erro != 0:

            qt_erros = qt_erros + 1

        print('Quantidade de erros: ')
        print(qt_erros)

    if qt_erros == 0:
        print('Pesos Ajustados: ')
        print(w_pesos_sinapticos)
        break
