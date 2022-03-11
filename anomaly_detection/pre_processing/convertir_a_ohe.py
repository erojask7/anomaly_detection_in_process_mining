# This file was based on the following code: https://github.com/jujucacris/process-mining/blob/master/MLP_python/Pre-processamento/Conversor%20de%20JSON/tabela_atividades_nolle.py
# author: @jujucacris (https://github.com/jujucacris/)
#FIX2022forpubrep: ready for publication
import csv

# função para gerar as representações binárias das atividades
def gera_numeros(atividades): # recebe o vetor com todas as atividades
    b = [] # vetor finalizado com o binário de todas as atividades
    s = [] # vetor binário para iteração
    i = 0 # iterador para contagem das atividades
    for item in atividades: # itera o vetor de atividades
        s.append("0") # adiciona um 0 no vetor binário para cada atividades
    for item in atividades: # itera o vetor de atividades
        s[i] = "1" # modifica o bit correspondente à atividade para 1
        b.append("".join(s)) # adiciona a representação da atividade ao vetor final
        s[i] = "0" # volta o bit para 0
        i = i + 1
    return b # retorna o vetor finalizado com os códigos binários

# função para descobrir todas as atividades presentes no log
def descobre_atividades(log_filename,input_path): # recebe o nome do arquivo a ser tratado
    f = open(input_path / log_filename, "r")
    f = csv.reader(f, delimiter=",") # abertura do arquivo csv como matriz
    #s = open((input_path / ("lista_atividades_" + log_filename)), "w")
    atividades = [] # vetor das atividades
    i = 1 # iterador
    next(f) # pula a primeira linha do arquivo
    for row in f: # itera as linhas do arquivo
        #line = next(f) # pula para a próxima linha (ignorando a primeira)
        if row[2] in atividades: # verifica se o nome da atividade já existe no vetor
            continue # se sim, passa para a próxima iteração
        else:
            atividades.append(row[2]) # se não, adiciona o nome da atividade no vetor
            i = i + 1
    c = gera_numeros(atividades) # armazena o vetor de representações binárias
    atividades.sort() # ordena as atividades alfabeticamente
    print(str(len(atividades)) + " activities found in log") # imprime a quantidade de atividades descobertas
    atividades_dict = dict(zip(atividades, c)) # cria um dicionário associando atividades e binários
    # for key, value in atividades_dict.items():
    #     s.write(key + " : " + value + "\n")
    return atividades_dict # retorna o dicionário criado

# função para formatar as strings dos traces para impressão no arquivo
def formata(p):
    p = str(p).replace("[", "").replace("]", "\n")
    p = p.replace("'", "").replace(", ", "")
    p = ",".join(p)
    return p

# função para completar cada trace com '0's para que todos tenham o mesmo tamanho
def completa(rows, max_size): # recebe a matriz de linhas e o tamanho máximo dos traces desse log
    for row in rows: # itera as linhas a serem impressas
        tipo = row.pop() # remove o identificador de tipo do final da lista
        while len(row) < max_size: # enquanto a linha não tiver o mesmo tamanho do maior trace
            row.append("0"*len(row[0])) # completa a linha com atividades em branco
        row.append(tipo) # devolve o identificador de tipo ao final da lista
    return rows # retorna matriz de traces

# função principal da conversão de eventos para traces binários
def converte(log_filename, a,input_path,output_path): # recebe o nome do arquivo de entrada e o dicionário de atividades
    import os
    f = open(input_path / log_filename, "r")
    f = csv.reader(f, delimiter=",") # abre o csv como matriz
    p = [] # vetor onde será armazenado cada trace
    ohe_filename="ohe_{}".format(log_filename)
    r = open(os.path.join(output_path,ohe_filename), "w") # abre o arquivo de saída
    i = 1 # iterador de traces

    max_size = 0 # variável de tamanho máximo de trace
    rows = [] # matriz de saída que será impressa no arquivo
    next(f)
    for row in f: # para cada linha do arquivo de entrada
         if row[0] == str(i): # se o identificador do trace for igual ao que estava sendo tratado
            tipo = row[1] # armazena o tipo de evento, normal ou anomalia
            atividade = a[row[2]] # armazena a atividade realizada em representação binária a partir do dicionário
            p.append(atividade) # adiciona a atividade binária ao vetor do trace
         else: # caso o identificador do trace mude
            if len(p) > max_size: # se o comprimento do trace for maior do que o valor maior armazenado
                 max_size = len(p) # subistitui o maior valor pelo comprimento do trace atual
            p.append(tipo[0]) # adiciona o tipo de trace ao final do vetor
            rows.append(p) # adiciona o trace à matriz final
            p = [] # zera o vetor do trace
            i = i + 1 # passa para o próximo identificador de trace
            atividade = a[row[2]] # armazena a nova atividade
            p.append(atividade) # adiciona atividade como primeira atividade do novo trace
    if len(p) > max_size:
         max_size = len(p)
    p.append(tipo[0])
    rows.append(p)
    rows = completa(rows, max_size) # chama função que nivela todos os traces
    for row in rows: # para cada linha na matriz de traces
         r.write(formata(row)) # imprime linha formatada no arquivo
    print(str(max_size) + " events existing in the biggest trace") # imprime o tamanho do maior trace
    print("The one hot encoding representation for %s was created!. \n"%log_filename)


def convertir(log_filename,input_path,output_path):
    """
    This function convert an event log (log_filename) to an one hot encoding reprsentation , as
    described in https://doi.org/10.1007/s10994-018-5702-8
    :param log_filename: name of the file that will be converted
    :param input_path: path to the directory where the log_filename is located
    :param output_path:  path to the directory where the one hot ncoding representation will be saved
    """

    atividades = descobre_atividades(log_filename,input_path)

    # Create Folder
    import os
    folder_output=os.path.join(output_path)
    if not (os.path.isdir(folder_output))  : #If folder doesnt exist
        os.mkdir(folder_output)    #Creat folder

    converte(log_filename, atividades,input_path,output_path)
