import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from flask import Flask, request, jsonify

"""
Sistema de recomendação baseado em SciKit Learn, onde cruza e trata os dados
dos usuários, retornando um arquivo json com os memes recomendados baseado
nas interações do usuário. O sistema é exposto em microserviço com Flask.
"""

# Datasets
memes = pd.read_csv('memes_dataset.csv', low_memory=False)
user_interactions = pd.read_csv('user_interactions.csv', low_memory=False)

# Filtrando as colunas necessárias
memes = memes[['meme_id', 'meme_tags', 'curtidas']]

# Renomeando as variáveis para facilitar a visualização
memes.rename(columns={'meme_id': 'ID_MEME','meme_tags': 'TAGS', 'curtidas': 'CURTIDAS'}, inplace=True)

# Cruzar interações dos usuários com os memes
dados_cruzados = pd.merge(user_interactions, memes,
                       left_on='meme_id', right_on='ID_MEME')

# Matriz de interações
user_meme_matriz = dados_cruzados.pivot_table(
    index='user_id', columns='ID_MEME', values='interacao', fill_value=0)

# Certificar-se de que o index (user_id) seja tratado como string
user_meme_matriz.index = user_meme_matriz.index.astype(str)

# Criar uma matriz esparsa de interações
user_meme_sparse = csr_matrix(user_meme_matriz)

# Treinando o modelo
modelo = NearestNeighbors(algorithm='brute')
modelo.fit(user_meme_sparse)

app = Flask(__name__)

@app.route('/recomendar', methods=['GET'])
def recomendar_memes():
    
    """Exemplo: http://127.0.0.1:5000/recomendar?user_id=1
    Caso não haja memes suficientes, recomenda os com mais curtidas"""

    user_id = request.args.get('user_id', type=str)  # Alterado para string
    
    # Definir o limite máximo de memes para recomendar
    max_recomendacoes = 3

    if user_id not in user_meme_matriz.index:
        return jsonify({"erro": "Usuário não encontrado ou sem interações suficientes."})

    usuario_interacoes = user_meme_matriz.loc[user_id].values.reshape(1, -1)

    num_memes_disponiveis = len(user_meme_matriz.columns)
    max_neighbors = min(max_recomendacoes, num_memes_disponiveis)

    # Encontrar os memes mais semelhantes
    distancias, indices = modelo.kneighbors(usuario_interacoes, n_neighbors=max_neighbors)

    # Verificar se os indices estão dentro do limite de colunas
    indices_validos = [i for i in indices.flatten() if i < len(user_meme_matriz.columns)]

    # Garantir que não haja indices fora do limite
    memes_recomendados = user_meme_matriz.columns[indices_validos].tolist()

    # Detalhes dos memes recomendados
    detalhes_memes = memes[memes['ID_MEME'].isin(memes_recomendados)]

    # Ordenar os memes recomendados pelas curtidas e média de interações (decrescente)
    detalhes_memes['media_interacao'] = detalhes_memes['ID_MEME'].apply(
        lambda x: user_meme_matriz[x].mean() if x in user_meme_matriz.columns else 0
    )
    detalhes_memes = detalhes_memes.sort_values(by=['media_interacao', 'CURTIDAS'], ascending=False)

    # Retornar os memes recomendados ordenados por interação e curtidas
    return jsonify({"recomendacoes": detalhes_memes.to_dict(orient='records')})

if __name__ == '__main__':
    app.run(debug=True)
