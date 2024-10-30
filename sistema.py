from flask import Flask, jsonify, request
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import CountVectorizer

system = Flask(__name__)
memes_dataset = pd.read_csv("dataset.csv")


def get_recomendacoes(user_id):

    """
    Função que ira filtrar os dados para o usuário especificado e pegar
    apenas os memes curtidos, vetorizando as tags e treinando o modelo
    com as tags vetorizadas, gerando assim uma lista com os ids dos
    memes que serão recomendados para o usuário.
    """

    user_data = memes_dataset[(memes_dataset['user_id'] == user_id) & (
        memes_dataset['curtido'] == 1)]

    if user_data.empty:
        print(f'Nenhum dado encontrado para o usuário {user_id}')
        return []

    vetorizar = CountVectorizer()
    tags_vetorizadas = vetorizar.fit_transform(memes_dataset['meme_tags'])
    print("Tags vetorizadas:", tags_vetorizadas.toarray())

    # A quantidade de memes sugeridos pode ser alterado aqui
    modelo = NearestNeighbors(n_neighbors=6)
    modelo.fit(tags_vetorizadas)

    usuario_tags_vetorizadas = vetorizar.transform(user_data['meme_tags'])
    recomendacoes = modelo.kneighbors(usuario_tags_vetorizadas, return_distance=False)

    memes_recomendados = memes_dataset.iloc[recomendacoes[0]]['meme_id'].tolist()
    print(f"Memes recomendados para o usuário {user_id}:", memes_recomendados)

    memes_recomendados = [meme for meme in memes_recomendados if meme not in user_data['meme_id'].tolist()]

    return memes_recomendados


@system.route('/recomendacoes/<int:user_id>', methods=['GET'])
def recomendacoes(user_id):

    """
    O resultado esperado dever ser um arquivo.json contendo um dicionario com
    as recomendações de memes baseado no que o usuário em questão curtiu.
    
    Exemplo: http://127.0.0.1:5000/recomendacoes/2
    """

    memes_recomendados = get_recomendacoes(user_id)
    return jsonify({'recomendacoes': memes_recomendados})


if __name__ == '__main__':
    system.run(host='0.0.0.0', port=5000)
