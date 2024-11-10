from flask import Flask, jsonify, request
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import CountVectorizer

system = Flask(__name__)

# Carrega o dataset com colunas 'user_id', 'meme_id', 'meme_tags', 'curtido'
memes_dataset = pd.read_csv("dataset.csv")


def get_recomendacoes(user_id):
    """
    Recomenda memes com base nas tags dos memes que o usuário curtiu.
    """
    # Filtra os memes que o usuário curtiu
    user_data = memes_dataset[(memes_dataset['user_id'] == user_id) & (memes_dataset['curtido'] == 1)]
    
    if user_data.empty:
        print(f'Nenhum dado encontrado para o usuário {user_id}')
        return []

    # Concatena todas as tags dos memes que o usuário curtiu
    tags_usuario = " ".join(user_data['meme_tags'])

    # Vetoriza todas as tags no dataset
    vetorizar = CountVectorizer()
    tags_vetorizadas = vetorizar.fit_transform(memes_dataset['meme_tags'])

    # Treina o modelo com as tags vetorizadas
    modelo = NearestNeighbors(n_neighbors=4)
    modelo.fit(tags_vetorizadas)

    # Vetoriza as tags concatenadas do usuário
    usuario_tags_vetorizadas = vetorizar.transform([tags_usuario])
    recomendacoes = modelo.kneighbors(usuario_tags_vetorizadas, return_distance=False)

    # Gera a lista de IDs dos memes recomendados
    memes_recomendados = memes_dataset.iloc[recomendacoes[0]]['meme_id'].tolist()
    print(f"Memes recomendados para o usuário {user_id}:", memes_recomendados)

    # Remove memes que o usuário já curtiu
    memes_recomendados = [meme for meme in memes_recomendados if meme not in user_data['meme_id'].tolist()]

    return memes_recomendados


@system.route('/recomendacoes/<int:user_id>', methods=['GET'])
def recomendacoes(user_id):
    """
    Retorna um JSON com recomendações de memes para um usuário específico.
    """
    memes_recomendados = get_recomendacoes(user_id)
    return jsonify({'recomendacoes': memes_recomendados})


if __name__ == '__main__':
    system.run(host='0.0.0.0', port=5000)
