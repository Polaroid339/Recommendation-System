import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify

"""
Microsserviço que recomenda memes com base nas interações dos usuários.
As interações vão de 0 a 3 (CURTIDO, SALVO, POSTADO, REPOSTADO).
Esse código cria um microserviço Flask que faz recomendações de memes.
"""

# Carregar os dados dos memes e interações
memes = pd.read_csv('memes_dataset.csv')
user_interactions = pd.read_csv('user_interactions.csv')

# Renomeando as colunas para facilitar o entendimento
memes.rename(columns={"ID_MEME": "ID_MEME", "CURTIDAS": "CURTIDAS"}, inplace=True)

# Criando um dicionário para traduzir os tipos de interação (0 a 3)
interaction_map = {
    0: "CURTIDO",
    1: "SALVO",
    2: "POSTADO",
    3: "REPOSTADO"
}

# Substituindo os números na coluna 'INTERACTION_TYPE' pelos nomes
user_interactions['INTERACTION_TYPE'] = user_interactions['INTERACTION_TYPE'].map(interaction_map)

# Preenchendo os espaços vazios nas tags dos memes
memes['TAG_IDS'] = memes['TAG_IDS'].fillna('')  # Se não tiver tag, coloca vazio

# Transformando as tags dos memes em números para calcular similaridade
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(memes['TAG_IDS'])

# Calculando a similaridade entre os memes com base nas tags
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Criando um dicionário para achar os memes rapidamente
indices_meme = pd.Series(memes.index, index=memes['ID_MEME']).to_dict()

app = Flask(__name__)

@app.route('/recomendar', methods=['GET'])
def recomendar_memes():
    
    """Exemplo: http://127.0.0.1:5000/recomendar?user_id=<USER_ID>"""

    user_id = request.args.get('user_id', type=str)

    # Checando se o usuário existe nas interações
    if user_id not in user_interactions['USER_ID'].values:
        return jsonify({"erro": "Usuário não encontrado ou sem interações."})

    # Pegando as interações do usuário
    interacoes_usuario = user_interactions[user_interactions['USER_ID'] == user_id]
    
    # Pegando os memes que o usuário já interagiu
    memes_interagidos = interacoes_usuario['MEME_ID'].unique()

    # Vamos procurar memes semelhantes aos que o usuário já interagiu
    recomendações = []
    memes_recomendados_set = set()  # Usando um set para garantir que memes não sejam repetidos

    for meme_id in memes_interagidos:
        idx = indices_meme.get(meme_id)
        
        if idx is not None:
            # Pegando a similaridade dos memes com o meme atual
            sim_scores = list(enumerate(cosine_sim[idx]))
            
            # Ordenando pelos memes mais parecidos (sem incluir o próprio meme)
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:]

            for i, score in sim_scores[:5]:  # Pega até 5 memes mais parecidos
                meme_recomendado = memes.iloc[i]
                
                # Verifica se o meme já foi interagido pelo usuário e se já foi recomendado
                if meme_recomendado['ID_MEME'] not in memes_interagidos and meme_recomendado['ID_MEME'] not in memes_recomendados_set:
                    # Adiciona o meme ao set de memes recomendados para não repetir
                    memes_recomendados_set.add(meme_recomendado['ID_MEME'])
                    
                    # Adiciona o tipo de interação, se houver
                    tipo_interacao = interacoes_usuario[interacoes_usuario['MEME_ID'] == meme_recomendado['ID_MEME']]['INTERACTION_TYPE'].values
                    recomendações.append({
                        'ID_MEME': meme_recomendado['ID_MEME'],
                        'TITLE': meme_recomendado['TITLE'],
                        'DESCRIPTION': meme_recomendado['DESCRIPTION'],
                        'URL': meme_recomendado['URL'],
                        'TAG_IDS': meme_recomendado['TAG_IDS'],
                        'CURTIDAS': meme_recomendado['CURTIDAS'],
                        'similaridade': score,
                        'TIPO_INTERACAO': tipo_interacao.tolist() if len(tipo_interacao) > 0 else 'Nenhuma Interação'
                    })
    
    # Ordenando as recomendações pela similaridade (maior para menor)
    recomendações = sorted(recomendações, key=lambda x: x['similaridade'], reverse=True)

    # Garantindo que valores de curtidas e similaridade sejam do tipo correto
    for recomendacao in recomendações:
        recomendacao['CURTIDAS'] = int(recomendacao['CURTIDAS'])  # Transformando em inteiro
        recomendacao['similaridade'] = float(recomendacao['similaridade'])  # Transformando em float

    # Retorna as recomendações no formato JSON
    return jsonify({"recomendacoes": recomendações})

# Rodando a aplicação Flask
if __name__ == '__main__':
    app.run(debug=True)