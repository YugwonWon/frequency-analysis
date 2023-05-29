
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import itertools
import json
import re
import pickle

from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd

from data_loader import DataLoader

from collections import Counter
from wordcloud import WordCloud

from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

from bareunpy import Tagger

import networkx as nx
import os
from matplotlib import rc
from tqdm import tqdm

rc('font', family='AppleGothic')
plt.rcParams['font.family'] = 'AppleGothic' # 사용할 한글 폰트명 입력
plt.rcParams['axes.unicode_minus'] = False # 마이너스 폰트 설정


class FrequencyAnalysis:
    def __init__(self):
        pass
        
    def cloud_visulize(self, noun_list, filename, white = True):
        if white == True:
            wordcloud = WordCloud(width=2000, height=1000, font_path = '/Library/Fonts/AppleGothic.ttf', collocations=True, background_color='white')
        else:
            wordcloud = WordCloud(width=2000, height=1000, font_path = '/Library/Fonts/AppleGothic.ttf', collocations=True)
        wordcloud.generate_from_frequencies(dict(noun_list))
        wordcloud.to_file(filename)
        wordcloud = None
        
    def cloud_visualize_idf(self, noun_list, filename):
        from sklearn.feature_extraction.text import TfidfVectorizer
        
    def wordcloud_Rank(self, token, N = 30, white = True, outdir = 'out', filename=''):
        if not os.path.exists(outdir + '/Wordcloud'):
            os.makedirs(outdir + '/Wordcloud')
        filename_png = outdir + f'/Wordcloud/{filename}.png'
        filename_csv = outdir + f'/Wordcloud/{filename}.csv'
        
        flatten_tokens = list(itertools.chain(*token))
        token_count = Counter(flatten_tokens)
        Top_token_count = token_count.most_common(N)
        
        word_count = pd.DataFrame(Top_token_count)
        word_count.columns = ['Word', 'Count']
        word_count.to_csv(filename_csv, index = False, encoding='euc-kr')
        
        self.cloud_visulize(Top_token_count, filename_png, white)
        
        return word_count
    
    def Top_Keyword_Edgelist(self, Token, Edge_list, N, outdir = '', node_size = 50, edge_size = 10, filename=''):
        if not os.path.exists(outdir + '/Graph'):
            os.makedirs(outdir + '/Graph')
        filename = outdir + f'/Graph/{filename}'
        flatten_tokens = list(itertools.chain(*Token))
        All_words = Counter(flatten_tokens)
        Top_words = All_words.most_common(N)
        Top_words = [word for (word, count) in Top_words]
        Top_Edge_list = list((x, y, v) for (x,y,v) in Edge_list if (x in Top_words and  y in Top_words))
        G_top = nx.Graph()
        
        for A,B, weight in Top_Edge_list: G_top.add_edge(A, B, weight=weight, distance = 1/weight)
        width_top = list(nx.get_edge_attributes(G_top, 'weight').values())
        degree = [All_words[node] * node_size for node in list(G_top.nodes)]
        edges_top = G_top.edges()
        weights = [1 for u,v in edges_top]
        
        with open(f'{filename}.json', 'w', encoding='utf-8') as f:
            json.dump(G_top.adj._atlas, f, ensure_ascii=False, indent=2)
        plt.figure(figsize=(20,20))
        pos = nx.spring_layout(G_top, k = 0.5)
        nx.draw(G_top, pos, width=weights, node_color= 'yellow', node_size = degree, edge_color= 'green', edge_cmap=plt.cm.Reds)
        nx.draw_networkx_labels(G_top, pos, font_size=32, font_family='AppleGothic', font_color='black')
        plt.savefig(f'{filename}.jpg')
        
    def make_Co_Keyword(self, Text):
        Co_Keyword = list(itertools.combinations(Text,2))
        Co_Keyword = [(x,y) for (x,y) in Co_Keyword if x != y]
        return Co_Keyword
    
    def make_Edgelist(self, Token):
        Co_Keyword = [self.make_Co_Keyword(text) for text in Token]
        Co_Keyword = list(itertools.chain(*Co_Keyword))
        Edge_list = list((x, y, v) for (x,y), v in Counter(Co_Keyword).items())
        return Edge_list
    
    def preprocess_text(self, lines):
        """ 
        불용어를 제거하기 위해 토큰화한다.
        :param lines: 문장 리스트
        :return: 토큰화된 문장 리스트 
        """
        count_dict = {}
        tagger = Tagger('2AVZBLY-D7QUNYQ-XNCRIOA-NVP3TLA', 'localhost', port=5656)
        tokenized_lines = []
        for line in tqdm(lines):
            tagged = tagger.tags([line])
            nouns = []
            for p in tagged.pos():
                if len(p[0]) > 1:
                    if p[1] == 'NNG' or p[1] == 'NNP':
                        nouns.append(p[0])
            if len(nouns) != 0:
                for n in nouns:
                    if n in count_dict:
                        count_dict[n] += 1
                    else:
                        count_dict[n] = 1
                tokenized_lines.append(nouns)

        return tokenized_lines

def main(files):
    for file in files:
        if file.endswith('.txt'):
            texts = []
            basename = os.path.basename(file).split('.txt')[0]
            with open(file, 'r', encoding='utf-8') as f:
                for line in f:
                    texts.append(line.strip())
            join_text = "".join(texts)     
            cleaned_text = re.sub(r"●|•|ㅇ|■", "", join_text)   
            texts =  [t + '.' for t in cleaned_text.split(". ")]
            # 빈도 분석 객체 생성
            fa = FrequencyAnalysis()
            
            # 형태소 분석 명사 추출
            Tokens = fa.preprocess_text(texts)
            
            # 워드클라우드 생성
            word_count = fa.wordcloud_Rank(Tokens, 50, white = True, outdir = 'out', filename=basename)
            print(word_count.head(10))
            
            # 키워드 네트워크
            Edge_list = fa.make_Edgelist(Tokens)
            fa.Top_Keyword_Edgelist(Tokens, Edge_list, 50, outdir = 'out', node_size = 50, edge_size = 6, filename=basename)

def find_most_frequent_elements(strings):
    frequency = {}
    max_frequency = 0

    # Count the frequency of each element
    for string in strings:
        if string in frequency:
            frequency[string] += 1
        else:
            frequency[string] = 1

        # Update max_frequency if necessary
        if frequency[string] > max_frequency:
            max_frequency = frequency[string]

    most_frequent_elements = []

    # Find elements with maximum frequency
    for string, count in frequency.items():
        if count == max_frequency:
            most_frequent_elements.append(string)

    return most_frequent_elements

# Example usage
strings = ["apple", "banana", "apple", "orange", "banana", "apple"]
most_frequent = find_most_frequent_elements(strings)
print("Most frequent elements:", most_frequent)
    

if __name__ == '__main__':
    # 데이터 로딩
    dl = DataLoader()
    files = dl.get_file_list('data')
    
    # 분석
    main(files)
    
    # 문서와 라벨을 생성한다.
    documents = []
    if os.path.isfile('data/pkl/idf.pkl'):
        with open('data/pkl/idf.pkl', 'rb') as f:
           documents = pickle.load(f)
    else:
        for file in files:
            texts = []
            basename = os.path.basename(file).split('.txt')[0]
            with open(file, 'r', encoding='utf-8') as f:
                for line in f:
                    texts.append(line.strip())
            join_text = "".join(texts)     
            cleaned_text = re.sub(r"●|•|ㅇ|■", "", join_text)   
            texts =  [t + '.' for t in cleaned_text.split(". ")]
            # 빈도 분석 객체 생성
            fa = FrequencyAnalysis()
            
            # 형태소 분석 명사 추출
            temp_lines = []
            Tokens = fa.preprocess_text(texts)
            for t in Tokens:
                temp_lines.append(' '.join(t))
            documents.append((" ".join(temp_lines), basename))
        with open('data/pkl/idf.pkl', 'wb') as f:
            pickle.dump(documents, f, pickle.HIGHEST_PROTOCOL)
    # TF-IDF 벡터를 생성
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([doc[0] for doc in documents])
    outdir = 'out/tf-idf'
    # 각 문서의 중요한 단어를 추출
    for i, doc in enumerate(documents):
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        filename = outdir + f'/{doc[1]}'
        tfidf_scores = dict(zip(vectorizer.get_feature_names_out(), tfidf_matrix[i].toarray().flatten()))
        sorted_scores = sorted(tfidf_scores.items(), key=lambda item: item[1], reverse=True)
        top_scores = sorted_scores[:20]  # 상위 n개 단어를 선택
        plot_title = f"대선별 선거 공약집에서 각 단어의 중요도({doc[1]})"
        plt.figure(figsize=(10, 5))
        plt.bar(*zip(*top_scores))
        plt.title(plot_title, y=1.05)
        plt.xlabel('상위 20개 단어', labelpad=10)
        plt.ylabel('TF-IDF 점수', labelpad=10)
        plt.savefig(f'{filename}.jpg')
        
    # # LDA 모델 생성
    # lda = LatentDirichletAllocation(n_components=10, random_state=0)
    # lda_doc = [d[0].split(' ') for d in documents]
    # # tf-idf 벡터에 대해 LDA 학습
    # from gensim import corpora
    # dictionary = corpora.Dictionary(lda_doc)
    # corpus = [dictionary.doc2bow(text) for text in lda_doc]
    # # print(corpus[1]) # 수행된 결과에서 두번째 뉴스 출력. 첫번째 문서의 인덱스는 0
    # import gensim
    # NUM_TOPICS = 10 # 20개의 토픽, k=20
    # ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15)
    # topics = ldamodel.print_topics(num_words=4)
    # # for topic in topics:
    # #     print(topic)
    # # lda.fit(tfidf_matrix)
    # # lda 시각화
    
    # import pyLDAvis.gensim_models

    # pyLDAvis.enable_notebook()
    # vis = pyLDAvis.gensim_models.prepare(ldamodel, corpus, dictionary)
    # pyLDAvis.display(vis)
    # for i, topic_list in enumerate(ldamodel[corpus]):
    #     if i==5:
    #         break
    #     print(i,'번째 문서의 topic 비율은',topic_list)
        
    # def make_topictable_per_doc(ldamodel, corpus):
    #     topic_table = pd.DataFrame()

    #     # 몇 번째 문서인지를 의미하는 문서 번호와 해당 문서의 토픽 비중을 한 줄씩 꺼내온다.
    #     for i, topic_list in enumerate(ldamodel[corpus]):
    #         doc = topic_list[0] if ldamodel.per_word_topics else topic_list            
    #         doc = sorted(doc, key=lambda x: (x[1]), reverse=True)
    #         # 각 문서에 대해서 비중이 높은 토픽순으로 토픽을 정렬한다.
    #         # EX) 정렬 전 0번 문서 : (2번 토픽, 48.5%), (8번 토픽, 25%), (10번 토픽, 5%), (12번 토픽, 21.5%), 
    #         # Ex) 정렬 후 0번 문서 : (2번 토픽, 48.5%), (8번 토픽, 25%), (12번 토픽, 21.5%), (10번 토픽, 5%)
    #         # 48 > 25 > 21 > 5 순으로 정렬이 된 것.

    #         # 모든 문서에 대해서 각각 아래를 수행
    #         for j, (topic_num, prop_topic) in enumerate(doc): #  몇 번 토픽인지와 비중을 나눠서 저장한다.
    #             if j == 0:  # 정렬을 한 상태이므로 가장 앞에 있는 것이 가장 비중이 높은 토픽
    #                 topic_table = topic_table._append(pd.Series([int(topic_num), round(prop_topic,4), topic_list]), ignore_index=True)
    #                 # 가장 비중이 높은 토픽과, 가장 비중이 높은 토픽의 비중과, 전체 토픽의 비중을 저장한다.
    #             else:
    #                 break
    #     return(topic_table)  
    # topictable = make_topictable_per_doc(ldamodel, corpus)
    # topictable = topictable.reset_index() # 문서 번호을 의미하는 열(column)로 사용하기 위해서 인덱스 열을 하나 더 만든다.
    # topictable.columns = ['문서 번호', '가장 비중이 높은 토픽', '가장 높은 토픽의 비중', '각 토픽의 비중']
    # topictable[:10]  
    # # 각 문서에서의 주제 비율 추출
    # topic_proportions = lda.transform(tfidf_matrix)
    
    # # 주제 비율을 데이터프레임으로 변환
    # df_topic_proportions = pd.DataFrame(topic_proportions)
    
    # n_top_words = 10 # 각 주제에서 상위 10개 단어를 확인
    # feature_names = vectorizer.get_feature_names_out()

    # topic_labels = []
    # for topic_idx, topic in enumerate(lda.components_):
    #     top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
    #     top_features = [feature_names[i] for i in top_features_ind]
    #     topic_labels.append(", ".join(top_features)) # 주제 라벨 생성
    # topic_labels = find_most_frequent_elements(topic_labels)
    # topic_labels = topic_labels[0].split(', ')
    # outdir = 'out/lda'
    # if not os.path.exists(outdir):
    #     os.makedirs(outdir)
    # filename = outdir + f'/{doc[1]}'
    
    
    # # 데이터프레임에서 각 문서의 주제 비율을 파이 차트로 그리기
    # for i in range(len(df_topic_proportions)):
    #     plt.figure(i)
    #     plt.pie(df_topic_proportions.iloc[i], labels=topic_labels)
    #     plt.title(f'Document {i+1} Topic Proportions')
    #     plt.savefig(f'{filename}.jpg')

