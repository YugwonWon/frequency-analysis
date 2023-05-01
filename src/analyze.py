import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import itertools
from collections import Counter
from wordcloud import WordCloud

from bareunpy import Tagger

import networkx as nx
import os
# import pyLDAvis.gensim 
# import gensim.corpora as corpora
# import gensim
# from gensim.models.ldamodel import LdaModel

# import matplotlib.font_manager as fm
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
        
    def wordcloud_Rank(self, token, N = 30, white = True, title = 'out'):
        if not os.path.exists(title + '/Wordcloud'):
            os.makedirs(title + '/Wordcloud')
        filename = title + '/Wordcloud/keyword_wordcloud.png'
        filename_csv = title + '/Wordcloud/keyword_count.csv'
        
        flatten_tokens = list(itertools.chain(*token))
        token_count = Counter(flatten_tokens)
        Top_token_count = token_count.most_common(N)
        
        word_count = pd.DataFrame(Top_token_count)
        word_count.columns = ['Word', 'Count']
        word_count.to_csv(filename_csv, index = False, encoding='euc-kr')
        
        self.cloud_visulize(Top_token_count, filename, white)
        return word_count
    
    def Top_Keyword_Edgelist(self, Token, Edge_list, N, title = '', node_size = 50, edge_size = 10):
        if not os.path.exists(title + '/Graph'):
            os.makedirs(title + '/Graph')
        filename = title + '/Graph/Keyword_Network.png'
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
        
        plt.figure(figsize=(20,20))
        pos = nx.spring_layout(G_top, k = 0.5)
        nx.draw(G_top, pos, width=weights, node_color= 'yellow', node_size = degree, edge_color= 'green', edge_cmap=plt.cm.Reds)
        nx.draw_networkx_labels(G_top, pos, font_size=32, font_family='AppleGothic', font_color='black')
        plt.savefig(filename)
        
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
        tagger = Tagger('api-key', 'localhost', port=5656)
        tokenized_lines = []
        for line in tqdm(lines):
            tagged = tagger.tags([line])
            nouns = []
            for p in tagged.pos():
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
    

if __name__ == '__main__':
    # 데이터 로딩
    texts = []
    with open('data/summary.txt', 'r', encoding='utf-8') as f:
        for line in f:
            texts.append(line.strip())
            
    # 빈도 분석 객체 생성
    fa = FrequencyAnalysis()
    
    # 형태소 분석 명사 추출
    Tokens = fa.preprocess_text(texts)
    
    # 워드클라우드 생성
    word_count = fa.wordcloud_Rank(Tokens, 50, white = True, title = 'out')
    print(word_count.head(10))
    
    # 키워드 네트워크
    Edge_list = fa.make_Edgelist(Tokens)
    fa.Top_Keyword_Edgelist(Tokens, Edge_list, 25, title = 'out', node_size = 50, edge_size = 6)
    