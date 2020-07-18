# Author: Group 9 untitled
# Date: 2020/7/12 - 2020/7/18

from urllib import request
import json
from snownlp import SnowNLP
import re
import csv
import collections
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import jieba
import operator
from gensim import corpora, models
import math

SPIDER_PATH = 'Result.csv'
COMPRESSED_PATH = 'Compressedresult.csv'
WORDCLOUDSOURCE_PATH = 'Wordcloudsource'
HEADER = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/'
                            '65.0.3325.1''81 Safari/537.36'}
URL = "https://club.jd.com/comment/productPageComments.action?callback=fetchJSON_comment98&productId={productId}" \
        "&score={score}&sortType=5&page={page}&pageSize=10&isShadowSku=0&rid=0&fold=1"
DECODE = "GBK"


def fetch_json(productId, page, score) -> dict:
    url = URL.format(
        productId=str(productId),
        page=str(page),
        score=str(score),
    )
    req = request.Request(url, headers=HEADER)
    rep = request.urlopen(req).read().decode(DECODE, errors="ignore")
    dict_ = re.findall(r"\{.*\}", rep)[0]
    return json.loads(dict_)


def get_data(productId, sheetName, fileName):
    all_comment = 0
    p = 1
    sum = 1
    while sum > 0:
        sum = 0
        comment_dict = fetch_json(productId, page=str(p), score=0)
        p += 1
        for i in comment_dict["comments"]:
            sum += 1
            write_col = [i['content']]
            df = pd.DataFrame(columns=(write_col))
            df.to_csv(fileName, line_terminator="\n", index=False, mode='a', encoding='gb18030')
            all_comment += 1


# 机械压缩部分函数
def judge_repeat(L1, L2):
    if len(L1) != len(L2):
        return False
    else:
        return operator.eq(L1, L2)


def machine_compressed(commentList):
    L1 = []
    L2 = []
    compressList = []
    for letter in commentList:
        if len(L1) == 0:
            L1.append(letter)
        else:
            if L1[0] == letter:
                if len(L2) == 0:
                    L2.append(letter)
                else:
                    if judge_repeat(L1, L2):
                        L2.clear()
                        L2.append(letter)
                    else:
                        compressList.extend(L1)
                        compressList.extend(L2)
                        L1.clear()
                        L2.clear()
                        L1.append(letter)

            else:
                if judge_repeat(L1, L2) and len(L2) >= 2:
                    compressList.extend(L1)
                    L1.clear()
                    L2.clear()
                    L1.append(letter)
                else:
                    if len(L2) == 0:
                        L1.append(letter)
                    else:
                        L2.append(letter)
    else:
        if judge_repeat(L1, L2):
            compressList.extend(L1)
        else:
            compressList.extend(L1)
            compressList.extend(L2)
    L1.clear()
    L2.clear()
    return compressList


def sentiment_analysis(datalist):
    sentiment_dic = {}
    for text in datalist:
        s = SnowNLP(text)
        sentiment_dic[text] = s.sentiments
        # print(text[:10] + " {}".format(s.sentiments))
    return sentiment_dic


def network():
    s = pd.read_csv('Result.csv', encoding=DECODE, header=None)
    data_processed = pd.DataFrame(s[0].unique())
    string_data = ''.join(list(data_processed[0]))
    num = 40
    G = nx.Graph()
    plt.figure(figsize=(20, 14))
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.sans-serif'] = ['SimHei']
    pattern = re.compile(u'\t|\。|，|：|；|！|\）|\（|\?|"')
    string_data = re.sub(pattern, '', string_data)
    seg_list_exact = jieba.cut(string_data, cut_all=False)
    object_list = []
    stop_words = list(open('stopwords.txt', 'r', encoding='utf-8').read())
    stop_words.append("\n")
    for word in seg_list_exact:
        if word not in stop_words:
            object_list.append(word)
    word_counts = collections.Counter(object_list)
    word_counts_top = word_counts.most_common(num)
    word = pd.DataFrame(word_counts_top, columns=['关键词', '次数'])
    word_T = pd.DataFrame(word.values.T, columns=word.iloc[:, 0])
    net = pd.DataFrame(np.mat(np.zeros((num, num))), columns=word.iloc[:, 0])
    k = 0
    object_list2 = []
    for i in range(len(string_data)):
        if string_data[i] == '\n':
            seg_list_exact = jieba.cut(string_data[k:i], cut_all=False)
            for words in seg_list_exact:
                if words not in stop_words:
                    object_list2.append(words)
            k = i + 1
    word_counts2 = collections.Counter(object_list2)
    word_counts_top2 = word_counts2.most_common(num)
    word2 = pd.DataFrame(word_counts_top2)
    word2_T = pd.DataFrame(word2.values.T, columns=word2.iloc[:, 0])
    relation = list(0 for x in range(num))
    for j in range(num):
        for p in range(len(word2)):
            if word.iloc[j, 0] == word2.iloc[p, 0]:
                relation[j] = 1
                break
    for j in range(num):
        if relation[j] == 1:
            for q in range(num):
                if relation[q] == 1:
                    net.iloc[j, q] = net.iloc[j, q] + word2_T.loc[1, word_T.iloc[0, q]]
    n = len(word)
    for i in range(n):
        for j in range(i, n):
            G.add_weighted_edges_from([(word.iloc[i, 0], word.iloc[j, 0], net.iloc[i, j])])
    minwidth = min([v['weight'] for (r, c, v) in G.edges(data=True)])
    minsize = min([net.iloc[i, i] for i in np.arange(20)])
    # try:
    #     nx.draw_networkx(G, pos=nx.shell_layout(G),
    #                  width=[float((v['weight'] - minwidth) / 300) for (r, c, v) in G.edges(data=True)],
    #                  edge_color=np.arange(len(G.edges)),
    #                  node_size=[float((net.iloc[i, i] - minsize) * 3) for i in np.arange(20)],
    #                  node_color=[(0.2, 0.2, i/20) for i in np.arange(40)])
    # except ValueError:
    colors = [(216, 0, 15), (221.25, 27.25, 11.25), (226.5, 54.5, 7.5), (231.75, 81.75, 3.75), (237, 109, 0),
              (241.5, 142, 0), (246, 175, 0), (250.5, 208, 0), (255, 241, 0),
              (240.25, 234.5, 0), (225.5, 228, 0), (210.75, 221.5, 0), (196, 215, 0),
              (152.25, 204.75, 25.75), (108.5, 194.5, 51.5), (64.75, 184.25, 77.25), (21, 174, 103),
              (15.75, 171.5, 119), (10.5, 169, 135), (5.25, 166.5, 151), (0, 164, 167),
              (0, 146.25, 168.25), (0, 128.5, 169.5), (0, 110.75, 170.75), (0, 93, 172),
              (4.75, 85.75, 167), (9.5, 78.5, 162), (14.25, 71.25, 157), (19, 64, 152),
              (50.75, 63.25, 150.5), (82.5, 62.5, 149), (114.25, 61.75, 147.5), (146, 61, 146),
              (137.25, 52, 137.25), (128.5, 43, 128.5), (119.75, 34, 119.75), (111, 25, 111),
              (137.25, 18.75, 87), (163.5, 12.5, 63), (189.75, 6.25, 39)]
    nx.draw_networkx(G, pos=nx.shell_layout(G),
                         width=[float((v['weight'] - minwidth) / 300) for (r, c, v) in G.edges(data=True)],
                         edge_color=(34/255, 174/255, 230/255),
                         node_size=[float((net.iloc[i, i] - 100) * 3) for i in np.arange(40)],
                         node_color=[(r/255, g/255, b/255) for (r,g,b) in colors])
    plt.axis('off')
    plt.savefig("NetWork.png")
    # plt.show()


def LDA(data_final_list, mark):
    data_final = []
    for data in data_final_list:
        cut = jieba.cut(data, cut_all=False, HMM=False)
        data_final.append(' '.join(cut))
    pddata = pd.DataFrame(data_final)
    # print(pddata[0])
    with open('stopwords.txt', 'r', encoding='utf-8') as f:
        stopwords = list(f.readlines())
        # print(stopwords)
    stopword = []
    for i in stopwords:
        stopword.append(i[:-1])
    # print(stopword)
    stopword = [' ', '', '　'] + list(stopword)
    pddata[1] = pddata[0].apply(lambda s: s.split(' '))
    pddata[2] = pddata[1].apply(lambda x: [i for i in x if i not in stopword])
    # print(pddata[2])

    dictionary = corpora.Dictionary(pddata[2])
    corpus = [dictionary.doc2bow(data) for data in pddata[2]]
    model = models.LdaModel(corpus, id2word=dictionary, iterations=500, num_topics=7, alpha='auto')

    if mark == 0:
        out = open(WORDCLOUDSOURCE_PATH + 'posi.csv', 'a', newline='', encoding='utf-8')
    else:
        out = open(WORDCLOUDSOURCE_PATH + 'nega.csv', 'a', newline='', encoding='utf-8')
    csv_write = csv.writer(out, dialect='excel')

    for i in range(7):
        print(model.print_topic(i))
        split_list = model.print_topic(i, 14).split('+')
        for temp_str in split_list:
            weight_and_word = temp_str.split('*')
            times_now = int(eval(weight_and_word[0])*1000)
            for j in range(times_now):
                csv_write.writerow([weight_and_word[1][1:-2]])


def transform(filename):
    s = pd.read_csv(filename, encoding='gb18030', header=None)
    data_processed = pd.DataFrame(s[0].unique())
    string_data = ''.join(list(data_processed[0]))
    file = open("rost.txt", 'a', encoding='ansi')
    for text in string_data:
        file.write(text)
    file.close()

network()