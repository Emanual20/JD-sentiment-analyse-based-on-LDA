# Author: Group 9 untitled
# Date: 2020/7/12 - 2020/7/18

import requests
import bs4
from pandas.core.frame import DataFrame
import re
import pandas as pd
import jieba
import define as df


if __name__ == '__main__':
    # 数据爬取
    # maxpage = eval(input('需要爬取的商品个数'))
    # item_name = input('需要爬取的商品名称')
    # url1 = 'https://search.jd.com/Search?keyword=' + item_name + '&enc=utf-8&wq=' + item_name
    # r = requests.get(url=url1, headers=df.HEADER)
    # soup = bs4.BeautifulSoup(r.text, "html5lib")
    # list1 = soup.find_all('i', {'class': "promo-words"})
    # for i in range(maxpage):
    #     try:
    #         print("正在爬取第" + str(i + 1) + "件商品评论")
    #         tag = list1[i]
    #         id = tag['id'][5:]
    #         df.get_data(id, item_name, 'Result.csv')
    #     except:
    #         print("爬取失败")
    # print("爬取完成")

    # 网路语义分析
    df.network()
    print("网路语义分析完成")

    # 数据清洗部分
    # 朴素去重
    s = pd.read_csv(df.SPIDER_PATH, encoding=df.DECODE, header=None)
    len1 = len(s)
    data_processed = pd.DataFrame(s[0].unique())
    len2 = len(data_processed)
    print("原有%d条评论" % len1)
    print("现有%d条评论" % len2)
    print("(删除了%d条重复评论)" % (len1 - len2))
    data_processed_list = data_processed[0].values.tolist()
    print("去重完成")

    # 调用机械压缩
    data_compressed_list = df.machine_compressed(data_processed_list)
    # print(data_compressed_list)
    print("机械压缩完成")

    # 短词删除
    data_final_list = []
    for data in data_compressed_list:
        strdata = str(data)
        strdata = re.sub("[\s+\.\!V,$%^*(+\"\"]+|[+!,.?、~@#$%......&*();`:]+", "", strdata)
        if len(strdata) <= 4:
            pass
        else:
            data_final_list.append(strdata)

    # 文本评论分词
    print("====正在分词====")
    data_final = []
    for data in data_final_list:
        cut = jieba.cut(data, cut_all=False, HMM=False)
        data_final.append(' '.join(cut))

        '''另外两种分词模式，可以放在模型比较中来比较效果
        cut = jieba.cut(s, cut_all=True)
        print(','.join(cut))

        cut = jieba.cut(s, cut_all=True, HMM=True)
        print(','.join(cut))'''
    # print(data_final_list)

    # 分词
    final_df = DataFrame(data_final_list)
    final_df.to_csv(df.COMPRESSED_PATH, encoding=df.DECODE, header=None)

    # 情感分析模型--SnowNlp
    print("====SnowNlp情感分析====")
    dic = df.sentiment_analysis(data_final_list)
    good = []
    bad = []
    k = list(dic.keys())
    for sentence in k:
        if dic[sentence] > 0.90:
            good.append(sentence)
        else:
            bad.append(sentence)
    print(good)
    print(bad)

    # LDA主题分析
    print("====LDA====")
    print("正面主题分析")
    df.LDA(good, 0)
    print("============")
    print("负面主题分析")
    df.LDA(bad, 1)

    # 利用ROST前的数据变换，将csv转换成txt
    print("====正在转换为txt====")
    df.transform('Result.csv')
    print("运行完成。请在本文件目录下获取生成的文件。")