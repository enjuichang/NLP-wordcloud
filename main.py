import pandas as pd
import jieba
from ckiptagger import WS
import numpy as np
import itertools
from ArticutAPI import Articut
from glob import glob
import json
import os

pwd = os.getcwd()

# Ckiptagger 
ws = WS("data")

# Articut
# Input username and api key
with open("account.info", mode="r", encoding="utf-8") as file:
    useraccountdict = json.loads(file.read())
username = useraccountdict["username"]
apikey = useraccountdict["api_key"]

articut = Articut(username, apikey)

# cd
path_list = glob(f"{pwd}/NLP-wordcloud/text/*")

# Storage
wordfreq_dict = []
wordfreq_df = []
themefreq_df = []
total_text = []
wordfreq_total = []

# Read file
stop_words_tw = open(f"{pwd}/NLP-wordcloud/data/stopwords-tw.txt", "r")
stop_words_tw = stop_words_tw.read().split()

# Identify stop words and bad char
bad_chars = [';', ',', '.', '?', '!', '_', '[', ']', ':', '“', '”',
             '"', '-', '-',"“","\ufeff","\n"," ","；","。","：","，","⋯⋯","…","——","－",
             "！","？","(",")","受","受訪","做","訪問","者","訪談","受訪者","訪談者",
             "這樣子","訪問者","說","誒","訪","問者","其實","如果"]


with open(f"{pwd}/NLP-wordcloud/catDICT.json", encoding="utf-8") as f:
    cat_dict = json.loads(f.read())

def read_txt(file_path):
    '''
    Read text
    '''
    f = open(file_path, "r")
    txt = f.read()
    return txt

def jieba_cut_txt(txt):
    '''
    Cut text using jieba
    '''
    cut_text = jieba.cut(txt, cut_all=False)
    cut_list = list(cut_text)
    return cut_list

def ckip_cut_txt(txt):
    '''
    Cut text using ckip
    '''
    txt = txt.split('\n\n')
    cut_text = ws(txt)
    cut_list = list(itertools.chain.from_iterable(cut_text))
    return cut_list

def articut_cut_txt(txt, lv = "lv2"):
    '''
    Cut text using articut
    '''
    cut_ls = articut.parse(txt, level=lv)
    return cut_ls['result_segmentation'].split('/')


def articut_cut_txt_user(txt, lv = "lv2"):
    '''
    Cut text using articut
    '''
    cut_ls = articut.parse(txt, level=lv, userDefinedDictFILE= f"{pwd}/NLP-wordcloud/data/catDICT.json")
    return cut_ls['result_segmentation'].split('/')

def del_stop_words(cut_list,stop_words=stop_words_tw,bad_chars=bad_chars):
    '''
    Delete stop words and bad char
    '''
    return list(filter(lambda a: a not in stop_words and a not in bad_chars, cut_list))

def word_ListToFreq_Dict(wordlist):
    '''
    Transform word list to frequency dict
    '''
    wordfreq = [wordlist.count(p) for p in wordlist]
    return dict(list(zip(wordlist,wordfreq)))

def sortFreqDict(freqdict):
    '''
    Sort a dictionary of word-frequency pairs in
    order of descending frequency.
    '''
    aux = [(freqdict[key], key) for key in freqdict]
    aux.sort()
    aux.reverse()
    return aux

def extractFromDict(freqdict,cat_dict):
    '''
    Extract useful phrases from dict
    '''
    cat1 = [word for word in freqdict if word in cat_dict["專業知識"]]
    cat2 = [word for word in freqdict if word in cat_dict["困難與需求"]]
    cat3 = [word for word in freqdict if word in cat_dict["專業發展"]]
    cat4 = [word for word in freqdict if word in cat_dict["行政資源"]]
    cat5 = [word for word in freqdict if word in cat_dict["家庭百態"]]
    cat6 = [word for word in freqdict if word in cat_dict["人際處理"]]
    cat7 = [word for word in freqdict if word in cat_dict["環境"]]

    outputDict = {"專業知識": len(cat1),"困難與需求": len(cat2),"專業發展": len(cat3),
    "行政資源": len(cat4),"家庭百態": len(cat5),"人際處理": len(cat6),"環境": len(cat7)}

    return outputDict

for i, path in enumerate(path_list):
    print("File: ", i)

    # Read and cut text
    txt = read_txt(path)
    #cut_list = jieba_cut_txt(txt)
    #cut_list = ckip_cut_txt(txt)
    #cut_list = articut_cut_txt(txt)
    cut_list = articut_cut_txt_user(txt)

    # Delete stop words
    words = del_stop_words(cut_list)

    # Append words into frequncy dict
    total_text.append(words)

    # Create frequency dict
    wordfreq = word_ListToFreq_Dict(words)
    themefreq = extractFromDict(words,cat_dict)

    # Generate to df
    file_word = pd.DataFrame.from_dict(wordfreq, orient='index', columns = ["次數"])
    file_word = file_word.sort_values(by = ['次數'],ascending = False)
    file_theme = pd.DataFrame.from_dict(themefreq, orient='index', columns = ["次數"])
    file_theme = file_theme.sort_values(by = ['次數'],ascending = False)

    # Append to larger ls
    wordfreq_dict.append(wordfreq)
    wordfreq_df.append(file_word)
    themefreq_df.append(file_theme)

if __name__ == "__main__":
    # Ouput to csv
    for i, df in enumerate(wordfreq_df):
        df.to_csv(f"{pwd}/NLP-wordcloud/csv/{path_list[i][39:]}_df.csv")

    # Ouput theme to csv
    for i, df in enumerate(themefreq_df):
        df.to_csv(f"{pwd}/NLP-wordcloud/theme/{path_list[i][39:]}_theme.csv")

    # Concatenate into total text
    total_text = list(np.concatenate(np.asarray(total_text)))

    # Create frequency dict
    wordfreq_total = word_ListToFreq_Dict(total_text)
    themefreq_total = extractFromDict(total_text,cat_dict)

    # Generate df
    file_word = pd.DataFrame.from_dict(wordfreq_total, orient='index', columns=["次數"])
    file_theme = pd.DataFrame.from_dict(themefreq_total, orient='index', columns=["次數"])
    total_df = file_word.sort_values(by=['次數'], ascending=False)
    total_theme_df = file_theme.sort_values(by=['次數'], ascending=False)

    # Export
    total_df.to_csv(f"{pwd}/NLP-wordcloud/csv/total_df.csv")
    total_theme_df.to_csv(f"{pwd}/NLP-wordcloud/theme/total_theme.csv")