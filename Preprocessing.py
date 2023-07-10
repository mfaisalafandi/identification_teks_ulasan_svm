import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.probability import FreqDist
import re
import string
import pandas as pd

from nltk.tokenize.treebank import TreebankWordDetokenizer

spill_words = {
    "": ["aaaa", "hsbdjdkwjehdbejwwjbdbqjajhdhdjsjwjjwjwjbd"],
    "kecil": ["kecillll"],
    "nangis": ["nngisssss", "huhuuuu"],
    "si": ["siiiii"],
    "bohong": ["bhong"],
    "bagus": ["baklgus", "bagusssssssssssssssssssssssss", "bagusssssssssssssssssssssssssssss", "bgaaaaaaaaus", "bagusssssssss", "bagusssssss", "baaaaaguuuuussss"],
    "banget": ["bangetttttttttt", "bangggeetttt", "bangttt"],
    "pesanan": ["pseanan"], 
    "baik": ["baikp"], 
    "pokoknya": ["pokoknyaaaaa"],
    "yaudah": ["ydh"],
    "makasih": ["thxx"],
    "seler": ["sellerr", "sellererrrrr"],
    "hihi": ["xixi", "xixixi", "wkwk"],
    "butuh": ["butuhhhh"],
    "kependekan": ["kependekannnnnn"],
    "gila": ["gilaaaakkkkkk"],
    "halo" : ["woeeeeee", "woiiiiiiiiiiii"],
    "parah": ["parahhhh"],
    "cuman": ["cumannn"],
    "sih": ["sihhhhh"],
    "suka": ["sukakkkkkk"],
    "oke": ["okeeeeeeeeeeeeeeeeeeeeeeeeeee"],
    "adem": ["ademmmmnnnn"],
    "seller": ["sellerrrr", "sellererrrrr"],
    "deskripsi": ["deskripsimkshhhh"], 
    "lucu": ["gemoyyyyyy"],
    "cinta": ["luvvv"],
    "cantik": ["wapikk"],
    "harga": ["hargae"],
    "mahal": ["mehongg"],
    "sahabat": ["bestii"],
    "bunga": ["bukett"],
    "murah": ["murceeee"],
    "lama": ["lamaaaaaaaaaa"],
    "makasih": ["makasiiiiiiiiiihhhhhhh"], 
    "ya": ["yaaaa"],
    "yang": ["yg"], 
    "di": ["d"], 
    "selalu": ["sll"], 
    "banget": ["bgt", "bangttt"],
    "aku": ["gue", "gw"], 
    "tidak": ["gk", "ga", "gak"],
    "cuma": ["cuman"],
    "aduh": ["deg"],
    "apa": ["papa"],
    "taruh": ["taro"],
    "nya": ["y"],
    "pas": ["ps"],
    "sangat": ["sangatttt"],
    "dengan": ["dg"],
    "tahu": ["tau"],
    "itu": ["tuh"],
    "kakak": ["kak"],
    "bagaimana": ["gimana"],
    "benar": ["bner"],
    "lah": ["loh", "lho"],
    "banget": ["bgttt", "bgtttt"],
    "padahal": ["pdhl"],
    "aku": ["ku"],
    "juga": ["jg"],
    "sayang": ["syg"],
    "lebar": ["wide"],
    "kaki": ["leg"],
    "yang": ["yg"],
    "merek": ["merk"],
    "sudah": ["udah"]
}

def casefolding(data_set):
    set_new = []
    for i, data in enumerate(data_set):
        new_data = data.lower() # mengecilkan huruf menjadi lowercase
        new_data = re.sub(r"\d+", "", new_data).replace('?', ' ').replace(',', ' ').replace('.', ' ')
        new_data = new_data.translate(str.maketrans("", "", string.punctuation)).strip() # menghapus karakter tanda baca & whitespace
        new_data = re.sub(r'[^(a-z|A-Z|0-9)| |]', '', new_data) # mengkhususkan karakter tertentu, sehingga icon terhapus
        new_data = re.sub('\s+',' ',new_data)
        set_new.append(new_data)        
    return set_new

# melakukan tokenisasi
def tokenizing(data_set):
    tokens = []
    for i, data in enumerate(data_set):
        tokens.append(nltk.tokenize.word_tokenize(data))
    return tokens;

def spilling(tokens):
    for i, line_token in enumerate(tokens):
        for j, token in enumerate(line_token):
            stop = 0
            for key in spill_words:
                for sp in spill_words[key]:
                    if(token == sp):
                        tokens[i][j] = key
                        stop = 1
                        break
                if(stop == 1) : break
    return tokens

def normalisasi(data_out):
    normalization_word = pd.read_csv('data/colloquial-indonesian-lexicon.csv')
    normalization_word_dict = {}
    for row in normalization_word:
        if row[0] not in normalization_word_dict:
            normalization_word_dict[row[0]] = row[1]

    def normalization(token):
        return [normalization_word_dict[term] if term in normalization_word_dict else term for term in token]
    
    normalized_data = []
    for tokens in data_out:
        normalized_tokens = normalization(tokens)
        normalized_data.append(normalized_tokens)
    
    return normalized_data

def stemming(tokens):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    list_stem = []
    for line_token in tokens:
        tmp_stem = []
        for t in line_token:
            tmp_stem.append(stemmer.stem(t))
        list_stem.append(tmp_stem)
    return list_stem

def filtering(tokens):
    listStopWords =  set(stopwords.words('indonesian'))
    removed = []
    for line_token in tokens:
        tmp_remove = []
        for t in line_token:
            if t not in listStopWords:
                tmp_remove.append(t)
        removed.append(tmp_remove)
    return removed