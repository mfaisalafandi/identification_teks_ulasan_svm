import pandas as pd
import numpy as np
import re
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# df = pd.read_csv("data/reviews.xlsx", delimiter=';', encoding='latin')
df = pd.read_excel("data/reviews.xlsx")
# texts = df["cerita"].tolist()
# labels = df["genre"].tolist()

data = df[['Reviews', 'Label']]
data.head()

list(data.columns.values)

number_of_cerita = data.Reviews.count()
data_counts = df.Label.value_counts()

print(number_of_cerita)
print(data_counts)

data['Case_Folding'] = df['Reviews'].str.lower()
data

import string

def cleansing(txt):
    txt = txt.encode('ascii', 'replace').decode('ascii')
    txt = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", txt).split())
    txt = re.sub(r"\d+", "", txt).replace('?', ' ').replace(',', ' ')
    txt = txt.translate(str.maketrans("-"," ",string.punctuation))
    txt = txt.strip()
    txt = re.sub('\s+',' ',txt)
    txt = re.sub(r"\b[a-zA-Z]\b", "", txt)
    return txt

data['Cleansing'] = data['Case_Folding'].apply(cleansing)
data
    

data['Tokenize'] = data['Cleansing'].str.split()
data

spill_words = {
    "": ["aaaa"],
    "kecil": ["kecillll"],
    "nangis": ["nngisssss", "huhuuuu"],
    "si": ["siiiii"],
    "bohong": ["bhong"],
    "bagus": ["baklgus", "bagusssssssssssssssssssssssss", "bgaaaaaaaaus", "bagusssssssss", "bagusssssss", "baaaaaguuuuussss"],
    "banget": ["bangetttttttttt", "bangggeetttt", "bangttt"],
    "pokoknya": ["pokoknyaaaaa"],
    "yaudah": ["ydh"],
    "makasih": ["thxx"],
    "seler": ["sellerr"],
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
    "seller": ["sellerrrr"],
    "lucu": ["gemoyyyyyy"],
    "cinta": ["luvvv"],
    "cantik": ["wapikk"],
    "harga": ["hargae"],
    "mahal": ["mehongg"],
    "sahabat": ["bestii"],
    "bunga": ["bukett"],
    "murah": ["murceeee"],
    "lama": ["lamaaaaaaaaaa"],
    "ya": ["yaaaa"],
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

def spilling(tokens):
    for i, line_token in enumerate(tokens):
        stop = 0
        for key in spill_words:
            for sp in spill_words[key]:
                if(line_token == sp):
                    tokens[i] = key
                    stop = 1
                    break
            if(stop == 1) : break
    return tokens

normalization_word = pd.read_csv('ynb/colloquial-indonesian-lexicon.csv')
normalization_word_dict = {}

for index, row in normalization_word.iterrows():
    if row[0] not in normalization_word_dict:
        normalization_word_dict[row[0]] = row[1]

def normalization(tokens):
    return [normalization_word_dict[term] if term in normalization_word_dict else term for term in tokens]

data['Normalization'] = data['Tokenize'].apply(normalization)

data['Normalization'] = data['Normalization'].apply(spilling)

from nltk.tokenize.treebank import TreebankWordDetokenizer
data['Normalization'] = data['Normalization'].apply(lambda x: TreebankWordDetokenizer().detokenize(x))
data

for i in data['Normalization']:
    print(i);


from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
def stemming(txt):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    stem = stemmer.stem(txt)
    print(stem)
    return stem

data['Stemming'] = data['Normalization'].apply(lambda txt: stemming(txt))
data


from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

def stopwords_removal(txt):
    factory = StopWordRemoverFactory()
    stopword = factory.create_stop_word_remover()
    filter = stopword.remove(txt)
    return filter

data['Stopword'] = data['Stemming'].apply(lambda txt: stopwords_removal(txt))
data


data = pd.read_excel("preprocessing_review.xlsx")
data


import sklearn
sklearn.__version__

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(data['Stopword']).toarray()
y = data.iloc[:, 1].values
print(X)

from sklearn.feature_selection import SelectKBest, chi2

# Menggunakan 100 fitur terbaik berdasarkan metode Chi-square
# selector = SelectKBest(chi2, k=int(0.1 * X.shape[1])) # 93%
# selector = SelectKBest(chi2, k=int(0.2 * X.shape[1])) # 95%
# selector = SelectKBest(chi2, k=int(0.3 * X.shape[1])) # 93%
selector = SelectKBest(chi2, k=int(0.4 * X.shape[1])) # 94%
X = selector.fit_transform(X, y)


from sklearn.feature_extraction.text import TfidfTransformer
tfidfconverter = TfidfTransformer()
X = tfidfconverter.fit_transform(X).toarray()
print(X)

# Tampilkan matriks TF-IDF dengan term-nya
# for row in X:
#     print(" ".join("{:.4f}".format(value) for value in row))


from sklearn.model_selection import cross_val_score
from sklearn import svm

# clf = svm.SVC(C=1, degree=2, gamma='scale', kernel='sigmoid')
# clf = svm.SVC(C=10, degree=2, gamma='scale', kernel='linear')
clf = svm.SVC(C=10, degree=2, gamma=0.1, kernel='rbf')
scores = cross_val_score(clf, X, y, cv=5)

print(scores)
scores.mean()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=y, random_state=42)

# Klasifikasi SVM

from sklearn import svm

classifierSVM = svm.SVC(C=10, degree=2, gamma=0.1, kernel='rbf')
# classifierSVM = svm.SVC(C=10, degree=2, gamma='scale', kernel='linear')
# classifierSVM = svm.SVC(C=1, degree=2, gamma=0.001, kernel='linear')
# training
classifierSVM.fit(X_train, y_train)
# prediksi data test

y_pred_SVM = classifierSVM.predict(X_test)
print(y_pred_SVM)

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred_SVM))

from sklearn.metrics import precision_score, recall_score, f1_score

labels = [0, 1]

pre = precision_score(y_test, y_pred_SVM, average='macro', labels=labels)
print("Precision : ", pre)

rec = recall_score(y_test, y_pred_SVM, average='macro', labels=labels)
print("Recall : ", rec)

f1 = f1_score(y_test, y_pred_SVM, average='macro', labels=labels)
print("F1-Score : ", f1)
print()
pre = precision_score(y_test, y_pred_SVM, average='weighted', labels=labels)
print("Precision : ", pre)

rec = recall_score(y_test, y_pred_SVM, average='weighted', labels=labels)
print("Recall : ", rec)

f1 = f1_score(y_test, y_pred_SVM, average='weighted', labels=labels)
print("F1-Score : ", f1)


import pickle

filename = "model/SVM_10_0-1_rbf.pkl"

with open(filename, 'wb') as file:
    pickle.dump(classifierSVM, file)

filepath = "model/SVM_10_0-1_rbf.pkl"
with open(filepath, 'rb') as file:
    SVM_model = pickle.load(file)

text = np.array([
    'Produk yang saya beli dari toko online ini sangat keren', 
    'Saya sangat puas dengan pengiriman produk ini Waktu pengirimannya cepat dan packingnya sangat rapi.', 
    'Produk ini benar-benar memenuhi harapan saya. Kualitasnya sangat baik, harga terjangkau, dan layanan pelanggan yang responsif.', 
    'Saya sangat kecewa dengan produk yang saya beli. Ketika tiba, barangnya dalam kondisi rusak dan tidak sesuai dengan deskripsi yang diberikan.', 
    'Pengiriman produk ini sangat lambat. Saya harus menunggu berhari-hari untuk mendapatkan barangnya, padahal saya telah membayar biaya pengiriman yang mahal.', 
    'Saya merasa tertipu oleh gambar produk yang ditampilkan. Ketika produk tiba, warna dan kualitasnya jauh berbeda dari yang ditampilkan di situs jual beli online.'
    ], dtype=object)

sample = cv.transform(text)
sample_selected = selector.transform(sample).toarray()
sample_selected = tfidfconverter.transform(sample_selected).toarray()

predicted = classifierSVM.predict(sample_selected)
print(predicted)


