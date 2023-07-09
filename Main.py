from flask import Flask
from flask import request
from flask import render_template, Flask
from flask_table import Table, Col

import pandas as pd
import numpy as np
import re
import nltk
import pickle

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn import svm

import Preprocessing
import Cross_vall
import Klasifikasi
import Prediksi

def read_data():
    # mengambil data pada file excel
    df = pd.read_excel("data/reviews.xlsx")
    
    data = df[['Reviews', 'Label']]

    list(data.columns.values)
    number_of_cerita = data.Reviews.count()
    data_counts = df.Label.value_counts()
    print(number_of_cerita)
    print(data_counts)
    return df, data

def cheat_pre():
    return pd.read_excel("data/preprocessing_review.xlsx")

def use_init():
    df, data = read_data();
    data_set = df['Reviews'].values.tolist()

    # case_folding = Preprocessing.casefolding(data_set); # melakukan case folding dan cleansing
    # tokens = Preprocessing.tokenizing(case_folding) # melakukan tokenisasi
    # tokens_sp = Preprocessing.normalisasi(Preprocessing.spilling(tokens)) # melakukan perubahan kata yg tidak tepat
    # stemm = Preprocessing.stemming(tokens_sp) # melakukan stemming
    # filter = Preprocessing.filtering(tokens_sp) # melakukan filter
    # data['Case_Folding'] = pd.Series(case_folding)
    # data['Tokenize'] = pd.Series(tokens)
    # data['Normalization'] = pd.Series(tokens_sp)
    # data['Stemming'] = pd.Series(stemm)
    # data['Stopword'] = pd.Series(filter)

    data = cheat_pre()
    
    cv = CountVectorizer()
    X = cv.fit_transform(data['Stopword']).toarray()
    y = data.iloc[:, 1].values

    # selector = SelectKBest(chi2, k=int(0.1 * X.shape[1])) # 93%
    # selector = SelectKBest(chi2, k=int(0.2 * X.shape[1])) # 95%
    # selector = SelectKBest(chi2, k=int(0.3 * X.shape[1])) # 93%
    selector = SelectKBest(chi2, k=int(0.4 * X.shape[1])) # 94%
    X = selector.fit_transform(X, y)

    tfidfconverter = TfidfTransformer()
    X = tfidfconverter.fit_transform(X).toarray()

    return df, data, cv, X, y, selector, tfidfconverter

def use_train(X, y, C_in=10, gamma_in=0.1, kernel_in='rbf'):
    # Grid Search
    # svm = svm.SVC()
    # param = {
    #     'C': (0.1, 1, 10, 100),
    #     'kernel': ('linear', 'rbf', 'sigmoid', 'poly'),
    #     'gamma': (0.0001, 0.001, 0.1, 1, 'scale', 'auto'),
    #     'degree': (2, 3, 4)
    #     }
    # grid = GridSearchCV(svm, param)
    # grid.fit(X,y)
    # print(grid.best_params_)
    # print(grid.best_score_)

    score, mean_score = Cross_vall.cvall(X, y, C_in, gamma_in, kernel_in)
    print(C_in, " | ", gamma_in, " | ", kernel_in)
    print(score)
    print(mean_score)
    print("========================")

    return Klasifikasi.svm_classification(X, y, C_in, gamma_in, kernel_in)

def use_model(fl):
    filepath = "model/" + fl
    with open(filepath, 'rb') as file:
        classifierSVM = pickle.load(file)

def __MAIN__():
    # Menggunakan modul flask untuk menampilkan aplikasi berbasis web
    app = Flask(__name__, template_folder='templates')

    df, data, cv, X, y, selector, tfidfconverter = use_init()

    # =======================================
    # Menjalankan model SVM dari training
    # =======================================
    classifierSVM, y_pred_SVM, y_test = use_train(X, y, 10, 0.1, 'rbf')
    # ---------------------------------------
    
    # =======================================
    # SVM_ C = 10_ gamma = 0.1_ kernel = rbf.pkl
    # SVM_10_0-1_rbf.pkl
    # =======================================
    # classifierSVM = use_model("SVM_10_0-1_rbf.pkl")
    # ---------------------------------------
    
    # Link routing pertama kali aplikasi dibuka
    @app.route('/')
    def index():
        return render_template('index.html')
    
    @app.route('/cek', methods=("POST", "GET"))
    def cek():
        if (request.method == "POST") and (request.form["in_kal"] != ""):
            # Membuat array numpy dengan input pengguna
            kalimat = np.array([request.form["in_kal"]], dtype=object)
            
            pred = Prediksi.sentiment(kalimat, cv, selector, tfidfconverter, classifierSVM)

            return render_template('cek.html', kal=request.form['in_kal'], result=pred)
        else:
            return render_template('cek.html')
    
    @app.route('/team')
    def team():
        return render_template('team.html')
    
    @app.route('/hype')
    def hype():
        classifierSVM1, y_pred_SVM1, y_test1 = use_train(X, y, 100, 0.0001, 'rbf')
        classifierSVM2, y_pred_SVM2, y_test2 = use_train(X, y, 1, 0.1, 'rbf') #
        classifierSVM3, y_pred_SVM3, y_test3 = use_train(X, y, 10, 0.1, 'rbf') #
        classifierSVM4, y_pred_SVM4, y_test4 = use_train(X, y, 100, 0.1, 'rbf') #
        classifierSVM5, y_pred_SVM5, y_test5 = use_train(X, y, 10, 0.001, 'rbf') #

        classifierSVM6, y_pred_SVM6, y_test6 = use_train(X, y, 100, 0.1, 'poly') #
        classifierSVM7, y_pred_SVM7, y_test7 = use_train(X, y, 10, 0.1, 'poly')
        classifierSVM8, y_pred_SVM8, y_test8 = use_train(X, y, 1, 0.1, 'poly')
        
        return render_template('hyper.html', 
                               class_report1=pd.DataFrame(classification_report(y_test1, y_pred_SVM1, output_dict=True)).transpose().to_html(classes='table table-hover'),
                               class_report2=pd.DataFrame(classification_report(y_test2, y_pred_SVM2, output_dict=True)).transpose().to_html(classes='table table-hover'),
                               class_report3=pd.DataFrame(classification_report(y_test3, y_pred_SVM3, output_dict=True)).transpose().to_html(classes='table table-hover'),
                               class_report4=pd.DataFrame(classification_report(y_test4, y_pred_SVM4, output_dict=True)).transpose().to_html(classes='table table-hover'),
                               class_report5=pd.DataFrame(classification_report(y_test5, y_pred_SVM5, output_dict=True)).transpose().to_html(classes='table table-hover'),
                               class_report6=pd.DataFrame(classification_report(y_test6, y_pred_SVM6, output_dict=True)).transpose().to_html(classes='table table-hover'),
                               class_report7=pd.DataFrame(classification_report(y_test7, y_pred_SVM7, output_dict=True)).transpose().to_html(classes='table table-hover'),
                               class_report8=pd.DataFrame(classification_report(y_test8, y_pred_SVM8, output_dict=True)).transpose().to_html(classes='table table-hover')
                               )
    # class_report=pd.DataFrame(classification_report(y_test, y_pred_SVM, output_dict=True)).transpose().to_html()
    
    @app.route('/pre')
    def pre():
        table = data.to_html(classes='table table-hover', index=False)
        return render_template('preprocessing.html', table=table)
    
    app.run(debug=True)

__MAIN__();