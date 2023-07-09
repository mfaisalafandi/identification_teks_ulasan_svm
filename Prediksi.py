from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def sentiment(text, cv, selector, tfidfconverter, classifierSVM):
    sample = cv.transform(text)
    sample_selected = selector.transform(sample).toarray()
    sample_selected = tfidfconverter.transform(sample_selected).toarray()

    predicted = classifierSVM.predict(sample_selected)
    return predicted