from sklearn.model_selection import train_test_split
from sklearn import svm

def svm_classification(X, y, C_in, gamma_in, kernel_in):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=y, random_state=42)

    classifierSVM = svm.SVC(C=C_in, degree=2, gamma=gamma_in, kernel=kernel_in)
    # training
    classifierSVM.fit(X_train, y_train)
    # prediksi data test

    y_pred_SVM = classifierSVM.predict(X_test)
    # return X_train, X_test, y_train, y_test, classifierSVM, y_pred_SVM
    return classifierSVM, y_pred_SVM, y_test