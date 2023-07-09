from sklearn.model_selection import cross_val_score
from sklearn import svm

def cvall(X, y, C_in, gamma_in, kernel_in):
    # clf = svm.SVC(C=1, degree=2, gamma='scale', kernel='sigmoid')
    # clf = svm.SVC(C=10, degree=2, gamma='scale', kernel='linear')
    clf = svm.SVC(C=C_in, degree=2, gamma=gamma_in, kernel=kernel_in)
    scores = cross_val_score(clf, X, y, cv=5)

    return scores, scores.mean()
