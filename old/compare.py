import sparse_interaction
from sklearn.datasets import fetch_20newsgroups_vectorized
from sklearn.cross_validation import StratifiedKFold
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score

dat = fetch_20newsgroups_vectorized()
X = dat.data
Y = dat.target
cv = StratifiedKFold(Y)

X = X[:, :20000]

si = sparse_interaction.SparseInteractionFeatures()
X_i = si.transform(X)

scores, scores_i = [], []

clf = SGDClassifier(penalty='l1', n_iter=10)

for train, test in cv:
    clf.fit(X[train], Y[train])
    scores.append(f1_score(Y[test], clf.predict(X[test]), average='macro', pos_label=None))
    clf.fit(X_i[train], Y[train])
    scores_i.append(f1_score(Y[test], clf.predict(X_i[test]), average='macro', pos_label=None))
print sum(scores), sum(scores_i)