from numpy import *
from Ch04 import bayes

listOPosts, listClasses = bayes.loadDataSet()
#得到词条listOPosts,每个词条的label listClasses



myVocabList = bayes.createVocabList(listOPosts)

trainMat = []

for postinDoc in listOPosts:
    trainMat.append(bayes.setOfWords2Vec(myVocabList, postinDoc))

p0V, p1V, pAb = bayes.trainNB0(trainMat, listClasses)
#p0V表示各种词汇出现在非侮辱性文档中的概率;p1V各种词汇出现在侮辱性文档中的概率

