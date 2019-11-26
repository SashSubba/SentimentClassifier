from collections import Counter
import numpy as np


if __name__ == "__main__":

    def document_separation(document):
        docsFile = []
        labelsFile = []
        for line in document:
            docsFile.append(line.strip().split()[3:])
            labelsFile.append(line.strip().split()[1])
        return docsFile, labelsFile

    def document_probabilities(documents, labels):
        finalProb = {}
        labelFreq = Counter()
        labelSum = 0

        for label in labels:
            labelFreq[label] += 1

        for value in labelFreq.values():
            labelSum += value

        for key in labelFreq:
            finalProb[key] = labelFreq[key] / labelSum
            ##TESTING PURPOSES: print(labelFreq[key], "-----------", labelSum, "------------", labelFreq[key] / labelSum)

        ##TESTING PURPOSES: print(labelFreq, labelSum, finalProb)

    print("----------------------------------------------------------------------------------------------------")
    print("Welcome to our Customer Review Sentiment Classification Program!\n")

    dataFile = input("Please write the document file you would like to have evaluated (e.g. all_sentiment.txt ): ")
    dataFile = open("all_sentiment_shuffled.txt", encoding="utf8")
    # for testing purposes only, need to switch "all_sentiment_shuffled.txt" back to dataFile

    docsFile, labelsFile = document_separation(dataFile)

    trainDocs = docsFile[:int(0.80*len(docsFile))]
    testDocs = docsFile[int(0.80*len(docsFile)):]
    trainLabels = labelsFile[:int(0.80*len(labelsFile))]
    testLabels = labelsFile[int(0.80*len(labelsFile)):]

    finalProb = document_probabilities(trainDocs, trainLabels)
