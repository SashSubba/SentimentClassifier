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
        posFiles = []
        negFiles = []
        posFreq = Counter()
        negFreq = Counter()
        labelFreq = Counter()
        totalNbReviews = len(labels)
        totalNbWordsFreq = Counter()
        totalNbWords = 0

        for review in documents:
            for word in review:
                totalNbWordsFreq[word] += 1

        totalNbWords = len(totalNbWordsFreq)

        print(totalNbWords)

        for label in labels:                                                        #Get Frequency of "neg" and "pos"
            labelFreq[label] += 1

        for key in labelFreq:                                       #Added P("pos") and P("neg") to the final prob dict.
            finalProb[key] = labelFreq[key] / totalNbReviews
            #print(labelFreq[key], "-----------", totalNbReviews, "------------", labelFreq[key] / totalNbReviews)

        #print(labelFreq, totalNbReviews, finalProb)

        for i in range(totalNbReviews):                                             #Separation of Pos and Neg Reviews
            if labels[i] == "pos":
                posFiles.append(documents[i])
            else:
                negFiles.append(documents[i])

        for review in posFiles:                                                         #Get Frequency of "pos" words
            for word in review:
                posFreq[word] += 1

        for review in negFiles:                                                         ##Get Frequency of "neg" words
            for word in review:
                negFreq[word] += 1

        for key in posFreq:                                                         #Add word prob of "pos" to finalProb
            finalProb[key + "/pos"] = posFreq[key] / len(posFiles)

        for key in negFreq:                                                         #Add word prob of "neg" to finalProb
            finalProb[key + "/neg"] = posFreq[key] / len(negFiles)

        return finalProb

    def docScore(document, label, final_probs):
        """

        :param document: existing or new document to be checked for label score
        :param label: sentiment polarity label
        :param final_probs: probabilities of each word given label and label itself, based on training set computed in document_probabilities
        :return: logarithm of probability of sentiment polarity label
        """
        posScore = np.log(final_probs["pos"])
        negScore = np.log(final_probs["neg"])


        for word in document:
            if (word + "/pos") in final_probs:
                posScore += np.log(final_probs[word+"/pos"])
            else:
                posScore += np.log(0.5/0)   #NOT DONE NEED TO ADD NEW PROB AT PREV FNC
            if (word + "/neg") in final_probs:
                negScore += np.log(final_probs[word+"/neg"])
            else:
                negScore += np.log(0.5/0)   #NOT DONE NEED TO ADD NEW PROB AT PREV FNC

        print(posScore)
        print(negScore)












    print("----------------------------------------------------------------------------------------------------")
    print("Welcome to our Customer Review Sentiment Classification Program!\n")

    dataFile = input("Please write the document file you would like to have evaluated (e.g. all_sentiment.txt ): ")
    dataFile = open("testingFile.txt", encoding="utf8")
    # for testing purposes only, need to switch "all_sentiment_shuffled.txt" back to dataFile

    docsFile, labelsFile = document_separation(dataFile)

    trainDocs = docsFile[:int(0.80*len(docsFile))]
    testDocs = docsFile[int(0.80*len(docsFile)):]
    trainLabels = labelsFile[:int(0.80*len(labelsFile))]
    testLabels = labelsFile[int(0.80*len(labelsFile)):]

    finalProb = document_probabilities(trainDocs, trainLabels)

    docScore(trainDocs, testDocs, finalProb)
