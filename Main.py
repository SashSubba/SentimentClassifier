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
        totalNbWordsPos = 0
        totalNbWordsNeg = 0

        for review in documents:
            for word in review:
                totalNbWordsFreq[word] += 1

        for label in labels:                                                        #Get Frequency of "neg" and "pos"
            labelFreq[label] += 1

        for key in labelFreq:                                       #Added P("pos") and P("neg") to the final prob dict.
            finalProb[key] = labelFreq[key] / totalNbReviews

        for i in range(totalNbReviews):                                             #Separation of Pos and Neg Reviews
            if labels[i] == "pos":
                posFiles.append(documents[i])
            else:
                negFiles.append(documents[i])

        for review in posFiles:                                                         #Get Frequency of "pos" words
            for word in review:
                posFreq[word] += 1
        for word in posFreq:
            totalNbWordsPos += posFreq[word]

        for review in negFiles:                                                         ##Get Frequency of "neg" words
            for word in review:
                negFreq[word] += 1
        for word in negFreq:
            totalNbWordsNeg += negFreq[word]

        for key in posFreq:                                                         #Add word prob of "pos" to finalProb
            finalProb[key + "/pos"] = (posFreq[key] + 0.5) / (totalNbWordsPos + (0.5 * len(totalNbWordsFreq)))

        for key in negFreq:                                                         #Add word prob of "neg" to finalProb
            finalProb[key + "/neg"] = (negFreq[key] + 0.5) / (totalNbWordsNeg + (0.5 * len(totalNbWordsFreq)))

        finalProb["posTotalSmoothing"] = (totalNbWordsPos + (0.5 * len(totalNbWordsFreq)))
        finalProb["negTotalSmoothing"] = (totalNbWordsNeg + (0.5 * len(totalNbWordsFreq)))

        print(finalProb)
        return finalProb

    def docScore(document, label, final_probs):
        Score = np.log(final_probs[label])

        for word in document.strip().split():
            temp = word + "/" + label
            if temp in final_probs:
                Score += np.log(final_probs[temp])
            else:
                temp2 = label + "TotalSmoothing"
                Score += np.log(0.5/final_probs[temp2])
        return np.exp(Score)

    def docClassify(document, final_probs):
        posScore = np.log(final_probs["pos"])
        negScore = np.log(final_probs["neg"])

        for word in document.strip().split():
            temp = word + "/pos"
            if temp in final_probs:
                posScore += np.log(final_probs[temp])
            else:
                posScore += np.log(0.5/final_probs["posTotalSmoothing"])   #NOT DONE NEED TO ADD NEW PROB AT PREV FNC

            temp = word + "/neg"
            if temp in final_probs:
                negScore += np.log(final_probs[temp])
            else:
                negScore += np.log(0.5/final_probs["negTotalSmoothing"])   #NOT DONE NEED TO ADD NEW PROB AT PREV FNC

        if posScore > negScore:
            print("The document classifies as a positive review.")
        else:
            print("The document classifies as a negative review.")

    print("----------------------------------------------------------------------------------------------------")
    print("Welcome to our Customer Review Sentiment Classification Program!\n")

    dataFile = input("Please write the document file you would like to have trained (e.g. all_sentiment.txt ): ")
    dataFile = open("all_sentiment_shuffled.txt", encoding="utf8")
    # for testing purposes only, need to switch "all_sentiment_shuffled.txt" back to dataFile

    docsFile, labelsFile = document_separation(dataFile)

    trainDocs = docsFile[:int(0.80*len(docsFile))]
    testDocs = docsFile[int(0.80*len(docsFile)):]
    trainLabels = labelsFile[:int(0.80*len(labelsFile))]
    testLabels = labelsFile[int(0.80*len(labelsFile)):]

    finalProb = document_probabilities(trainDocs, trainLabels)

    scoreQuestion = input("\nDo you want to have a document scored? (yes or no): ")
    if scoreQuestion.lower() == "yes":
        scoreFileDoc = input("\nPlease write the document file you would like to have scored: ")
        scoreFileLabel = input("\nPlease write the label of the document you would like to have scored: ")
        print(docScore(scoreFileDoc, scoreFileLabel, finalProb))

    classifyQuestion = input("\nDo you want to have a document classified? (yes or no): ")
    if classifyQuestion.lower() == "yes":
        classifyFile = input("\nPlease write the document file you would like to have classified: ")
        docClassify(classifyFile, finalProb)