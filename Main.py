from collections import Counter
import numpy as np

if __name__ == "__main__":
    scoreQuestion, scoreFileLabel, classifyQuestion, accuracyQuestion, fileBool = "", "", "", "", False

    def document_separation(document):
        """Separates the document into 2 separate files, one for the review and the other for labels.

        Arguments:
            document {txt file} -- document

        Returns:
            docsFile, labelsFile {list} -- list of the reviews and list of the corresponding labels
        """
        docsFile = []
        labelsFile = []
        for line in document:
            docsFile.append(line.strip().split()[3:])
            labelsFile.append(line.strip().split()[1])
        return docsFile, labelsFile

    def document_probabilities(documents, labels):
        """Computes the probabilities of the given documents, their words and labels.

        Arguments:
            document {List} -- documents list
            label {List} -- labels list

        Returns:
            finalProb {Dict} -- Dictionary that contains all of the final probabilities needed for Naive Bayes Formula
        """
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

        return finalProb

    def document_score(document, label, final_probs):
        """Computes the score of the given document (review) given the label provided.

        Arguments:
            document {String} -- given document
            label {String} -- given label
            final_probs {list} -- list containing probabilities of each word from training set

        Returns:
            Score {Int} -- score (probability) of the given document and label
        """
        Score = np.log(final_probs[label])

        for word in document.strip().split():
            temp = word + "/" + label
            if temp in final_probs:
                Score += np.log(final_probs[temp])
            else:
                temp2 = label + "TotalSmoothing"
                Score += np.log(0.5/final_probs[temp2])
        return np.exp(Score)

    def string_classification(document, final_probs):
        """Return the label of the given document.
        
        Arguments:
            document {String} -- document 
            final_probs {list} -- list containing probabilities of each word from training set
        
        Returns:
            String -- guessed label of document
        """
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

    def document_classification(docs, final_probs):
        """finds the labels of each document in docs.

        Arguments:
            docs {list} -- list containing test documents
            final_probs {list} -- list containing probabilities of word from training set
        
        Returns:
            guessedLabels -- list containing guessed labels of each document
        """

        guessedLabels = []  #list containing labels of each document

        for doc in docs:
            posScore = np.log(final_probs["pos"])
            negScore = np.log(final_probs["neg"])

            for word in doc:
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

            if posScore > negScore :
                guessedLabels.append("pos")
            else:
                guessedLabels.append("neg")

        return guessedLabels


    def document_accuracy(true_labels, guessed_labels):
        """Computes the accuracy of guessed labels against true_labels of the test set.
        
        Arguments:
            true_labels {List} -- [true labels for each document]
            guessed_labels {List} -- [guessed labels for each document]
        
        Returns:
            [accuracy] -- accuracy of guessed_labels
        """

        nbCorrectlyClassifiedDocuments = 0
        nbTestDocuments = len(true_labels)

        for i in range(nbTestDocuments):
            if guessed_labels[i] == true_labels[i]:
                nbCorrectlyClassifiedDocuments += 1

        return (nbCorrectlyClassifiedDocuments / nbTestDocuments)

    print("----------------------------------------------------------------------------------------------------")
    print("Welcome to our Customer Review Sentiment Classification Program!\n")

    while fileBool is False:
        try:
            fileBool = True
            dataFile = input("\nPlease write the document file you would like to have trained (e.g. all_sentiment.txt ): ")
            dataFile = open(dataFile, encoding="utf8")
        except:
            fileBool = False

    docsFile, labelsFile = document_separation(dataFile)

    trainDocs = docsFile[:int(0.80*len(docsFile))]
    testDocs = docsFile[int(0.80*len(docsFile)):]
    trainLabels = labelsFile[:int(0.80*len(labelsFile))]
    testLabels = labelsFile[int(0.80*len(labelsFile)):]

    finalProb = document_probabilities(trainDocs, trainLabels)

    while scoreQuestion.lower() not in ["yes", "no"]:
        scoreQuestion = input("\nDo you want to have a document scored? (yes or no): ")
        if scoreQuestion.lower() == "yes":
            scoreFileDoc = input("\nPlease write the document file you would like to have scored: ")
            while scoreFileLabel.lower() not in ["pos", "neg"]:
                scoreFileLabel = input("\nPlease write the label of the document you would like to have scored: ")
                print(document_score(scoreFileDoc, scoreFileLabel, finalProb))

    while classifyQuestion.lower() not in ["yes", "no"]:
        classifyQuestion = input("\nDo you want to have a document classified? (yes or no): ")
        if classifyQuestion.lower() == "yes":
            classifyFile = input("\nPlease write the document file you would like to have classified: ")
            string_classification(classifyFile, finalProb)

    while accuracyQuestion.lower() not in ["yes", "no"]:
        accuracyQuestion = input("\nDo you want to evaluate the accuracy of the document file that was trained? (yes or no): ")
        if accuracyQuestion.lower() == "yes":
            guessedLabels = document_classification(testDocs,finalProb)#list containing guessed labels for each documens
            print("The accuracy is : ", document_accuracy(testLabels,guessedLabels))

    """
    Task 4:
    
        Example 1 : health neg 758.txt results were never consistent . many " err " readings . i was definitely no satisfied
                    Positive Score: 1.1563219248016396e-49
                    Negative Score: 1.0684234703227198e-49
        Example 2 : health neg 309.txt i like the idea , but the slippers just are n't comfortable to stand on . it 's great while you are sitting , though
                    Positive Score: 7.045658449914817e-65
                    Negative Score: 6.737927616283555e-65
        Example 3 : health neg 765.txt shaker is adequate for breaking down whey , but it leaks . i always have to wash my hands and the outside of the bottle after making anything with it
                    Positive Score: 1.511951986499868e-87
                    Negative Score: 1.379507197688343e-87
         
         In terms of the first example, one reason as to why it could have been classified erroneously would be that there was a grammatical error.
         We believe what was meant to be written was "i was definitely not satisfied", however "no" was written instead, which could have potentially
         caused a lower score for the negative class, hence classifiying the review as positive.
         For the second example, there is bias because of the fact that the reviewer also talks about the positive aspects of the product. In fact,
         the reviewer claimed the product review was overall negative, yet spent a majority of the review talking about the positives of the product.
         This allowed for an erroneous classification of our classifier.
         Finally, in terms of the third example, one of the reasons that the classification might have been induced into error is also similar to the
         reason stated for example two. This review, stated as being a negative review by the reviewer, is mostly neutral in opinion, as it describes
         one negative aspect of the product as well as one positive aspect. Furthermore, the review was also depicting a situation that occured with
         the product that was neither positive nor negative. All of this could have led the Sentiment Classifier to miscalculate the probabilities
         and caused the improper classification.
    """