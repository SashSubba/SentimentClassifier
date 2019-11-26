import numpy

if __name__ == "__main__":

    dataFile = open("all_sentiment_shuffled.txt", encoding="utf8")
    temp = 0

    for line in dataFile:
        if temp == 10:
            break

        fileLines = line.split()
        print(fileLines[3:], "\n\n" )

        temp += 1

