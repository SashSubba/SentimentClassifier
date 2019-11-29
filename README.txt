To run the program, open a python command and run the Main.py file as such : 

"python Main.py"

A prompt will first ask to input the document file (txt format) that will be trained on.
Input the name of the text file with .txt extension as such :

"
Please write the document file you would like to have trained (e.g. all_sentiment.txt ): all_sentiment_shuffled.txt
"

Once the program has completed training on the provided txt file within a second, a prompt will ask if you would like the program
to determine the label score of a new single document, then it will print the score for the requested label score .
Input the review of the document and label, as such:

"
Please write the document file you would like to have scored: martin short is priceless in this movie - i enjoyed every minute .

Please write the label of the document you would like to have scored (pos or neg): pos
7.415391382863904e-39
"

Afterwards, a new prompt will ask if you would like to classify a new document. If the input was yes, it will print out which 
is the most likely label associated with the document on the next line. The prmopt and result is as such:

"
Please write the document file you would like to have classified: martin short is priceless in this movie - i enjoyed every minute .
The document classifies as a positive review.
"

Finally, another prompt will ask if you would like to evaluate the accuracy of the initially provided document.
If you input yes, it will print out the accuracy on the next line.

"
Do you want to evaluate the accuracy of the document file that was trained? (yes or no): yes
The accuracy is :  0.8090642047838859
"