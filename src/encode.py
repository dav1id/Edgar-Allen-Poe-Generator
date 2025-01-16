import keras
import tensorflow as tf
import os
import numpy as np

#Loading a data - Loading data for the model to train on 
#Read contents of file
text = open("/Users/davidola/Desktop/tensorFlow/edgarAllenPoe.txt", 'rb').read().decode(encoding='utf-8')
text = text.replace("\\", " ")

#Encoding - assigning a number for every word, and then using word embeddings to group it together 
global encodingDict # Word : Num
global encodingMax 

unapproved = ["!", ".", "," "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "'", ":", ";"]
encodingDict = {'the' : 1}

#Max number of words being encoded - GIving it a limit to the number it encodes to constantly playtest with it. 
listedText = (text[:50000]).split()
encodingMax = 3000 #Give or take estimate of the amount of unique words in my list

def cleanText():
    for i in range(len(listedText)-1, -1, -1): #I encounter index out of range errors the most, and want to know a method to deal with them
        listedText[i] = listedText[i].lower() #refusing to make things lower 

        if len(listedText[i]) - 1 in unapproved:
            listedText[i] = listedText[i][0:len(listedText[i])- 2]
        elif (len(listedText[i]) > 1 and listedText[i][0] in unapproved) or len(listedText[i]) == 1: #listedText[i][1] in unapproved or 
            listedText.pop(i)

def encoding(text, indexStop):
    increment = 0 
    encodingCounter = 2 #Tells which number to assign the encoding 

    while (increment != indexStop) and increment < encodingMax:
        if not (text[increment] in encodingDict) and increment < encodingMax:
            encodingDict[text[increment]] = encodingCounter
            #print('encoded: {}'.format(text[increment]))
            encodingCounter += 1

        increment += 1

cleanText()
encoding(listedText, encodingMax)
#print(encodingDict)

#Creating unique encoded words with tensorFlow
textAsInt = [encodingDict.get(word, 0) for word in listedText] #0 handles words that are not encoded
print(textAsInt)

#Creating Training Examples
'''
The data is given as sequences - number of inputs you want to produce a given target. 
In a recurrent neural network you still need a target because it's going to use its training data to predict what word comes next, i.e:
"To be or not to (target)" - The model is going to predict from analysing the statements it's been trained on that target is going to be the word 'be'

That's why a sequence has one more number (seq_length+1) where the plus one accounts for the target. 
The first batch acts as a way to categorise the number of words in a sequence (to be or not to ___), the second batch acts a way to categorise the number of sequences
'''
seqLength = 100  # The data is given as sequences - number of inputs you want to produce a given target. 
#examples_per_epoch = len(text)//(seq_length+1)

# Create training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(textAsInt)
sequences = char_dataset.batch(seqLength+1, drop_remainder=True)

def split_input_target(chunk):  # Function splits the sequence into the input and the target, and then alters the sequence list so that it produces input, then target.
    input_text = chunk[:-1]  # For example hello: hell
    target_text = chunk[1:]  # ello
    return input_text, target_text  # hell, ello

dataset = sequences.map(split_input_target)  # we use map to apply the above function to every entry

batchSize = 64
#vocabSize (?)
#embeddingDim (?)
#RNN_UNITS (?)

#bufferSize (?)
# data = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)


#Building the Model
