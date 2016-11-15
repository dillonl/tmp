#!/usr/bin/python
import sys, random
import numpy as np

runQ1 = False
runQ2 = False
runQ3 = True

def getDataFromFile(filename):
    data = []
    with open(filename) as f:
        for line in f:
            line = line.replace("\n", "")
            dataLine = []
            dataLine.append(1)
            lineSplit = line.split(" ")
            for d in lineSplit:
                if len(d) == 0: continue
                if "." in d:
                    dataLine.append(int(float(d)))
                else:
                    dataLine.append(int(d))
            data.append(dataLine)
    return np.array(data)

def getLabelsFromFile(filename):
    data = []
    with open(filename) as f:
        for line in f:
            tmp = float(line)
            if tmp == 1.0:
                data.append(1)
            else:
                data.append(-1)
    return np.array(data)

handwritingTrainingExamples = getDataFromFile("data/handwriting/train.data")
handwritingTrainingLabels = getLabelsFromFile("data/handwriting/train.labels")
handwritingTestExamples = getDataFromFile("data/handwriting/test.data")
handwritingTestLabels = getLabelsFromFile("data/handwriting/test.labels")

madelonTrainingExamples = getDataFromFile("data/madelon/madelon_train.data")
madelonTrainingLabels = getLabelsFromFile("data/madelon/madelon_train.labels")
madelonTestExamples = getDataFromFile("data/madelon/madelon_test.data")
madelonTestLabels = getLabelsFromFile("data/madelon/madelon_test.labels")

def trainExample(trainingExamples, trainingLabels, numEpochs, gamma_0, C):
    w = np.zeros(trainingExamples.shape[1]) #initialize w to 0s
    t = 1
    gamma_t = gamma_0/(1+gamma_0*(1/C))
    for epoch in range(1, numEpochs):
        exampleIndices = range(0, len(trainingExamples) - 1)
        random.shuffle(exampleIndices)

        for idx in exampleIndices:
            gamma_t = float(gamma_0)/(1+gamma_0*(t/C))
            # if (trainingLabels[idx] * np.dot(w, trainingExamples[idx])) <= 1:
            if np.sign(trainingLabels[idx]) != np.sign(np.dot(w, trainingExamples[idx])):
                w = np.multiply((1 - gamma_t), w) + np.multiply((gamma_t * C * trainingLabels[idx]), trainingExamples[idx])
            else:
                w = np.multiply((1 - gamma_t), w)
            t += 1
    return w

def testExample(testExamples, testLabels, w):
    counts = {'TP': 0, 'FP': 0, 'TN': 0, 'FN':0, 'TC': 0}
    pSign = np.sign(1)
    nSign = np.sign(-1)
    for i in range(0, len(testLabels) -1):
        counts['TC'] = counts['TC'] + 1
        yiSign = np.sign(testLabels[i])
        xiSign = np.sign(np.dot(w, testExamples[i]))
        if xiSign == pSign and yiSign == xiSign:
            counts['TP'] = counts['TP'] + 1
        elif xiSign == pSign and yiSign != xiSign:
            counts['FP'] = counts['FP'] + 1
        elif xiSign == nSign and yiSign == xiSign:
            counts['TP'] = counts['TP'] + 1
        elif xiSign == nSign and yiSign != xiSign:
            counts['FN'] = counts['FN'] + 1
        counts['Accuracy'] = float(counts['TP']) / float(counts['TC'])
    return counts


if runQ1:
    gamma = 0.01
    C = 1
    handwritingTrainingVec = trainExample(handwritingTrainingExamples, handwritingTrainingLabels, 20, gamma, C)
    handwritingTrainingCounts = testExample(handwritingTrainingExamples, handwritingTrainingLabels, handwritingTrainingVec)
    handwritingTestCounts = testExample(handwritingTestExamples, handwritingTestLabels, handwritingTrainingVec)
    print "output for SVM q1:"
    print "gamma: %s" % gamma
    print "C: %s" % C
    print "Accuracy Handwriting [Test]: %s" % handwritingTestCounts['Accuracy']

if runQ2:
    print "output for SVM q2:"
    trainingAccuracy = {}
    testAccuracy = {}
    averageCount = {}
    C_ = [1, 2**-1, 2**-2, 2**-3, 2**-4, 2**-5, 2**-6]
    gamma_ = [float(0.05), float(0.01), float(0.005), float(0.001)]
    folds = 5
    trainingDataChunks = np.split(madelonTrainingExamples, 5)
    trainingLabelChunks = np.split(madelonTrainingLabels, 5)

    for i in range(0, folds):
        combinedDataSet = []
        combinedLabelSet = []
        for j in range(0, folds):
            if j == i: continue
            combinedDataSet = combinedDataSet + trainingDataChunks[j].tolist()
            combinedLabelSet = combinedLabelSet + trainingLabelChunks[j].tolist()
        combinedDataSet = np.array(combinedDataSet)
        combinedLabelSet = np.array(combinedLabelSet)
        for c in C_:
            for g in gamma_:
                key = str(c) + ":" + str(g)
                if key not in trainingAccuracy:
                    trainingAccuracy[key] = 0
                    testAccuracy[key] = 0
                    averageCount[key] = 0
                w = trainExample(combinedDataSet, combinedLabelSet, 20, g, c)
                madelonTrainingCounts = testExample(trainingDataChunks[i], trainingLabelChunks[i], w)
                madelonTestCounts = testExample(madelonTestExamples, madelonTestLabels, w)
                averageCount[key] = averageCount[key] + 1
                trainingAccuracy[key] = trainingAccuracy[key] + madelonTrainingCounts['Accuracy']
                testAccuracy[key] = testAccuracy[key] + madelonTestCounts['Accuracy']


    for key in averageCount:
        trainingValue = (float(trainingAccuracy[key]) / float(averageCount[key]))
        testValue = (float(testAccuracy[key]) / float(averageCount[key]))
        g = key.split(":")[0]
        C = key.split(":")[1]
        print "---"
        print "Gamma: %s" % g
        print "C: %s" % C
        print "Average Training: %s" % trainingValue
        print "Average Testing: %s" % testValue
        print "---"

if runQ3:
    gamma =  0.01
    C = 1
    handwritingTrainingVec = trainExample(handwritingTrainingExamples, handwritingTrainingLabels, 20, gamma, C)
    handwritingTestCounts = testExample(handwritingTestExamples, handwritingTestLabels, handwritingTrainingVec)
    p = float(handwritingTestCounts["TP"]) / (float(handwritingTestCounts["TP"]) + float(handwritingTestCounts["FP"]))
    r = float(handwritingTestCounts["TP"]) / (float(handwritingTestCounts["TP"]) + float(handwritingTestCounts["FN"]))
    F1 = 2* (float(p*r) / (float(p) + float(r)))
    print "output for SVM q3 [handwriting]:"
    print "gamma: %s" % gamma
    print "C: %s" % C
    print "Breakdown: %s" % handwritingTestCounts
    print "p %s" % p
    print "r %s" % r
    print "F1 %s" % F1

    gamma =  0.015625
    C = 0.05
    madelonTrainingVec = trainExample(madelonTrainingExamples, madelonTrainingLabels, 20, gamma, C)
    madelonTestCounts = testExample(madelonTestExamples, madelonTestLabels, madelonTrainingVec)
    p = float(madelonTestCounts["TP"]) / (float(madelonTestCounts["TP"]) + float(madelonTestCounts["FP"]))
    r = float(madelonTestCounts["TP"]) / (float(madelonTestCounts["TP"]) + float(madelonTestCounts["FN"]))
    F1 = 2* (float(p*r) / (float(p) + float(r)))
    print "output for SVM q3 [handwriting]:"
    print "gamma: %s" % gamma
    print "C: %s" % C
    print "Breakdown: %s" % madelonTestCounts
    print "p %s" % p
    print "r %s" % r
    print "F1 %s" % F1
