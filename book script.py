# -*- coding: utf-8 -*-
"""
Created on Sun May 24 09:44:52 2020

@author: egultekin
"""

### Lesson 1 : Introduction and Optimization Problems

# class Item(object):
#     def __init__(self, n, v, w):
#         self.name = n
#         self.value = v
#         self.weight = w

#     def getName(self):
#         return self.name

#     def getValue(self):
#         return self.value

#     def getWeight(self):
#         return self.weight

#     def __str__(self):
#         result = '<' + self.name + ', ' + str(self.value) + ', ' + str(self.weight) + '>'
#         return result

# def value(item):
#     return item.getValue()

# def weightInverse(item):
#     return 1.0/item.getWeight()

# def density(item):
#     return item.getValue()/item.getWeight()

# def greedy(items, maxWeight, keyFunction):
#     """Assumes Items a list, maxWeight >= 0,
#     keyFunction maps elements of Items to numbers"""

#     itemsCopy = sorted(items, key = keyFunction, reverse = True)
#     result = []
#     totalValue, totalWeight = 0.0, 0.0
#     for i in range(len(itemsCopy)):
#         if (totalWeight + itemsCopy[i].getWeight()) <= maxWeight:
#             result.append(itemsCopy[i])
#             totalWeight += itemsCopy[i].getWeight()
#             totalValue += itemsCopy[i].getValue()
#     return (result, totalValue)


# def buildItems():
#     names = ['clock','painting','radio','vase','book','computer']
#     values = [175,90,20,50,10,200]
#     weights = [10,9,4,2,1,20]
#     Items = []
#     for i in range(len(values)):
#         Items.append(Item(names[i], values[i], weights[i]))
#     return Items

# def testGreedy(items, maxWeight, keyFunction):
#     taken, val = greedy(items, maxWeight, keyFunction)
#     print('Total value of items taken is', val)
#     for item in taken:
#         print(' ', item)

# def testGreedys(maxWeight = 20):
#     items = buildItems()
#     print('Use greedy by value to fill knapsack of size', maxWeight)
#     testGreedy(items, maxWeight, value)
#     print('\nUse greedy by weight to fill knapsack of size', maxWeight)
#     testGreedy(items, maxWeight, weightInverse)
#     print('\nUse greedy by density to fill knapsack of size', maxWeight)
#     testGreedy(items, maxWeight, density)

# testGreedys()


### Lesson 2 : Optimization Problems

### Lesson 3 : Graph-theoretic Models

# class Node(object):
#     def __init__(self, name):
#         """Assumes name is a string"""
#         self.name = name
#     def getName(self):
#         return self.name
#     def __str__(self):
#         return self.name

# class Edge(object):
#     def __init__(self, src, dest):
#         """Assumes src and dest are nodes"""
#         self.src = src
#         self.dest = dest
#     def getSource(self):
#         return self.src
#     def getDestination(self):
#         return self.dest
#     def __str__(self):
#         return self.src.getName() + '->' + self.dest.getName()

# class WeightedEdge(Edge):
#     def __init__(self, src, dest, weight = 1.0):
#         """Assumes src and dest are nodes, weight a number"""
#         self.src = src
#         self.dest = dest
#         self.weight = weight
#     def getWeight(self):
#         return self.weight
#     def __str__(self):
#         return self.src.getName() + '->(' + str(self.weight) + ')' + self.dest.getName()

# class Digraph(object):
#     #nodes is a list of the nodes in the graph
#     #edges is a dict mapping each node to a list of its children
#     def __init__(self):
#         self.nodes = []
#         self.edges = {}
#     def addNode(self, node):
#         if node in self.nodes:
#             raise ValueError('Duplicate node')
#         else:
#             self.nodes.append(node)
#             self.edges[node] = []
#     def addEdge(self, edge):
#         src = edge.getSource()
#         dest = edge.getDestination()
#         if not (src in self.nodes and dest in self.nodes):
#             raise ValueError('Node not in graph')
#         self.edges[src].append(dest)
#     def childrenOf(self, node):
#         return self.edges[node]
#     def hasNode(self, node):
#         return node in self.nodes
#     def __str__(self):
#         result = ''
#         for src in self.nodes:
#             for dest in self.edges[src]:
#                 result = result + src.getName() + '->' + dest.getName() + '\n'
#         return result[:-1] #omit final newline

# class Graph(Digraph):
#     def addEdge(self, edge):
#         Digraph.addEdge(self, edge)
#         rev = Edge(edge.getDestination(), edge.getSource())
#         Digraph.addEdge(self, rev)

### Lecture 4 : Stochastic Thinking

### Lecture 5 : Random Walks

# import pylab
# # pylab.figure(1) #create figure 1
# # pylab.plot([1,2,3,4], [1,7,3,5]) #draw on figure 1
# # pylab.show() #show figure on screen

# # principal = 10000 #initial investment
# # interestRate = 0.05
# # years = 20
# # values = []
# # for i in range(years + 1):
# #     values.append(principal)
# #     principal += principal * interestRate
# # pylab.plot(values, 'ko')
# # pylab.title('5% Growth, Compounded Annually')
# # pylab.xlabel('Years of Compounding')
# # pylab.ylabel('Value of Principal ($)')

# # pylab.plot(values, linewidth = 30)
# # pylab.title('5% Growth, Compounded Annually', fontsize = 'xx-large')
# # pylab.xlabel('Years of Compounding', fontsize = 'x-small')
# # pylab.ylabel('Value of Principal ($)')

# #set line width
# pylab.rcParams['lines.linewidth'] = 4
# #set font size for titles
# pylab.rcParams['axes.titlesize'] = 20
# #set font size for labels on axes
# pylab.rcParams['axes.labelsize'] = 20
# #set size of numbers on x-axis
# pylab.rcParams['xtick.labelsize'] = 16
# #set size of numbers on y-axis
# pylab.rcParams['ytick.labelsize'] = 16
# #set size of ticks on x-axis
# pylab.rcParams['xtick.major.size'] = 7
# #set size of ticks on y-axis
# pylab.rcParams['ytick.major.size'] = 7
# #set size of markers, e.g., circles representing points
# pylab.rcParams['lines.markersize'] = 10
# #set number of times marker is shown when displaying legend
# pylab.rcParams['legend.numpoints'] = 1


# def findPayment(loan, r, m):
#     """Assumes: loan and r are floats, m an int
#     Returns the monthly payment for a mortgage of size
#     loan at a monthly rate of r for m months"""
#     return loan*((r*(1+r)**m)/((1+r)**m - 1))

# class Mortgage(object):
#     """Abstract class for building different kinds of mortgages"""
#     def __init__(self, loan, annRate, months):
#         self.loan = loan
#         self.rate = annRate/12.0
#         self.months = months
#         self.paid = [0.0]
#         self.outstanding = [loan]
#         self.payment = findPayment(loan, self.rate, months)
#         self.legend = None #description of mortgage

#     def makePayment(self):
#         self.paid.append(self.payment)
#         reduction = self.payment - self.outstanding[-1]*self.rate
#         self.outstanding.append(self.outstanding[-1] - reduction)

#     def getTotalPaid(self):
#         return sum(self.paid)

#     def __str__(self):
#         return self.legend

#     def plotPayments(self, style):
#         pylab.plot(self.paid[1:], style, label = self.legend)

#     def plotBalance(self, style):
#         pylab.plot(self.outstanding, style, label = self.legend)

#     def plotTotPd(self, style):
#         totPd = [self.paid[0]]
#         for i in range(1, len(self.paid)):
#             totPd.append(totPd[-1] + self.paid[i])
#         pylab.plot(totPd, style, label = self.legend)

#     def plotNet(self, style):
#         totPd = [self.paid[0]]
#         for i in range(1, len(self.paid)):
#             totPd.append(totPd[-1] + self.paid[i])
#         equityAcquired = pylab.array([self.loan] * len(self.outstanding))
#         equityAcquired = equityAcquired - pylab.array(self.outstanding)
#         net = pylab.array(totPd) - equityAcquired
#         pylab.plot(net, style, label = self.legend)

# class Fixed(Mortgage):
#     def __init__(self, loan, r, months):
#         Mortgage.__init__(self, loan, r, months)
#         self.legend = 'Fixed, ' + str(r*100) + '%'

# class FixedWithPts(Mortgage):
#     def __init__(self, loan, r, months, pts):
#         Mortgage.__init__(self, loan, r, months)
#         self.pts = pts
#         self.paid = [loan*(pts/100.0)]
#         self.legend = 'Fixed, ' + str(r*100) + '%, ' + str(pts) + ' points'

# class TwoRate(Mortgage):
#     def __init__(self, loan, r, months, teaserRate, teaserMonths):
#         Mortgage.__init__(self, loan, teaserRate, months)
#         self.teaserMonths = teaserMonths
#         self.teaserRate = teaserRate
#         self.nextRate = r/12.0
#         self.legend = str(teaserRate*100) + '% for ' + str(self.teaserMonths) + ' months, then ' + str(r*100) + '%'

#     def makePayment(self):
#        if len(self.paid) == self.teaserMonths + 1:
#            self.rate = self.nextRate
#            self.payment = findPayment(self.outstanding[-1], self.rate, self.months - self.teaserMonths)
#        Mortgage.makePayment(self)


# def plotMortgages(morts, amt):
#     def labelPlot(figure, title, xLabel, yLabel):
#         pylab.figure(figure)
#         pylab.title(title)
#         pylab.xlabel(xLabel)
#         pylab.ylabel(yLabel)
#         pylab.legend(loc = 'best')
#     styles = ['k-', 'k-.', 'k:']
#     #Give names to figure numbers
#     payments, cost, balance, netCost = 0, 1, 2, 3
#     for i in range(len(morts)):
#         pylab.figure(payments)
#         morts[i].plotPayments(styles[i])
#         pylab.figure(cost)
#         morts[i].plotTotPd(styles[i])
#         pylab.figure(balance)
#         morts[i].plotBalance(styles[i])
#         pylab.figure(netCost)
#         morts[i].plotNet(styles[i])
#     labelPlot(payments, 'Monthly Payments of $' + str(amt) + ' Mortgages', 'Months', 'Monthly Payments')
#     labelPlot(cost, 'Cash Outlay of $' + str(amt) + ' Mortgages', 'Months', 'Total Payments')
#     labelPlot(balance, 'Balance Remaining of $' + str(amt) + ' Mortgages', 'Months', 'Remaining Loan Balance of $')
#     labelPlot(netCost, 'Net Cost of $' + str(amt) + ' Mortgages', 'Months', 'Payments - Equity $')

# def compareMortgages(amt, years, fixedRate, pts, ptsRate, varRate1, varRate2, varMonths):
#     totMonths = years*12
#     fixed1 = Fixed(amt, fixedRate, totMonths)
#     fixed2 = FixedWithPts(amt, ptsRate, totMonths, pts)
#     twoRate = TwoRate(amt, varRate2, totMonths, varRate1, varMonths)
#     morts = [fixed1, fixed2, twoRate]
#     for m in range(totMonths):
#         for mort in morts:
#             mort.makePayment()
#     plotMortgages(morts, amt)

# compareMortgages(amt=200000, years=30, fixedRate=0.07, pts = 3.25, ptsRate=0.05, varRate1=0.045, varRate2=0.095, varMonths=48)


### Lecture 6 : Monte Carlo Simulation

# import random
# import pylab

# def flip(numFlips):
#     """Assumes numFlips a positive int"""
#     heads = 0
#     for i in range(numFlips):
#         if random.choice(('H', 'T')) == 'H':
#             heads += 1
#     return heads / numFlips

# def flipSim(numFlipsPerTrial, numTrials):
#     """Assumes numFlipsPerTrial and numTrials positive ints"""
#     fracHeads = []
#     for i in range(numTrials):
#         fracHeads.append(flip(numFlipsPerTrial))
#     mean = sum(fracHeads) / len(fracHeads)
#     return mean

# # print('Mean: ', flipSim(10, 1))
# # print('Mean: ', flipSim(10, 100))
# # print('Mean: ', flipSim(100, 100000))

# def regressToMean(numFlips, numTrials):
#     #Get fraction of heads for each trial of numFlips
#     fracHeads = []
#     for t in range(numTrials):
#         fracHeads.append(flip(numFlips))
#     #Find trials with extreme results and for each the next trial
#     extremes, nextTrials = [], []
#     for i in range(len(fracHeads) - 1):
#         if fracHeads[i] < 0.33 or fracHeads[i] > 0.66:
#             extremes.append(fracHeads[i])
#             nextTrials.append(fracHeads[i+1])

#     #Plot results
#     pylab.plot(range(len(extremes)), extremes, 'ko', label = 'Extreme')
#     pylab.plot(range(len(nextTrials)), nextTrials, 'k^', label = 'Next Trial')
#     pylab.axhline(0.5)
#     pylab.ylim(0, 1)
#     pylab.xlim(-1, len(extremes) + 1)
#     pylab.xlabel('Extreme Example and Next Trial')
#     pylab.ylabel('Fraction Heads')
#     pylab.title('Regression to the Mean')
#     pylab.legend(loc = 'best')

# # regressToMean(15, 40)


# def flipPlot(minExp, maxExp):
#     """Assumes minExp and maxExp positive integers; minExp < maxExp
#     Plots results of 2**minExp to 2**maxExp coin flips"""
#     ratios, diffs, xAxis = [], [], []
#     for exp in range(minExp, maxExp + 1):
#         xAxis.append(2**exp)
#     for numFlips in xAxis:
#         numHeads = 0
#         for n in range(numFlips):
#             if random.choice(('H', 'T')) == 'H':
#                 numHeads += 1
#         numTails = numFlips - numHeads
#         try:
#             ratios.append(numHeads/numTails)
#             diffs.append(abs(numHeads - numTails))
#         except ZeroDivisionError:
#             continue

#     pylab.title('Difference Between Heads and Tails')
#     pylab.xlabel('Number of Flips')
#     pylab.ylabel('Abs(#Heads - #Tails)')
#     pylab.plot(xAxis, diffs, 'ko')
#     pylab.figure()
#     pylab.title('Heads/Tails Ratios')
#     pylab.xlabel('Number of Flips')
#     pylab.ylabel('#Heads/#Tails')
#     pylab.plot(xAxis, ratios, 'ko')

# # random.seed(0)
# # flipPlot(4, 20)

# def variance(X):
#     """Assumes that X is a list of numbers.
#     Returns the standard deviation of X"""
#     mean = sum(X)/len(X)
#     tot = 0.0
#     for x in X:
#         tot += (x - mean)**2
#     return tot/len(X)

# def stdDev(X):
#     """Assumes that X is a list of numbers.
#     Returns the standard deviation of X"""
#     return variance(X)**0.5


# def makePlot(xVals, yVals, title, xLabel, yLabel, style, logX = False, logY = False):
#     pylab.figure()
#     pylab.title(title)
#     pylab.xlabel(xLabel)
#     pylab.ylabel(yLabel)
#     pylab.plot(xVals, yVals, style)
#     if logX:
#         pylab.semilogx()
#     if logY:
#         pylab.semilogy()

# def runTrial(numFlips):
#     numHeads = 0
#     for n in range(numFlips):
#         if random.choice(('H', 'T')) == 'H':
#             numHeads += 1
#     numTails = numFlips - numHeads
#     return (numHeads, numTails)

# def CV(X):
#     mean = sum(X)/len(X)
#     try:
#         return stdDev(X)/mean
#     except ZeroDivisionError:
#         return float('nan')

# def flipPlot1(minExp, maxExp, numTrials):
#     """Assumes minExp, maxExp, numTrials ints >0; minExp < maxExp
#     Plots summaries of results of numTrials trials of
#     2**minExp to 2**maxExp coin flips"""
#     ratiosMeans, diffsMeans, ratiosSDs, diffsSDs = [], [], [], []
#     ratiosCVs, diffsCVs, xAxis = [], [], []
#     for exp in range(minExp, maxExp + 1):
#         xAxis.append(2**exp)
#     for numFlips in xAxis:
#         ratios, diffs = [], []
#         for t in range(numTrials):
#             numHeads, numTails = runTrial(numFlips)
#             ratios.append(numHeads/numTails)
#             diffs.append(abs(numHeads - numTails))
#         ratiosMeans.append(sum(ratios)/numTrials)
#         diffsMeans.append(sum(diffs)/numTrials)
#         ratiosSDs.append(stdDev(ratios))
#         diffsSDs.append(stdDev(diffs))
#         ratiosCVs.append(CV(ratios))
#         diffsCVs.append(CV(diffs))

#     numTrialsString = ' (' + str(numTrials) + ' Trials)'
#     title = 'Mean Heads/Tails Ratios' + numTrialsString
#     makePlot(xAxis, ratiosMeans, title, 'Number of flips', 'Mean Heads/Tails', 'ko', logX = True)

#     title = 'SD Heads/Tails Ratios' + numTrialsString
#     makePlot(xAxis, ratiosSDs, title, 'Number of Flips', 'Standard Deviation', 'ko', logX = True, logY = True)

#     title = 'Mean abs(#Heads - #Tails)' + numTrialsString
#     makePlot(xAxis, diffsMeans, title, 'Number of Flips', 'Mean abs(#Heads - #Tails)', 'ko', logX = True, logY = True)

#     title = 'SD abs(#Heads - #Tails)' + numTrialsString
#     makePlot(xAxis, diffsSDs, title, 'Number of Flips', 'Standard Deviation', 'ko', logX = True, logY = True)

#     title = 'Coeff. of Var. abs(#Heads - #Tails)' + numTrialsString
#     makePlot(xAxis, diffsCVs, title, 'Number of Flips', 'Coeff. of Var.', 'ko', logX = True)

#     title = 'Coeff. of Var. Heads/Tails Ratio' + numTrialsString
#     makePlot(xAxis, ratiosCVs, title, 'Number of Flips', 'Coeff. of Var.', 'ko', logX = True, logY = True)


# flipPlot1(4, 20, 20)


# Lecture 9 - 10: Understanding experimental data

# import pylab
# import random

# def getData(fileName):
#     dataFile = open(fileName, 'r')
#     distances = []
#     masses = []
#     dataFile.readline() #ignore header
#     for line in dataFile:
#         d, m = line.split(' ')
#         distances.append(float(d))
#         masses.append(float(m))
#     dataFile.close()
#     return (masses, distances)

# def plotData(inputFile):
#     masses, distances = getData(inputFile)
#     distances = pylab.array(distances)
#     masses = pylab.array(masses)
#     forces = masses*9.81
#     pylab.plot(forces, distances, 'bo', label = 'Measured displacements')
#     pylab.title('Measured Displacement of Spring')
#     pylab.xlabel('|Force| (Newtons)')
#     pylab.ylabel('Distance (meters)')
#     pylab.show()

# plotData('C:\\Users\\egultekin\\Desktop\\Python\\MIT - Introduction to Computational Thinking and Data Science - 6.002\\Lecture9\\springData.txt')

# def fitData(inputFile):
#     masses, distances = getData(inputFile)
#     distances = pylab.array(distances)
#     forces = pylab.array(masses)*9.81
#     pylab.plot(forces, distances, 'ko',label = 'Measured displacements')
#     pylab.title('Measured Displacement of Spring')
#     pylab.xlabel('|Force| (Newtons)')
#     pylab.ylabel('Distance (meters)')
#     #find linear fit
#     a,b = pylab.polyfit(forces, distances, 1)
#     predictedDistances = a*pylab.array(forces) + b
#     k = 1.0/a #see explanation in text
#     pylab.plot(forces, predictedDistances, label = 'Displacements predicted by\nlinear fit, k = ' + str(round(k, 5)))

#     #find cubic fit
#     fit = pylab.polyfit(forces, distances, 3)
#     predictedDistances = pylab.polyval(fit, forces)
#     pylab.plot(forces, predictedDistances, 'k:', label = 'cubic fit')

#     pylab.legend(loc = 'best')
#     pylab.show()

# fitData('C:\\Users\\egultekin\\Desktop\\Python\\MIT - Introduction to Computational Thinking and Data Science - 6.002\\Lecture9\\springData.txt')

# def getTrajectoryData(fileName):
#     dataFile = open(fileName, 'r')
#     distances = []
#     heights1, heights2, heights3, heights4 = [],[],[],[]
#     dataFile.readline()
#     for line in dataFile:
#         d, h1, h2, h3, h4 = line.split()
#         distances.append(float(d))
#         heights1.append(float(h1))
#         heights2.append(float(h2))
#         heights3.append(float(h3))
#         heights4.append(float(h4))
#     dataFile.close()
#     return (distances, [heights1, heights2, heights3, heights4])

# def rSquared(measured, predicted):
#     """Assumes measured a one-dimensional array of measured values
#     predicted a one-dimensional array of predicted values
#     Returns coefficient of determination"""
#     estimateError = ((predicted - measured)**2).sum()
#     meanOfMeasured = measured.sum()/len(measured)
#     variability = ((measured - meanOfMeasured)**2).sum()
#     return 1 - estimateError/variability

# def getHorizontalSpeed(quadFit, minX, maxX):
#     """Assumes quadFit has coefficients of a quadratic polynomial
#     minX and maxX are distances in inches
#     Returns horizontal speed in feet per second"""
#     inchesPerFoot = 12
#     xMid = (maxX - minX)/2
#     a,b,c = quadFit[0], quadFit[1], quadFit[2]
#     yPeak = a*xMid**2 + b*xMid + c
#     g = 32.16*inchesPerFoot #accel. of gravity in inches/sec/sec
#     t = (2*yPeak/g)**0.5 #time in seconds from peak to target
#     print('Horizontal speed =', int(xMid/(t*inchesPerFoot)), 'feet/sec')

# def processTrajectories(fileName):
#     distances, heights = getTrajectoryData(fileName)
#     numTrials = len(heights)
#     distances = pylab.array(distances)
#     #Get array containing mean height at each distance
#     totHeights = pylab.array([0]*len(distances))
#     for h in heights:
#         totHeights = totHeights + pylab.array(h)
#     meanHeights = totHeights/len(heights)
#     pylab.title('Trajectory of Projectile (Mean of ' + str(numTrials) + ' Trials)')
#     pylab.xlabel('Inches from Launch Point')
#     pylab.ylabel('Inches Above Launch Point')
#     pylab.plot(distances, meanHeights, 'ko')
#     fit = pylab.polyfit(distances, meanHeights, 1)
#     altitudes = pylab.polyval(fit, distances)
#     pylab.plot(distances, altitudes, 'b', label = 'Linear Fit')
#     print('RSquare of linear fit =', rSquared(meanHeights, altitudes))
#     fit = pylab.polyfit(distances, meanHeights, 2)
#     altitudes = pylab.polyval(fit, distances)
#     pylab.plot(distances, altitudes, 'k:', label = 'Quadratic Fit')
#     print('RSquare of quadratic fit =', rSquared(meanHeights, altitudes))
#     pylab.legend()
#     getHorizontalSpeed(fit, distances[-1], distances[0])
#     pylab.show()

# processTrajectories('C:\\Users\\egultekin\\Desktop\\Python\\MIT - Introduction to Computational Thinking and Data Science - 6.002\\Lecture9\\launcherData.txt')


# vals = []
# for i in range(10):
#     vals.append(3**i)
# pylab.plot(vals,'ko', label = 'Actual points')
# xVals = pylab.arange(10)
# fit = pylab.polyfit(xVals, vals, 5)
# yVals = pylab.polyval(fit, xVals)
# pylab.plot(yVals, 'kx', label = 'Predicted points', markeredgewidth = 2, markersize = 25)
# pylab.title('Fitting y = 3**x')
# pylab.legend(loc = 'upper left')
# print('Model predicts that 3**20 is roughly', pylab.polyval(fit, [3**20])[0])
# print('Actual value of 3**20 is', 3**20)
# pylab.show()


# xVals, yVals = [], []
# for i in range(10):
#     xVals.append(i)
#     yVals.append(3**i)
# pylab.plot(xVals, yVals, 'k')
# pylab.semilogy()
# pylab.show()

# import math

# def createData(f, xVals):
#     """Asssumes f is afunction of one argument
#     xVals is an array of suitable arguments for f
#     Returns array containing results of applying f to the
#     elements of xVals"""
#     yVals = []
#     for i in xVals:
#         yVals.append(f(xVals[i]))
#     return pylab.array(yVals)

# def fitExpData(xVals, yVals):
#     """Assumes xVals and yVals arrays of numbers such that
#     yVals[i] == f(xVals[i]), where f is an exponential function
#     Returns a, b, base such that log(f(x), base) == ax + b"""
#     logVals = []
#     for y in yVals:
#         logVals.append(math.log(y, 2.0)) #get log base 2
#     fit = pylab.polyfit(xVals, logVals, 1)
#     return fit, 2.0

# xVals = range(10)
# f = lambda x: 3**x
# yVals = createData(f, xVals)
# pylab.plot(xVals, yVals, 'ko', label = 'Actual values')
# fit, base = fitExpData(xVals, yVals)
# predictedYVals = []
# for x in xVals:
#     predictedYVals.append(base**pylab.polyval(fit, x))
# pylab.plot(xVals, predictedYVals, label = 'Predicted values')
# pylab.title('Fitting an Exponential Function')
# pylab.legend(loc = 'upper left')
# #Look at a value for x not in original data
# print('f(20) =', f(20))
# print('Predicted value =', int(base**(pylab.polyval(fit, [20]))))
# pylab.show()


### Lecture 11: Introduction to machine learning

# import pylab

# # Minkowski distance
# def minkowskiDist(v1, v2, p):
#     """Assumes v1 and v2 are equal-length arrays of numbers
#     Returns Minkowski distance of order p between v1 and v2"""
#     dist = 0.0
#     for i in range(len(v1)):
#         dist += abs(v1[i] - v2[i])**p
#     return dist**(1/p)

# class Animal(object):
#     def __init__(self, name, features):
#         """Assumes name a string; features a list of numbers"""
#         self.name = name
#         self.features = pylab.array(features)

#     def getName(self):
#         return self.name

#     def getFeatures(self):
#         return self.features

#     def distance(self, other):
#         """Assumes other an Animal
#         Returns the Euclidean distance between feature vectors
#         of self and other"""
#         return minkowskiDist(self.getFeatures(), other.getFeatures(), 2)

# def compareAnimals(animals, precision):
#     """Assumes animals is a list of animals, precision an int >= 0
#     Builds a table of Euclidean distance between each animal"""
#     #Get labels for columns and rows
#     columnLabels = []
#     for a in animals:
#         columnLabels.append(a.getName())
#     rowLabels = columnLabels[:]
#     tableVals = []
#     #Get distances between pairs of animals
#     #For each row
#     for a1 in animals:
#         row = []
#     #For each column
#         for a2 in animals:
#             if a1 == a2:
#                 row.append('--')
#             else:
#                 distance = a1.distance(a2)
#                 row.append(str(round(distance, precision)))
#         tableVals.append(row)
#     #Produce table
#     table = pylab.table(rowLabels = rowLabels, colLabels = columnLabels, cellText = tableVals, cellLoc = 'center', loc = 'center', colWidths = [0.2]*len(animals))
#     table.scale(1, 2.5)
#     pylab.show()
#     # pylab.savefig('distances')

# rattlesnake = Animal('rattlesnake', [1,1,1,1,0])
# boa = Animal('boa\nconstrictor', [0,1,0,1,0])
# # dartFrog = Animal('dart frog', [1,0,1,0,4])
# dartFrog = Animal('dart frog', [1,0,1,0,1]) # use 1 if it has legs, 0 if it doesn't instead of number of legs
# animals = [rattlesnake, boa, dartFrog]
# # alligator = Animal('alligator', [1,1,0,1,4])
# alligator = Animal('alligator', [1,1,0,1,1]) # use 1 if it has legs, 0 if it doesn't instead of number of legs
# animals.append(alligator)
# compareAnimals(animals, 3)


### Lecture 12: Clustering

# import pylab, random
#
# # Minkowski distance
# def minkowskiDist(v1, v2, p):
#     """Assumes v1 and v2 are equal-length arrays of numbers
#     Returns Minkowski distance of order p between v1 and v2"""
#     dist = 0.0
#     for i in range(len(v1)):
#         dist += abs(v1[i] - v2[i])**p
#     return dist**(1/p)
#
#
# class Example(object):
#     def __init__(self, name, features, label = None):
#         #Assumes features is an array of floats
#         self.name = name
#         self.features = features
#         self.label = label
#     def dimensionality(self):
#         return len(self.features)
#     def getFeatures(self):
#         return self.features[:]
#     def getLabel(self):
#         return self.label
#     def getName(self):
#         return self.name
#     def distance(self, other):
#         return minkowskiDist(self.features, other.getFeatures(), 2)
#     def __str__(self):
#         return self.name +':'+ str(self.features) + ':' + str(self.label)
#
# class Cluster(object):
#     def __init__(self, examples):
#         """Assumes examples a non-empty list of Examples"""
#         self.examples = examples
#         self.centroid = self.computeCentroid()
#     def update(self, examples):
#         """Assume examples is a non-empty list of Examples
#         Replace examples; return amount centroid has changed"""
#         oldCentroid = self.centroid
#         self.examples = examples
#         self.centroid = self.computeCentroid()
#         return oldCentroid.distance(self.centroid)
#     def computeCentroid(self):
#         vals = pylab.array([0.0]*self.examples[0].dimensionality())
#         for e in self.examples: #compute mean
#             vals += e.getFeatures()
#         centroid = Example('centroid', vals/len(self.examples))
#         return centroid
#     def getCentroid(self):
#         return self.centroid
#     def variability(self):
#         totDist = 0.0
#         for e in self.examples:
#             totDist += (e.distance(self.centroid))**2
#         return totDist
#     def members(self):
#         for e in self.examples:
#             yield e
#     def __str__(self):
#         names = []
#         for e in self.examples:
#             names.append(e.getName())
#         names.sort()
#         result = 'Cluster with centroid ' + str(self.centroid.getFeatures()) + ' contains:\n '
#         for e in names:
#             result = result + e + ', '
#         return result[:-2] #remove trailing comma and space
#
# def kmeans(examples, k, verbose = False):
#     #Get k randomly chosen initial centroids, create cluster for each
#     initialCentroids = random.sample(examples, k)
#     clusters = []
#     for e in initialCentroids:
#         clusters.append(Cluster([e]))
#     #Iterate until centroids do not change
#     converged = False
#     numIterations = 0
#     while not converged:
#         numIterations += 1
#         #Create a list containing k distinct empty lists
#         newClusters = []
#         for i in range(k):
#             newClusters.append([])
#         #Associate each example with closest centroid
#         for e in examples:
#         #Find the centroid closest to e
#             smallestDistance = e.distance(clusters[0].getCentroid())
#             index = 0
#             for i in range(1, k):
#                 distance = e.distance(clusters[i].getCentroid())
#                 if distance < smallestDistance:
#                     smallestDistance = distance
#                     index = i
#             #Add e to the list of examples for appropriate cluster
#             newClusters[index].append(e)
#         for c in newClusters: #Avoid having empty clusters
#             if len(c) == 0:
#                 raise ValueError('Empty Cluster')
#         #Update each cluster; check if a centroid has changed
#         converged = True
#         for i in range(k):
#             if clusters[i].update(newClusters[i]) > 0.0:
#                 converged = False
#         if verbose:
#             print('Iteration #' + str(numIterations))
#             for c in clusters:
#                 print(c)
#             print('') #add blank line
#     return clusters
#
# def dissimilarity(clusters):
#     totDist = 0.0
#     for c in clusters:
#         totDist += c.variability()
#     return totDist
# def trykmeans(examples, numClusters, numTrials, verbose = False):
#     """Calls kmeans numTrials times and returns the result with the
#     lowest dissimilarity"""
#     best = kmeans(examples, numClusters, verbose)
#     minDissimilarity = dissimilarity(best)
#     trial = 1
#     while trial < numTrials:
#         try:
#             clusters = kmeans(examples, numClusters, verbose)
#         except ValueError:
#             continue #If failed, try again
#         currDissimilarity = dissimilarity(clusters)
#         if currDissimilarity < minDissimilarity:
#             best = clusters
#             minDissimilarity = currDissimilarity
#         trial += 1
#     return best
#
# def genDistribution(xMean, xSD, yMean, ySD, n, namePrefix):
#     samples = []
#     for s in range(n):
#         x = random.gauss(xMean, xSD)
#         y = random.gauss(yMean, ySD)
#         samples.append(Example(namePrefix+str(s), [x, y]))
#     return samples
# def plotSamples(samples, marker):
#     xVals, yVals = [], []
#     for s in samples:
#         x = s.getFeatures()[0]
#         y = s.getFeatures()[1]
#         pylab.annotate(s.getName(), xy = (x, y), xytext = (x+0.13, y-0.07), fontsize = 'x-large')
#         xVals.append(x)
#         yVals.append(y)
#     pylab.plot(xVals, yVals, marker)
#
# def contrivedTest(numTrials, k, verbose = False):
#     xMean = 3
#     xSD = 1
#     yMean = 5
#     ySD = 1
#     n = 10
#     d1Samples = genDistribution(xMean, xSD, yMean, ySD, n, 'A')
#     plotSamples(d1Samples, 'k^')
#     d2Samples = genDistribution(xMean+3, xSD, yMean+1, ySD, n, 'B')
#     plotSamples(d2Samples, 'ko')
#     pylab.show()
#     clusters = trykmeans(d1Samples+d2Samples, k, numTrials, verbose)
#     print('Final result')
#     for c in clusters:
#         print('', c)
#
# def contrivedTest2(numTrials, k, verbose = False):
#     xMean = 3
#     xSD = 1
#     yMean = 5
#     ySD = 1
#     n = 8
#     d1Samples = genDistribution(xMean,xSD, yMean, ySD, n, 'A')
#     plotSamples(d1Samples, 'k^')
#     d2Samples = genDistribution(xMean+3,xSD,yMean, ySD, n, 'B')
#     plotSamples(d2Samples, 'ko')
#     d3Samples = genDistribution(xMean, xSD, yMean+3, ySD, n, 'C')
#     plotSamples(d3Samples, 'kx')
#     clusters = trykmeans(d1Samples + d2Samples + d3Samples, k, numTrials, verbose)
#     pylab.ylim(0,11)
#     pylab.show()
#     print('Final result has dissimilarity', round(dissimilarity(clusters), 3))
#     for c in clusters:
#         print('', c)

# contrivedTest(50, 2, True)
# contrivedTest2(40, 2)
# contrivedTest2(40, 3)
# contrivedTest2(40, 6)


### Lecture 13: Classification

import random

import pylab

from Lecture11.lectureCode import stdDev


def accuracy(truePos, falsePos, trueNeg, falseNeg):
    numerator = truePos + trueNeg
    denominator = truePos + trueNeg + falsePos + falseNeg
    return numerator / denominator


def sensitivity(truePos, falseNeg):
    try:
        return truePos / (truePos + falseNeg)
    except ZeroDivisionError:
        return float('nan')


def specificity(trueNeg, falsePos):
    try:
        return trueNeg / (trueNeg + falsePos)
    except ZeroDivisionError:
        return float('nan')


def posPredVal(truePos, falsePos):
    try:
        return truePos / (truePos + falsePos)
    except ZeroDivisionError:
        return float('nan')


def negPredVal(trueNeg, falseNeg):
    try:
        return trueNeg / (trueNeg + falseNeg)
    except ZeroDivisionError:
        return float('nan')


def getStats(truePos, falsePos, trueNeg, falseNeg, toPrint=True):
    accur = accuracy(truePos, falsePos, trueNeg, falseNeg)
    sens = sensitivity(truePos, falseNeg)
    spec = specificity(trueNeg, falsePos)
    ppv = posPredVal(truePos, falsePos)
    if toPrint:
        print(' Accuracy =', round(accur, 3))
        print(' Sensitivity =', round(sens, 3))
        print(' Specificity =', round(spec, 3))
        print(' Pos. Pred. Val. =', round(ppv, 3))
    return accur, sens, spec, ppv


### Lecture 13: Classification - Titanic Example

import sklearn.linear_model


# Minkowski distance
def minkowskiDist(v1, v2, p):
    """Assumes v1 and v2 are equal-length arrays of numbers
    Returns Minkowski distance of order p between v1 and v2"""
    dist = 0.0
    for i in range(len(v1)):
        dist += abs(v1[i] - v2[i]) ** p
    return dist ** (1 / p)


def divide80_20(examples):
    sampleIndices = random.sample(range(len(examples)), len(examples) // 5)
    trainingSet, testSet = [], []
    for i in range(len(examples)):
        if i in sampleIndices:
            testSet.append(examples[i])
        else:
            trainingSet.append(examples[i])
    return trainingSet, testSet


def applyModel(model, testSet, label, prob=0.5):
    # Create vector containing feature vectors for all test examples
    testFeatureVecs = [e.getFeatures() for e in testSet]
    probs = model.predict_proba(testFeatureVecs)
    truePos, falsePos, trueNeg, falseNeg = 0, 0, 0, 0
    for i in range(len(probs)):
        if probs[i][1] > prob:
            if testSet[i].getLabel() == label:
                truePos += 1
            else:
                falsePos += 1
        else:
            if testSet[i].getLabel() != label:
                trueNeg += 1
            else:
                falseNeg += 1
    return truePos, falsePos, trueNeg, falseNeg


def buildROC(model, testSet, label, title, plot=True):
    xVals, yVals = [], []
    p = 0.0
    while p <= 1.0:
        truePos, falsePos, trueNeg, falseNeg = applyModel(model, testSet, label, p)
        xVals.append(1.0 - specificity(trueNeg, falsePos))
        yVals.append(sensitivity(truePos, falseNeg))
        p += 0.01
    auroc = sklearn.metrics.auc(xVals, yVals, True)
    if plot:
        pylab.plot(xVals, yVals)
        pylab.plot([0, 1], [0, 1, ], '--')
        pylab.title(title + ' (AUROC = ' + str(round(auroc, 3)) + ')')
        pylab.xlabel('1 - Specificity')
        pylab.ylabel('Sensitivity')
    return auroc


class Passenger(object):
    features = ('C1', 'C2', 'C3', 'age', 'male gender')

    def __init__(self, pClass, age, gender, survived, name):
        self.name = name
        self.featureVec = [0, 0, 0, age, gender]
        self.featureVec[pClass - 1] = 1
        self.label = survived
        self.cabinClass = pClass

    def distance(self, other):
        return minkowskiDist(self.veatureVec, other.featureVec, 2)

    def getClass(self):
        return self.cabinClass

    def getAge(self):
        return self.featureVec[3]

    def getGender(self):
        return self.featureVec[4]

    def getName(self):
        return self.name

    def getFeatures(self):
        return self.featureVec[:]

    def getLabel(self):
        return self.label


def summarizeStats(stats):
    """assumes stats a list of 5 floats: accuracy, sensitivity,
    specificity, pos. pred. val, ROC"""

    def printStat(X, name):
        mean = round(sum(X) / len(X), 3)
        std = stdDev(X)
        print(' Mean', name, '=', str(mean) + ',', '95% confidence interval =', round(1.96 * std, 3))

    accs, sens, specs, ppvs, aurocs = [], [], [], [], []
    for stat in stats:
        accs.append(stat[0])
        sens.append(stat[1])
        specs.append(stat[2])
        ppvs.append(stat[3])
        aurocs.append(stat[4])
    printStat(accs, 'accuracy')
    printStat(sens, 'sensitivity')
    printStat(accs, 'specificity')
    printStat(sens, 'pos. pred. val.')
    printStat(aurocs, 'AUROC')


def testModels(examples, numTrials, printStats, printWeights):
    survived = 1  # value of label indicating survived
    stats, weights = [], [[], [], [], [], []]
    for i in range(numTrials):
        training, testSet = divide80_20(examples)
        featureVecs, labels = [], []
        for e in training:
            featureVecs.append(e.getFeatures())
            labels.append(e.getLabel())
        featureVecs = pylab.array(featureVecs)
        labels = pylab.array(labels)
        model = sklearn.linear_model.LogisticRegression().fit(featureVecs, labels)

        for i in range(len(Passenger.features)):
            weights[i].append(model.coef_[0][i])
        truePos, falsePos, trueNeg, falseNeg = applyModel(model, testSet, survived, 0.5)
        auroc = buildROC(model, testSet, survived, None, False)
        tmp = getStats(truePos, falsePos, trueNeg, falseNeg, False)
        stats.append(tmp + (auroc,))
    print('Averages for', numTrials, 'trials')
    if printWeights:
        for feature in range(len(weights)):
            featureMean = sum(weights[feature]) / numTrials
            featureStd = stdDev(weights[feature])
            print(' Mean weight of', Passenger.features[feature], '=', str(round(featureMean, 3)) + ',',
                  '95% confidence interval =', round(1.96 * featureStd, 3))
    if printStats:
        summarizeStats(stats)


def testModels2(examples, numTrials, printStats, printWeights):
    stats, weights = [], [[], [], [], [], []]
    for i in range(numTrials):
        training, testSet = divide80_20(examples)
        xVals, yVals = [], []
        for e in training:
            xVals.append(e.getFeatures())
            yVals.append(e.getLabel())
        xVals = pylab.array(xVals)
        yVals = pylab.array(yVals)
        model = sklearn.linear_model.LogisticRegression().fit(xVals, yVals)

        for i in range(len(Passenger.features)):
            weights[i].append(model.coef_[0][i])
        truePos, falsePos, trueNeg, falseNeg = applyModel(model, testSet, 1, 0.5)
        auroc = buildROC(model, testSet, 1, None, False)
        tmp = getStats(truePos, falsePos, trueNeg, falseNeg, False)
        stats.append(tmp + (auroc,))
    print('Averages for', numTrials, 'trials')
    if printWeights:
        for feature in range(len(weights)):
            featureMean = sum(weights[feature]) / numTrials
            featureStd = stdDev(weights[feature])
            print(' Mean weight of', Passenger.features[feature], '=', str(round(featureMean, 3)) + ',',
                  '95% confidence interval =', round(1.96 * featureStd, 3))
    if printStats:
        summarizeStats(stats)


testModels2(examples, 100, True, False)
