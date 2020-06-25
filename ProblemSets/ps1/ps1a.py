###########################
# 6.0002 Problem Set 1a: Space Cows 
# Name:
# Collaborators:
# Time:

from ps1_partition import get_partitions
import time

#================================
# Part A: Transporting Space Cows
#================================

# Problem 1
def load_cows(filename):
    """
    Read the contents of the given file.  Assumes the file contents contain
    data in the form of comma-separated cow name, weight pairs, and return a
    dictionary containing cow names as keys and corresponding weights as values.

    Parameters:
    filename - the name of the data file as a string

    Returns:
    a dictionary of cow name (string), weight (int) pairs
    """
    # open the file for reading
    file = open(filename, 'r')
    # create an empty dict for cows
    cowsData = {}
    # read the content of the files into the dictionary
    for line in file:
        contents = line.strip().split(',')
        cowsData[contents[0]] = contents[1]
    return cowsData
    
cows = load_cows('ps1_cow_data.txt')
# cows = load_cows('ps1_cow_data_2.txt')

# Problem 2
def greedy_cow_transport(cows,limit=10):
    """
    Uses a greedy heuristic to determine an allocation of cows that attempts to
    minimize the number of spaceship trips needed to transport all the cows. The
    returned allocation of cows may or may not be optimal.
    The greedy heuristic should follow the following method:

    1. As long as the current trip can fit another cow, add the largest cow that will fit
        to the trip
    2. Once the trip is full, begin a new trip to transport the remaining cows

    Does not mutate the given dictionary of cows.

    Parameters:
    cows - a dictionary of name (string), weight (int) pairs
    limit - weight limit of the spaceship (an int)
    
    Returns:
    A list of lists, with each inner list containing the names of cows
    transported on a particular trip and the overall list containing all the
    trips
    """
    cowsCopy = sorted(cows.items(), key = lambda cows : cows[1], reverse = True)
    allTrips = []
    
    def singleTrip(cowsData):
        currentWeight = 0
        singleTrip = []
        for i in range(len(cowsData)):
            if (currentWeight + int(cowsData[i][1])) <= limit:
                singleTrip.append(cowsData[i])
                currentWeight += int(cowsData[i][1])
        return singleTrip
    
    
    while len(cowsCopy) > 0:
        if len(cowsCopy) == 1:
            allTrips.append([cowsCopy[0][0]])
            break
        else:
            trip = singleTrip(cowsCopy)
            cowsInCurrentTrip = []
            for cow in trip:
                cowsInCurrentTrip.append(cow[0])
                cowsCopy.remove(cow)
            allTrips.append(cowsInCurrentTrip)
                
    return allTrips
    


# print(greedy_cow_transport(cows))

# Problem 3
def brute_force_cow_transport(cows,limit=10):
    """
    Finds the allocation of cows that minimizes the number of spaceship trips
    via brute force.  The brute force algorithm should follow the following method:

    1. Enumerate all possible ways that the cows can be divided into separate trips 
        Use the given get_partitions function in ps1_partition.py to help you!
    2. Select the allocation that minimizes the number of trips without making any trip
        that does not obey the weight limitation
            
    Does not mutate the given dictionary of cows.

    Parameters:
    cows - a dictionary of name (string), weight (int) pairs
    limit - weight limit of the spaceship (an int)
    
    Returns:
    A list of lists, with each inner list containing the names of cows
    transported on a particular trip and the overall list containing all the
    trips
    """
    def calcWeightOfTrip(tripList):
        totalWeight = 0
        for cow in tripList:
            totalWeight += int(cows[cow])
        return totalWeight

    possPartitions = []
    for partition in get_partitions(cows.keys()):
        add = True
        for trip in partition:
            if calcWeightOfTrip(trip) > limit:
                add = False
                break
        if add:
            # return partition
            possPartitions.append(partition)

    minNumOfTrips = min(possPartitions, key = len)
        
    return minNumOfTrips
        

# print(brute_force_cow_transport(cows))


# Problem 4
def compare_cow_transport_algorithms():
    """
    Using the data from ps1_cow_data.txt and the specified weight limit, run your
    greedy_cow_transport and brute_force_cow_transport functions here. Use the
    default weight limits of 10 for both greedy_cow_transport and
    brute_force_cow_transport.
    
    Print out the number of trips returned by each method, and how long each
    method takes to run in seconds.

    Returns:
    Does not return anything.
    """
    start = time.time()
    resultGreedy = greedy_cow_transport(cows)
    end = time.time()
    print('Testing Greedy algorithm...')
    print({'Number Of Trips' : len(resultGreedy), 'Time passed' : end - start})
    
    start = time.time()
    resultBrute = brute_force_cow_transport(cows)
    end = time.time()
    print('Testing Brute Force algorithm...')
    print({'Number Of Trips' : len(resultBrute), 'Time passed' : end - start})
    
compare_cow_transport_algorithms()


