###########################
# 6.0002 Problem Set 1b: Space Change
# Name:
# Collaborators:
# Time:
# Author: charz, cdenise

#================================
# Part B: Golden Eggs
#================================

# Problem 1
def dp_make_weight(egg_weights, target_weight, memo = {}):
    """
    Find number of eggs to bring back, using the smallest number of eggs. Assumes there is
    an infinite supply of eggs of each weight, and there is always a egg of value 1.
    
    Parameters:
    egg_weights - tuple of integers, available egg weights sorted from smallest to largest value (1 = d1 < d2 < ... < dk)
    target_weight - int, amount of weight we want to find eggs to fit
    memo - dictionary, OPTIONAL parameter for memoization (you may not need to use this parameter depending on your implementation)
    
    Returns: int, smallest number of eggs needed to make target weight
    """
    def myFunc(egg_weights, target_weight, memo = {}):
        # copy by sorting it in descending order of weight
        eggWeightsCopy = sorted(egg_weights, reverse = True)
        if (len(eggWeightsCopy), target_weight) in memo:
            result = memo[(len(eggWeightsCopy), target_weight)]
        elif len(eggWeightsCopy) == 0 or target_weight == 0:
            result = (0, ())
        elif eggWeightsCopy[0] > target_weight:
            # explore right branch because you can't take that egg
            result = myFunc(eggWeightsCopy[1:], target_weight, memo)
        else:
            # explore left branch because you can take that item
            nextEggToTake = eggWeightsCopy[0]
            withVal, withTaken = myFunc(eggWeightsCopy, target_weight - nextEggToTake, memo)
            withVal += nextEggToTake
    
            result = (withVal, withTaken + (nextEggToTake,))

        memo[(len(eggWeightsCopy), target_weight)] = result
        return result
    
    finalWeight, finalItemList = myFunc(egg_weights, target_weight)
    return finalWeight, finalItemList


def dp_make_weight2(egg_weights, target_weight, memo = {}):
    """
    Find number of eggs to bring back, using the smallest number of eggs. Assumes there is
    an infinite supply of eggs of each weight, and there is always a egg of value 1.
    
    Parameters:
    egg_weights - tuple of integers, available egg weights sorted from smallest to largest value (1 = d1 < d2 < ... < dk)
    target_weight - int, amount of weight we want to find eggs to fit
    memo - dictionary, OPTIONAL parameter for memoization (you may not need to use this parameter depending on your implementation)
    
    Returns: int, smallest number of eggs needed to make target weight
    """
    def myFunc(egg_weights, target_weight, memo = {}):
        if (len(egg_weights), target_weight) in memo:
            result = memo[(len(egg_weights), target_weight)]
        elif len(egg_weights) == 0 or target_weight == 0:
            result = (0, ())
        elif egg_weights[0] > target_weight:
            # explore right branch because you can't take that egg
            result = myFunc(egg_weights[1:], target_weight, memo)
        else:
            # explore left branch because you can take that item
            nextEggToTake = egg_weights[0]
            withVal, withTaken = myFunc(egg_weights, target_weight - nextEggToTake, memo)
            withVal += nextEggToTake
            # explore right branch
            withoutVal, withoutTaken = myFunc(egg_weights[1:], target_weight, memo)
            
            if withVal > withoutVal:
                result = (withVal, withTaken + (nextEggToTake,))
            else:
                result = (withoutVal, withoutTaken)

        memo[(len(egg_weights), target_weight)] = result
        return result
    
    # return myFunc(egg_weights, target_weight)
    
    finalWeight, finalItemList = myFunc(egg_weights, target_weight)
    return finalWeight, finalItemList


# EXAMPLE TESTING CODE, feel free to add more if you'd like
if __name__ == '__main__':
    egg_weights = (1, 5, 10, 25)
    n = 99
    print("Egg weights = (1, 5, 10, 25)")
    print("n = 99")
    print("Expected ouput: 9 (3 * 25 + 2 * 10 + 4 * 1 = 99)")
    print("Actual output:", dp_make_weight(egg_weights, n))
    print()