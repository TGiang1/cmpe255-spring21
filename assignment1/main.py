import package.test as test
import package.part1 as part1
import package.part2 as part2

import package.utility as ut

#This is the main file. 
#It imports the assignment1 package's modules.
#part1 module contains methods related to part1 instructions
#part2 module contains methods related to part2 instructions
#part3 module contains methods related to part3 instructions

#utility module contains utility methods

#test module contains a method to run the main methods of part1, part2, and part3.

def main():
    data = ut.load_data() #Make sure housing.csv is in the same directory as this file!
    print("Now running part1, part2, and part3 requirements!")
    print("Running part1:")
    r2_score = part1.run(data) 

    print("Running part2:")
    part2.run(data) 

    print("\nAs we can see, the polynomial regression model performs better when using RM as the feature to predict \
    MEDV! With polynomial regression, the RMSE score and R-squared score is 5.799 and 0.6596 (compared to 6.38 and 0.5877 of the linear regression model.")

    # print("Now running my extra exploration!")
    # print("Running part1 extras:")
    # part1.run_loop(data)
    # part2.run_loop(data)

if __name__ == "__main__":
    # execute only if run as a script
    main()
