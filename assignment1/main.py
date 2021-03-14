import package.test as test
import package.part1 as part1
import package.part2 as part2
import package.part3 as part3

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
    part1.run(data) 

    print("Running part2:")
    part2.run(data) 

    print("\nAs we can see, the polynomial regression model performs better when using RM as the feature to predict MEDV!") 
    print("With polynomial regression (deg2), the RMSE score and R-squared score is 5.799 and 0.6596 (compared to 6.38 and 0.5877 of the linear regression model.")
    print("Furthermore, with polynomial regression of degree20, the RMSE score and R-squared score is 5.489 and 0.695, which is even better!")
    print("However, this is most likely an overfit on the data, as polynomial regression of degree20 has 21 features! And as we can see from the plots, it is trying to fit everything.")

    print("\nRunning part3:")
    part3.run(data) 

    print("With multiple regression with 6 features (INDUS, NOX, RM, TAX, PTRATIO, LSTAT), the RMSE and R-squared score is the best out of all the models we explored.")
    print("With multiple regression with our 6 features the RMSE and R-squared score are 5.47 and 0.6968 respectively")
    print("And, the adjusted R2 of this multiple regression is 0.6922, which is just a little less than the r2_score, which shows that the features we selected are significant!")

    print("\nNow running my extra exploration!")
    print("\nRunning part1 extras:")
    part1.run_loop(data)

    print("\nRunning part2 extras:")
    part2.run_loop(data)

if __name__ == "__main__":
    # execute only if run as a script
    main()
