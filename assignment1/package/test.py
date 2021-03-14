from package import utility
from package import part1
from package import part2
from package import part3

#This file contains the method to test my assignment1 work.
#It runs the main method of part1, part2, and part3.

def test():
    data = utility.load_data()
    part1.run(data)
    part2.run(data)
    part3.run(data)
