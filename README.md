# Latin-Square-Filler
The code allows a user to fill an n by n Latin Square by inputting n-1 colors such that no 2 same colors lie on a row or column

# How to play
An n by n Latin Square is an n by n grid whose cells are filled with the values 1 to n such that each value appears once in every row or column. The LatinSquareFiller allows
users to enter n-1 colors, and generates a Latin Square if no two same colors lie on a single row or column. To fill distinct colors, simply click the cell you wish to fill.
To fill a cell with a color of a previously filled cell, click that cell, and then click whichever cell you wish to fill. 

[LatinSquareFillerEx.webm](https://user-images.githubusercontent.com/107969255/183511290-ac234c7d-152f-401b-9a47-53c8b0787fda.webm)


# Implementation

The code is based of a proof of Smetaniuk's theorem, which states that any admissible n by n square (one in which no two cells in the same row or column
have the same value) with at most n-1 cells filled can be completed to an n by n Latin Square. The code uses the Hopcroft-Karp matching algorithm to ensure no two same colors
are in the same row or column. Swapping algorithms are used to recursively build bigger Latin squares from smaller ones when the number of distinct colors inputted by the user
is greater than half the length of the square.

# Improvements

The filler is for the most part succesful. However, there do exist a few cases where the following errors occur. The errors seem to occur randomly.
- RecursionError: Recursion depth exceeded during depth first search for Hopcroft-Karp
- Class initialization error: Backtracking fails during the breadth first search
