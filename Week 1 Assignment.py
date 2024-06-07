#Create lower triangular, upper triangular and pyramid containing the "*" character.

# Lower Triangular Pattern
def lower_triangular(n):
    for i in range(1, n + 1):
        print("* " * i)

# Upper Triangular Pattern
def upper_triangular(n):
    for i in range(n, 0, -1):
        print("* " * i)

# Pyramid Pattern
def pyramid(n):
    for i in range(n):
        print(" " * (n - i - 1) + "* " * (i + 1))

# Number of rows for the patterns
n = 5

print("Lower Triangular Pattern:")
lower_triangular(n)
print("\nUpper Triangular Pattern:")
upper_triangular(n)
print("\nPyramid Pattern:")
pyramid(n)
