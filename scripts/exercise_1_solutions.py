# Practical Exercise 1: Python Fundamentals

# Q1: Write statement to call the function.
def Add():
    X = 10 + 20
    print(X)
# Solution:
Add()

# Q2: Write statement to call the function.
def Add(X, Y):
    Z = X + Y
    print(Z)
# Solution:
Add(10, 20)

# Q3: Write statement to call the function.
def Add(X, Y):
    Z = X + Y
    return Z
# Solution:
C = Add(10, 20)
print("Total =", C)

# Q4: Which Line Number Code will never execute?
def Check(num): # Line 1
    if num % 2 == 0: # Line 2
        print("Hello") # Line 3
        return True # Line 4
        print("Bye") # Line 5 (Unreachable)
    else: # Line 6
        return False # Line 7
# Solution: Line 5

# Q5: What will be the output of following code?
def Cube(n):
    print(n * n * n)
n = 10
print("Q5 Output:")
Cube(n)
print(Cube(n))

# Q6: Identify the error in CalculateInterest(Principal, Rate=.06, Time)
# Solution:
# Error: Non-default argument 'Time' follows default argument 'Rate'.
# Solution: Change order to (Principal, Time, Rate=.06) or provide default for Time.

# Q7: Call the function using KEYWORD ARGUMENT with values 100 and 200
def Swap(num1, num2):
    num1, num2 = num2, num1
    print(num1, num2)
# Solution:
Swap(num1=100, num2=200)

# Q8: Which line number of code(s) will not work and why?
def Interest(P, R, T=7):
    I = (P * R * T) / 100
    print(I)
# Interest(20000, .08, 15) # Line 1: Ok
# Interest(T=10, 20000, .075) # Line 2: Error (Positional argument follows keyword argument)
# Interest(50000, .07) # Line 3: Ok
# Interest(P=10000, R=.06, Time=8) # Line 4: Error (Unexpected keyword 'Time', should be 'T')
# Interest(80000, T=10) # Line 5: Error (Missing required positional argument 'R')

# Q9: Output of the code?
def Calculate(A, B, C):
    return A * 2, B * 2, C * 2
val = Calculate(10, 12, 14)
print("Q9 Output:")
print(type(val))
print(val)

# Q10: Local vs Global Variables
# Solution:
# Local variables: Defined inside a function, accessible only within that function.
# Global variables: Defined outside any function, accessible throughout the module.

# Q11: Output of the code?
def check():
    num = 50
    print(num)
num = 100
print("Q11 Output:")
print(num)
check()
print(num)

# Q12: Output of the code?
def check():
    global num
    num = 1000
    print(num)
num = 100
print("Q12 Output:")
print(num)
check()
print(num)

# Q13: Output of the code?
print("Q13 Output:")
print("Welcome!")
print("Iam " + __name__)

# Q14: Output for display("EXAM2025.com")?
# m="exam$$$$*COM"
# Solution: exam$$$$*COM

# Q15: Output of Alter(A, B)?
def Alter(M, N=50):
    M = M + N
    N = M - N
    print(M, "@", N)
    return M
A = 200
B = 100
print("Q15 Output:")
A = Alter(A, B)
print(A, "#", B)
B = Alter(B)
print(A, "@", B)

# Q16: Output of Total(4), Total(7), Total()?
# Total(4): range(1,5) -> odd 1, 3 -> Sum=4
# Total(7): range(1,8) -> odd 1, 3, 5, 7 -> Sum=16
# Total(): defaults to 10 -> odd 1, 3, 5, 7, 9 -> Sum=25

# Q17: Output of Change()?
# X=100
# Change(): P=10, Q=25 -> X=150 -> prints 10 # 25 $ 185
# Change(18, 50): P=18, Q=50 -> X=250 -> prints 18 # 50 $ 318
# Change(30, 100): P=30, Q=100 -> X=350 -> prints 30 # 100 $ 480

# Q18: Output of invoke()?
# a=100 -> show() a=200 -> invoke() a=500 -> print(a)
# Solution: 500

# Q19: Output of drawline()?
# drawline() -> $$$$$
# drawline('@', 10) -> @@@@@@@@@@
# drawline(65) -> Error (65 * 5 is not possible for string char) - Wait, if char=65, time=5 (default). int * int = int. 
# BUT char defaults to '$'. If called as drawline(65), char becomes 65. 65*5 = 325.
# drawline(chr(65)) -> AAAAA
# Solution: 
# $$$$$
# @@@@@@@@@@
# 325
# AAAAA

# Q20: Output of Updater?
# A=100, B=30
# Updater(100, 30): A=3, B=3 -> prints 3 $ 3, returns 6
# Outside: A=6, B=30 -> prints 6 # 30
# Updater(30): A=6, B=1 -> prints 6 $ 1, returns 7
# Outside: A=6, B=7 -> prints 6 # 7
# Updater(6): A=1, B=1 -> prints 1 $ 1, returns 2
# Outside: A=2, B=7 -> prints 2 $ 7

# Q21: Output of Fun1(120)?
# Fun1(120): num1=240 -> Fun2(240): returns 120 -> Fun1 returns 120
# Solution: 120

# Q22: Output of Alpha(100)?
# X=50, num=100
# Alpha(100): num1=150, X=70 -> Beta(150): num1=220, X=80 -> Gamma(220): local X=200, num1=420 -> returns 420
# Solution: 420 80

# Q23: Output for list mutation?
# [21, 20, 6, 7, 9, 18, 100, 50, 13]
# 21*2=42, 20//2=10, 6//2=3, 7*2=14, 9*2=18, 18//2=9, 100//2=50, 50//2=25, 13*2=26
# Solution: [42, 10, 3, 14, 18, 9, 50, 25, 26]
