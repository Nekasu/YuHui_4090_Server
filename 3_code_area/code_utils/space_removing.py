
a = input("Enter the string: ")
x = [x  for x in a if x not in [' ', '', ''] ]
# convert list into string

x = ''.join(x)

print(x)