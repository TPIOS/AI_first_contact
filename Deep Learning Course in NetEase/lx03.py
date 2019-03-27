x = int(input('x= '))
y = int(input('y= '))
z = int(input('z= '))

temp = x
if x < y:
    temp = y
    if y < z:
        temp = z
elif x < z:
    temp = z

print(temp)