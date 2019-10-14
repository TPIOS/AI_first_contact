def f(a):
    b = a.copy()
    b["3"] += 1
a = {"1":2, "2":3, "3":4}
print(a)
f(a)
print(a)