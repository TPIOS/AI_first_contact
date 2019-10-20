# import re
# def is_number(num):
#     if num == "'" or num == "'s": return False
#     pattern = re.compile(r'^[-+]?[\']?([0-9]+[.,-/]?)*[s]?$')
#     result = pattern.match(num)
#     if result:
#         return True
#     else:
#         return False
# while(True):
#     s = input("Please input: ")
#     print(is_number(s))
s = "d"
print(s.endswith("ed"))