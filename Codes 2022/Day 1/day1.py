
# Getting data
with open('day1.in') as file:
    X = [i for i in file.read().strip().split("\n")]


# print(data)


# Traversing every STRING in our DATA
Q = []
for elf in ('\n'.join(X)).split('\n\n'):
    q = 0
    for x in elf.split('\n'):
        q += int(x)
    Q.append(q)
Q = sorted(Q)
print(Q[-1])
print(Q[-1]+Q[-2]+Q[-3])