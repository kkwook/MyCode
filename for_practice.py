''' 1. 정수 N개로 이루어진 수열 A와 정수 X가 주어진다. 이때, A에서 X보다 작은 수를 모두 출력하는 프로그램 '''
import random
print(random.randrange(10))
l1=[]
k = int(input("숫자를 입력하시오."))
for x in range(k):
    x = random.randrange(k)
    if k > 1 and k < 10000:
        l1.append(x)
print(l1)
print(len(l1))

l2=[]
for x in range(k):
    if k > 1 and k < 10000:
        x = random.randrange(k)
        if x not in l2:
            l2.append(x)
        else:
            pass
print(l2)
print(len(l2))
