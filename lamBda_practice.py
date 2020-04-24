''' 람다식 연습하기 '''

x = lambda a : a+10
print(x(5))

y = lambda a,b,c : a*b/c
print(y(10,10,20))

# 람다식 이용한 함수 만들기
def plus(n):
    return lambda a : a + n

first_plus = plus(10)
second_plus = plus(60)
print(first_plus(40))
print(second_plus(40))

# lambda 식 이용해서 map 함수 (iterator) 사용하기
l1 = [1,2,3,4,5]
maps = map(lambda i : i*i, l1)
next(maps)
next(maps)
next(maps)
next(maps)
next(maps)
next(maps)

x = [1,2,3,4]
y = [10,20,30,40]
map_list = map(lambda a, b : a*b, x, y)
print(map_list)
for x in map_list:
    print(x)

# lambda식 이용해서 filter 사용해보기
a = [1,2,3,4,5,6,7,8,9,10]
map_filter = filter(lambda x :x%2 == 0, a)
for k in map_filter:
    print(k)
