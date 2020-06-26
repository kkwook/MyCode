''' 신경망 기초 '''

''' weight : 가중치 -> 신호 전달 역할
    bias : 편향 ->  '''




import numpy as np
def AND(x1, x2):
    w1, w2, bias = 0.5, 0.5, 0.7
    tmp = x1 * w1 + x2 * w2
    if tmp <= bias:
        return 0
    elif tmp > bias:
        return 1


AND(0, 0)
AND(0, 1)
AND(1, 0)
AND(1, 1)


def OR(x1, x2):
    w1, w2, bias = 0.5, 0.5, 0
    b = -0.2
    tmp = x1 * w1 + x2 * w2
    if tmp - bias <= 0:
        return 0
    elif tmp - bias > 0:
        return 1


OR(0, 0)
OR(0, 1)
OR(1, 0)
OR(1, 1)


def OR2(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w * x)
    if tmp + b <= 0:
        return 0
    else:
        return 1


OR2(0, 0)
OR2(0, 1)
OR2(1, 0)
OR2(1, 1)


def NAND(x1, x2):
    w1, w2, bias = -0.5, -0.5, -0.7
    tmp = x1 * w1 + x2 * w2
    if tmp - bias <= 0:
        return 0
    elif tmp - bias > 0:
        return 1


NAND(0, 0)
NAND(1, 0)
NAND(0, 1)
NAND(1, 1)


def myPerceptron(x1, x2, w1, w2, bias):
    tep = x1 * w2 + x2 * w2
    if tep - bias <= 0:
        return 0
    else:
        return 1

''' AND '''
myPerceptron(0, 0, 0.5, 0.5, 0.7)
myPerceptron(0, 1, 0.5, 0.5, 0.7)
myPerceptron(1, 0, 0.5, 0.5, 0.7)
myPerceptron(1, 1, 0.5, 0.5, 0.7)
''' OR '''
myPerceptron(0, 0, 0.5, 0.5, 0)
myPerceptron(0, 1, 0.5, 0.5, 0)
myPerceptron(1, 0, 0.5, 0.5, 0)
myPerceptron(1, 1, 0.5, 0.5, 0)
''' NAND '''
myPerceptron(0, 0, -0.5, -0.5, -0.7)
myPerceptron(0, 1, -0.5, -0.5, -0.7)
myPerceptron(1, 0, -0.5, -0.5, -0.7)
myPerceptron(1, 1, -0.5, -0.5, -0.7)

''' 퍼셉트론 xor 구현 못함 '''

def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y

XOR(0, 0)
XOR(1, 0)
XOR(0, 1)
XOR(1, 1)
