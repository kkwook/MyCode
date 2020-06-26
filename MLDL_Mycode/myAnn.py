''' 확률적 경사하강법(SGD, Stochastic Gradient Descent)

미니 배치 : train 데이터에서 무작위로 데이터를 뽑는다. (=미니 배치를 선정한다.)
기울기 계산 : weight 매개변수에 대한 미니 배치의 손실 함수의 기울기를 계산한다.
매개변수 업데이트 : 앞에서 계산한 기울기의 방향으로 weight 매개변수를 업데이트 한다.
반복 : 손실 함수의 값이 어느 정도 작아질 때까지 1, 2, 3을 반복한다. '''

import numpy as np

class AnnFunction:
    def sigmoid(x):
        return 1 / (1 * np.exp(-x))

    def softmax(x):
        c = np.max(x) # oveflow 방지
        exp_x = np.exp(x - c) # single source of truth
        return exp_x / np.sum(exp_x)

    ''' 교차 엔트로피 - 실제 정답과의 오차만을 파악하는 손실함수 '''
    # 1개의 값을 위한 CEE
    # log에 0이 들어갈 경우 -inf가 나오기 때문에, -inf가 나오지 않도록, 아주 작은 숫자를 더해줍니다.
    def cross_entropy_error(x, t):
        delta = 1e-7
        return -np.sum(t*np.log(x+delta))

    # 베치를 위한 CEE
    def cross_entropy_error_batch(x, t):
        delta = 1e-7
        if x.dim == 1: 
            x = x.reshape(1, x.size) # 2차원 배열로 만들어주기
            t = t.reshape(1, t.size)
        
        batch_size = x.shape[0]
        return -np.sum(t*np.log(x+delta)) / batch_size

    def softmax_loss(X, t): # 소프트맥스 손실함수
        y = softmax(X)
        return cross_entropy_error(y, t)

    def numerical_gradient(f, x): # 
        h = 1e-4  # 0.0001
        # np.zeros_like => 이미 있는 array와 동일한 모양과 데이터 형태를 유지한 상태에서 0을 반환 
        grad = np.zeros_like(x)
        ''' np.nditer : 행렬 원소 접근
            다차원 배열을 다차원 iterator로 변환, it.multi_index = 다차원 배열의 인덱스를 튜플로 받는다.
            예를 들어, 2x3 행렬의 경우 : (0,0), (0,1), (0,2), (1,0), (1,1), (1,2)를 차례로 접근하여 받는다. '''
        it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            tmp_val = x[idx]
            x[idx] = float(tmp_val) + h
            fxh1 = f(x)  # f(x+h)

            x[idx] = tmp_val - h
            fxh2 = f(x)  # f(x-h)
            grad[idx] = (fxh1 - fxh2) / (2 * h)

            x[idx] = tmp_val  # 값 복원
            it.iternext()
            ''' it 부분 ==
            for idx in range(x.size):
                tmp_val = x[idx]
                x[idx] = tmp_val + h  <- f(x+h)
                fxh1 = f(x)
                x[idx] = tmp_val -h <- f(x-h)
                fxh2 = f(x)
                grad[idx] = (fxh1 - fxh2) / (2+h)
                x[idx] = imp_val '''
        return grad
    
class TwoLayerNet:
    # 신경망의 초기에서 입력 크기, 은닉 크기, 출력 크기, std의 가중치 초기를 구함 
    # import numpy as np
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {} # 신경망의 매개변수 (가중치 초기화)
        
        # 가중치를 정규분포를 따르는 난수로 초기화
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(input_size, hidden_size)
        # 편향을 0으로 초기화
        self.params['b1'] = np.zeros(hidden_size)
        self.params['b2'] = np.zeros(hidden_size)

    # 신겸망을 거쳐 예측하는 함수, 분류 문제의 결과로 softmax된 값을 반환
    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']        
        
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        return y
        
    # 손실함수 값을 구하는 함수
    def loss(self, x, t):
        y = self.predict(x) # 신경망을 거쳐온 결과 값
        return cross_entropy_error_batch(y, t) # 손실함수의 값 
    
    def accuracy(self, x, t):
        y = self.predict(x) # 예측값
        y = np.argmax(y, axis = 1) # 예측 정답 인덱스
        t = np.argmax(t, axis = 1) # 실제 정답 인덱스
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    # 기울기 구하는 함수
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t) # 가중치를 받아 loss를 구하는 함수
        
        # 손실함수에 대해 기울기를 구함
        grads- {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads
        