import numpy as np
import math

def square (a) :
  return 2 ** (2.0 * math.log2(a))

def mul (a, b) :
  return -0.5 * ( (a - b) * (a - b) - a * a - b * b)
  #return -0.5 * ( square(abs(a - b)) - square(a) - square(b))

def sigmoid(x):
  return 1 /(1 + 1 / np.exp(10000.0 * x))

ADC_RANGE = 256.0
def dac (binary_value) :
  factor = 1.0
  # factor = len(binary_value)
  total = 0.0
  for bit in binary_value :
    total = total + bit * factor
    factor = factor * 2.0

  return total

def adc (real_value) :
  n = ADC_RANGE
  x = real_value + 0.5
  bits = []
  for i in range(8) :
    n = 0.5 * n
    diff = x - n
    bit = sigmoid(diff)
    # bit = torch.nn.functional.sigmoid(diff)
    one_bit_dac = bit * n
    x = x - one_bit_dac
    bits.append(bit)

  bits.reverse()
  return bits

for i in range(256) :
  print(i, adc(i), dac(adc(i)))

'''
sum = 0.0
for i in range(10):#00000) :
  a = abs(0.1 * np.random.normal())
  b = abs(0.1 * np.random.normal())

  ab = adc(a)
  bb = adc(b)
  # print(a, ab, b, bb)
  aa = dac(ab)
  ba = dac(bb)
  print('Error', aa - a, ba - b)

  diff = a * b - mul(a, b)

  sum += diff

print(sum)
'''
