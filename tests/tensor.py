from delgrada import Tensor

print('Test 1: Matmul test')
a = Tensor([[[1.,.2],[.3,.4]], [[.5,.6],[.7,.8]]])
b = Tensor([[[3,4],[5,6]], [[7,8],[9,10]]])
print(a)
print(b)
c = a@b
print(repr(c))
c.backprop()
print(a.grad)
print(b.grad, '\n')

print('\nTest 2: Simple broadcasting test')
s = Tensor(20)
v = Tensor([1,2,3])
w = s * v
w.backprop()
print(s.grad)
print(v.grad)

print('\nTest 3: Matmul behaviour test (post)')
W = Tensor([[[1,2],[3,4]], [[1,2],[3,4]]][0])
x = Tensor([10,20])
print(W)
print(x)
y = W@x
print(y)
y.backprop()
print(W.grad)
print(x.grad)

print('\nTest 4: Matmul behaviour test (pre)')
x = Tensor([10,20])
W = Tensor([[[1,2],[3,4]], [[1,2],[3,4]]][0])
print(x)
print(W)
y = x@W
print(y)
y.backprop()
print(W.grad)
print(x.grad)

print('\nTest 5: Matmul broadcasting test')
W = Tensor([[[1,2],[3,4]], [[1,2],[3,4]]])
x = Tensor([10,20])
print(W)
print(x)
y = W@x
print(y)
y.backprop()
print(W.grad)
W = Tensor([[[1,2],[3,4]], [[1,2],[3,4]]])
x = Tensor([10,20])
print(W)
print(x)
y = W@x
print(y)
y.backprop()
print(W.grad)
print(x.grad)
