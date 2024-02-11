from delgrada import Tensor

a = Tensor([[[1.,.2],[.3,.4]], [[.5,.6],[.7,.8]]])
b = Tensor([[[3,4],[5,6]], [[7,8],[9,10]]])
print(a)
print(b)
c = a@b
print(repr(c))

c.backprop()
print(a.grad)
print(b.grad)


