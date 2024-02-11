import delgrada.scalar as dgs
from delgrada import Scalar
import math

x = Scalar(4)
y = Scalar(0.5)
z = x ** y
out = z + x
print(out)

out.backprop()
print(x.grad, y.grad, z.grad, '\n')

a = Scalar(2)
b = dgs.exp(a**2)
print(b)
b.backprop()
print(a.grad, '\n')

c = Scalar(math.pi/2)
d = dgs.cos(c)
print(d)
d.backprop()
print(c.grad, '\n')
