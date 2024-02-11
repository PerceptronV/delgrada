import delgrada as dg
from delgrada import Tensor
import numpy as np

x = Tensor(4)
y = Tensor(0.5)
z = x ** y
out = z + x
print(out)

out.backprop()
print(x.grad, y.grad, z.grad, '\n')

a = Tensor(2)
b = dg.exp(a**2)
print(b)
b.backprop()
print(a.grad, '\n')

c = Tensor(np.pi/2)
d = dg.cos(c)
print(d)
d.backprop()
print(c.grad, '\n')
