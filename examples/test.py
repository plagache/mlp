import numpy as np
from tinygrad import Tensor as Tiny, nn
from dlf.dataset import load_dataset
from tinygrad import Device

print(Device.DEFAULT)

x = Tiny.eye(3, requires_grad=True)
y = Tiny([[2.0, 0.5, 0.1]], requires_grad=True)
d = y.matmul(x)
# l = d.log()
# s = l.sum()
s = d.sum()
s.backward()

print(x.tolist())
print(d.tolist())
# print(l.tolist())
print(s.tolist())

print(f"{s.grad.tolist()=}")  # dz/dy
print(f"{d.grad.tolist()=}")  # dz/dy
print(f"{y.grad.tolist()=}")  # dz/dy
print(f"{x.grad.tolist()=}")  # dz/dy
# print(f"{l.grad.tolist()=}")  # dz/dx

from dlf.tensor import Tensor
xd = Tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
yd = Tensor([[2.0, 0.5, 0.1]])
dd = yd.DOT(xd)
# ld = dd.LOG()
# sd = ld.SUM()
sd = dd.SUM()
sd.backward()

print(f"{xd=}")
print(f"{dd=}")
# print(f"{ld=}")
print(f"{sd=}")

print(f"{sd.grad=}")
print(f"{dd.grad=}")
print(f"{yd.grad=}")
print(f"{xd.grad=}")
# print(f"{ld.grad=}")
