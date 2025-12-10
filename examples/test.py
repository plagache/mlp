import numpy as np
from tinygrad import Tensor as Tiny, nn
from dlf.dataset import load_dataset
from tinygrad import Device

print(Device.DEFAULT)

x = Tiny.eye(3, requires_grad=True)
y = Tiny([[2.0, 0.5, 0.1]], requires_grad=True)
d = x.dot(y.T)
# d = x.add(y.T)
s = d.sum()
# l = y.log()
# s = l.sum()
# e = y.exp()
# s = e.sum()
# s = y.sum()
s.backward()

print(f"{x.tolist()=}")
print(f"{d.tolist()=}")
# print(f"{l.tolist()=}")
# print(f"{e.tolist()=}")
print(f"{s.tolist()=}")

print(f"{s.grad.tolist()=}")  # dz/dy
print(f"{d.grad.tolist()=}")  # dz/dy
# print(f"{l.grad.tolist()=}")  # dz/dx
# print(f"{e.grad.tolist()=}")  # dz/dx
print(f"{y.grad.tolist()=}")  # dz/dy
print(f"{y.grad.shape=}")  # dz/dy
print(f"{x.grad.tolist()=}")  # dz/dy

from dlf.tensor import Tensor
xd = Tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
yd = Tensor([[2.0, 0.5, 0.1]])
# dd = yd.DOT(xd)
dd = xd.DOT(yd.T)
# dd = xd.ADD(yd.T)
sd = dd.SUM()
# ld = yd.LOG()
# sd = ld.SUM()
# ed = yd.EXP()
# sd = ed.SUM()
# sd = yd.SUM()
sd.backward()

print(f"{xd=}")
print(f"{dd=}")
# print(f"{ld=}")
# print(f"{ed=}")
print(f"{sd=}")

print(f"{sd.grad=}")
print(f"{dd.grad=}")
# print(f"{ld.grad=}")
# print(f"{ed.grad=}")
print(f"{yd.grad=}")
print(f"{yd.grad.shape=}")
print(f"{yd.data.shape=}")
print(f"{xd.grad=}")
