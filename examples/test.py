import numpy as np
from dlf.tensor import Tensor

lst = [4, 4, 5, 2]

l_np = np.array(lst, dtype=np.int64)

print(l_np)
print(type(l_np))
print(lst)

# mv_l = memoryview(l)
mv_l_np = memoryview(l_np)
b_l_np = bytearray(l_np)
b_l = bytearray(lst)

print(b_l)
print(b_l_np)
# print(mv_l)
print(mv_l_np)

### Part 2 ###

t_l = Tensor(lst)
t_l_np = Tensor(l_np)

print(t_l is t_l_np)
print(t_l == t_l_np)

print(t_l_np.data, t_l.data)

res_add = t_l + t_l_np
print(res_add.data)
print(type(res_add))
# print(result.context)

add_ops, add_parent1, add_parent2 = res_add.context
print("ops:", add_ops, "parents:", add_parent1, add_parent2)

res_mul = t_l * res_add

print(res_mul.data)
print(type(res_mul))
# print(result.context)

mul_ops, mul_parent1, mul_parent2 = res_mul.context
print("ops:", mul_ops, "parents:", mul_parent1, mul_parent2)

# print(type(result.__add__))

### Part 3 ###

# test difference between IntEnum and Enum
# test auto from enum

# print(Tensor.__dict__)
# print(vars(Tensor))
# print(dir(Tensor))

# print(Operations.ADD)
# print(Operations.MUL)
# print(list(Operations))
# print(Operations['ADD'].value)
# print(Operations['MUL'].value)

### Part 4 ###

# add_enum_value, add_lambda = backward_operations[Operations.ADD - 1]
# add_lambda = backward_operations[add_ops]
# print(add_lambda)

# add_lambda_result = add_lambda((add_parent1.data, add_parent2.data), res_add)
# print("add lambd result:", add_lambda_result)

# add_one, add_two = add_lambda(res_add.data, add_parent1)
# print("one\n", add_one, "---end one")
# print("add_two\n", add_two, "---end add_two")

# print(backward_operations[Operations.MUL - 1])
# mul_lambda = backward_operations[mul_ops]
# print(mul_enum_value, mul_lambda)

# print(mul_parent1.data)
# print(type(mul_parent1.data))
# mul_lambda_result = mul_lambda([mul_parent1.data, mul_parent2.data], res_mul.data)
# one, two = mul_lambda(res_mul.data, [mul_parent1.data, mul_parent2.data])
# print("mul lambda result:", mul_lambda_result)
# print("one---\n", one, "---end one")
# print("two\n", two, "---end two")


### Part 5 ###
print("--- Part 5 ---\n")

# Create a simple function that takes has parameters a Tensor and with the function backward wright all the Operations that created it for example F > Sum > Relu > Add > Sum
# for the write of backward we might say this (ops) goes here and the ... rest goes here

bias = Tensor([1, 2, 3, 4])

mul = t_l * t_l_np
add = mul + bias
# print(add.data.shape)
z = add + bias
sum = z.SUM()
# sum = add.SUM()
# print(sum.data, sum.grad)
sum.backward()

print("--- Part 6 ---\n")

from dlf.optimizer import SGD
