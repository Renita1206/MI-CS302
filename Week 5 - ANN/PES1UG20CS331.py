# Renita Kurian - PES1UG20CS331
# Week 5 - ANN Lab Assignment

import numpy as np

class Tensor:

    def __init__(self, arr, requires_grad=True):
        self.arr = arr
        self.requires_grad = requires_grad
        self.history = ['leaf', None, None]
        self.zero_grad()
        self.shape = self.arr.shape

    def zero_grad(self):
        self.grad = np.zeros_like(self.arr)

    def set_history(self, op, operand1, operand2):
        self.history = []
        self.history.append(op)
        self.requires_grad = False
        self.history.append(operand1)
        self.history.append(operand2)

        if operand1.requires_grad or operand2.requires_grad:
            self.requires_grad = True

    def __add__(self, other):
        if isinstance(other, self.__class__):
            if self.shape != other.shape:
                raise ArithmeticError(
                    f"Shape mismatch for +: '{self.shape}' and '{other.shape}' ")
            out = self.arr + other.arr
            out_tensor = Tensor(out)
            out_tensor.set_history('add', self, other)

        else:
            raise TypeError(
                f"unsupported operand type(s) for +: '{self.__class__}' and '{type(other)}'")

        return out_tensor

    def __matmul__(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError(
                f"unsupported operand type(s) for matmul: '{self.__class__}' and '{type(other)}'")
        if self.shape[-1] != other.shape[-2]:
            raise ArithmeticError(
                f"Shape mismatch for matmul: '{self.shape}' and '{other.shape}' ")
        out = self.arr @ other.arr
        out_tensor = Tensor(out)
        out_tensor.set_history('matmul', self, other)
        return out_tensor

    def grad_add(self, gradients=None):
        if gradients is None:
            left_grad = np.ones_like(self.arr)
            right_grad = np.ones_like(self.arr)
        else:
            left_grad = gradients
            right_grad = gradients

        return left_grad, right_grad

    def grad_matmul(self, gradients=None):
        if gradients is None:
            multiplier = np.ones_like(self.arr)
        else:
            multiplier = gradients

        left_grad = multiplier @ self.history[2].arr.T
        right_grad = self.history[1].arr.T @ multiplier
        return left_grad, right_grad

    def backward(self, gradients=None):
        left_g = right_g = None
        if self.history[0] == 'leaf':
            return

        if self.history[0] == 'add':
            left_g, right_g = self.grad_add(gradients)
            if self.history[1].requires_grad:
                self.history[1].grad += left_g
            if self.history[2].requires_grad:
                self.history[2].grad += right_g

        elif self.history[0] == 'matmul':
            left_g, right_g = self.grad_matmul(gradients)
            if self.history[1].requires_grad:
                self.history[1].grad += left_g
            if self.history[2].requires_grad:
                self.history[2].grad += right_g

        self.history[1].backward(left_g)
        self.history[2].backward(right_g)


