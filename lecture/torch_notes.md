# Detach

- `t.detach()` return a tensor which is detached from the computation graph. However, this tensor is a reference to the original tensor `t`.
- just calling `detach()` won't destroy the computational graph.

```python
x = torch.tensor([1.,2.],requires_grad=True)
xfang = x * x
xlifang = x * xfang
xfang_detached = xfang.detach()
loss = xlifang.sum()
loss.backward()
print(x.grad) # Not None
```

# Clone

- If you want to mutate `t` after detaching it from the graph, you should use `t.detach().clone()`, so that the mutation won't affect `t` in the graph.

# Backward

- Can backward twice for one leaf tensor `x`, but can't backward for one non-leaf tensor `y` twice. For example, this is possible
```python
x = torch.tensor([1.,2.],requires_grad=True)
y = (x * x).sum()
z = (x * x).sum()
y.backward()
z.backward()
```
