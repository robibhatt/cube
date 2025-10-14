# ReLU Network Scaling Counterexample

This note provides an explicit counterexample showing that multiplying **all**
parameters (weights and biases) of a ReLU multi-layer perceptron by a constant
factor `λ` does **not** simply rescale the original network output by `λ^L`,
where `L` is the number of linear layers. Once biases are non-zero, the ReLU
activation pattern can change after rescaling, so the network implements a
different function entirely.

## Network definition

Consider a 3-layer scalar ReLU network evaluated on the input `x = 1`:

\[
\begin{aligned}
h_1 &= \operatorname{ReLU}(w_1 x + b_1), \\
 h_2 &= \operatorname{ReLU}(w_2 h_1 + b_2), \\
 f(x) &= w_3 h_2 + b_3.
\end{aligned}
\]

Choose the parameters

* `w1 = 1`, `b1 = -0.5`
* `w2 = 1`, `b2 = 0.5`
* `w3 = 1`, `b3 = 0.5`

Evaluating the original network gives

\[
\begin{aligned}
h_1 &= \operatorname{ReLU}(1 \cdot 1 - 0.5) = 0.5, \\
 h_2 &= \operatorname{ReLU}(1 \cdot 0.5 + 0.5) = 1.0, \\
 f(1) &= 1 \cdot 1.0 + 0.5 = 1.5.
\end{aligned}
\]

## Scaling every parameter by `λ = 2`

Now multiply **every** parameter by two:

* `w1' = 2`, `b1' = -1`
* `w2' = 2`, `b2' = 1`
* `w3' = 2`, `b3' = 1`

Re-evaluating the network yields

\[
\begin{aligned}
h_1' &= \operatorname{ReLU}(2 \cdot 1 - 1) = 1.0, \\
 h_2' &= \operatorname{ReLU}(2 \cdot 1.0 + 1) = 3.0, \\
 f'(1) &= 2 \cdot 3.0 + 1 = 7.0.
\end{aligned}
\]

If uniform scaling preserved the original computation we would expect the new
output to equal `λ^3 f(1) = 2^3 * 1.5 = 12`, but the actual value is 7. The
change comes from the bias shifts altering the intermediate ReLU activations:
`h_1` increased from 0.5 to 1.0, so the downstream units see larger pre-activations
than a simple `λ` factor, and the final bias contributes an additional shift.

## Python confirmation

```python
import torch
import torch.nn as nn

class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(1, 1)
        self.l2 = nn.Linear(1, 1)
        self.l3 = nn.Linear(1, 1)

    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        return self.l3(x)

net = TinyNet()
with torch.no_grad():
    net.l1.weight.fill_(1.0); net.l1.bias.fill_(-0.5)
    net.l2.weight.fill_(1.0); net.l2.bias.fill_(0.5)
    net.l3.weight.fill_(1.0); net.l3.bias.fill_(0.5)

x = torch.tensor([[1.0]])
orig = net(x).item()

lam = 2.0
with torch.no_grad():
    for p in net.parameters():
        p.mul_(lam)
scaled = net(x).item()

print(f"original output: {orig}")
print(f"scaled output:   {scaled}")
print(f"λ^3 * original:   {lam ** 3 * orig}")
```

Running this script prints

```
original output: 1.5
scaled output:   7.0
λ^3 * original:   12.0
```

confirming that the naive `λ^3` scaling law does not hold when biases are
present.

```
