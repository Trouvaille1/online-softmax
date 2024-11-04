# online-softmax
simplest online-softmax notebook for explain Flash Attention

## Softmax Series

### softmax 

$$
\tilde{x}_i =\frac{e^{x_i}}{\sum_j^Ne^{x_j}}
$$

### safe softmax

$$
\tilde{x}_i =\frac{e^{x_i-\max(x_{:N})}}{\sum_j^Ne^{x_j-\max(x_{:N})}}
$$

### online softmax

1. we could calculate  `1:N`  element $\max(x_{:N})$ and $l_N$

2. when we add one element  $x_{N+1}$, we should update $\max(x_{:N+1})$ and update $l_{N+1}$ as follow. 

$$
\begin{align}
l_{N} &= \sum_j^N e^{x_j-\max(x_{:N})}\\
\max(x_{:N+1})&=\max( \max(x_{:N}), x_{N+1} )\\
l_{N+1} &= \sum_j^{N+1} e^{x_j-\max(x_{:N+1})} \\
&= (\sum_j^N e^{x_j-\max(x_{:N})}) +e^{x_{N+1}-\max(x_{:N+1})} \\
&=(\sum_j^N e^{x_j-\max(x_{:N})}e^{\max(x_{:N})-\max(x_{:N+1})})+e^{x_{N+1}-\max(x_{:N+1})} \\
&=(\sum_j^N e^{x_j-\max(x_{:N})})(e^{\max(x_{:N})-\max(x_{:N+1})}) +e^{x_{N+1}-\max(x_{:N+1})} \\
&=l_N (e^{\max(x_{:N})-\max(x_{:N+1})})+e^{x_{N+1}-\max(x_{:N+1})} \\
\end{align}
$$

​	why not use $l_{N+1}=l_{N}+x_{N+1}$, because safe softmax need all element sub a same max value.

3. we could softmax with updated numerator and denominator

$$
\tilde{x}_{i} =\frac{e^{x_i-\max(x_{:N+1})}}{l_{N+1}}
$$

### block online softmax

online softmax make denominator sum $l$ dynamic update while a new element added. It's more effiecent method is to update sum $l$ with block-wise element added. This advantage is we could parallelism to calculate online softmax

1. we cloud seperate calculate different block $l^{(t)}$  and $m^{(t)}$

$$
\begin{align}
l^{(1)} &= l_{N} = \sum_j^N e^{x_j-\max(x_{:N})}\\
m^{(1)} &= \max(x_{:N}) \\
l^{(2)} &= l_{N:2N} = \sum_{j=N+1}^{2N} e^{x_j-\max(x_{{N+1}:2N})}\\
m^{(2)} &= \max(x_{N+1:2N}) \\
\end{align}
$$

2. it’s easy to update global $m$ and $l$ 
   $$
   \begin{align}
   m=\max({x_{:2N}}) &= \max(\max({x_{:N}}), \max(x_{N+1:2N})) \\
   &=max(m^{(1)}, m^{(2)})
   
   \end{align}
   $$
   but the $l$  NOT update  as follow when we use safe-softmax
   $$
   l=l_{:2N} \neq l^{(1)}+l^{(2)}
   $$

3. So  we cloud based block sum $l^{(t)}$ and max $m^{(t)}$  to **online** update global $l$

$$
\begin{align}
l^{(1)}&= \sum_j^N e^{x_j-\max(x_{:N})} = \sum_j^N e^{x_j-m^{(1)}}\\
l^{(2)} &= \sum_{j=N+1}^{2N} e^{x_j-\max(x_{{N+1}:2N})} = \sum_{j=N+1}^{2N} e^{x_j-m^{(2)}}\\

l &= \sum_{j}^{2N} e^{x_j-\max(x_{:2N})} \\

&= (\sum_j^N e^{x_j-\max(x_{:2N})}) +(\sum_{j=N+1}^{2N}e^{x_j-\max(x_{:2N})}) \\


&= (\sum_j^N e^{x_j-m}) +(\sum_{j=N+1}^{2N}e^{x_j-m}) \\

&= (\sum_j^N e^{x_j-m^{(1)}}) (e^{m^{(1)}-m}) +(\sum_{j=N+1}^{2N}e^{x_j-m^{(2)}})(e^{m^{(2)}-m}) \\

&= l^{(1)} (e^{m^{(1)}-m}) +l^{(2)}(e^{m^{(2)}-m})
\end{align}
$$

4. we cloud online update block softmax like:

$$
\tilde{x}_{i} =\frac{e^{x_i-m}}{l}
$$

### batch online softmax

In attention machine, we need softmax for attention score
$$
S = QK^T, S \in \mathbb{R}^{N \times N}
$$
the query is row-wise matrix $Q\in \mathbb{R}^{N \times D}$;

and we need softmax attention score:
$$
P_{i,:} = \text{softmax}(S_{i,:})
$$
when we use online-softmax, we could parallel update k-row max $M^{(t)}$ and row-wise sum $L^{(t)}$, 
$$
L = L^{(1)} (e^{M^{(1)}-M}) +L^{(2)}(e^{M^{(2)}-M})
$$
where $L,M \in \mathbb{R}^{k \times 1}$

## Implemention

run `online_softmax_torch.ipynb`

we show the block online softmax result

```python
X = torch.tensor([-0.3, 0.2, 0.5, 0.7, 0.1, 0.8])
X_softmax = F.softmax(X, dim = 0)
print(X_softmax)

X_block = torch.split(X, split_size_or_sections = 3 , dim = 0) 

# we parallel calculate  different block max & sum
X_block_0_max = X_block[0].max()
X_block_0_sum = torch.exp(X_block[0] - X_block_0_max).sum()

X_block_1_max = X_block[1].max()
X_block_1_sum = torch.exp(X_block[1] - X_block_1_max).sum()

# online block update max & sum
X_block_1_max_update = torch.max(X_block_0_max, X_block_1_max) # X[-1] is new data
X_block_1_sum_update = X_block_0_sum * torch.exp(X_block_0_max - X_block_1_max_update) \
                     + torch.exp(X_block[1] - X_block_1_max_update).sum() # block sum

X_block_online_softmax = torch.exp(X - X_block_1_max_update) / X_block_1_sum_update
print(X_block_online_softmax)
```

output is 

```
tensor([0.0827, 0.1364, 0.1841, 0.2249, 0.1234, 0.2485])
tensor([0.0827, 0.1364, 0.1841, 0.2249, 0.1234, 0.2485])
```

## Reference

[手撕Flash Attention](https://zhuanlan.zhihu.com/p/663932651)

[Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867)

