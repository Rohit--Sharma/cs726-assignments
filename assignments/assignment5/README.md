# CS726 - HW5

## Constrained Minimization

### Q1. Importance of Choosing Appropriate Geometry

#### Usage

Run the following in the current directory in a MatLab shell:

```MatLab
q1(<n>, <n_iter>)
```

where the dimension, `n = 500` and `n_iter = 2000`.

For more information, type the following:

```MatLab
help q1
```

### Q2. Comparing Frank-Wolfe and PGD

#### Usage

Run the following in the current directory in a MatLab shell:

```MatLab
q2(<p>, <n>, <n_iter>)
```

where the dimension, `p = 10`, `n = 200` and `n_iter = 300`.

For more information, type the following:

```MatLab
help q2
```

## Q3. First order Methods with Noisy Gradient Oracle

### Usage

Run the following in the current directory in a MatLab shell:

```MatLab
q3(<n>, <n_iter>, <epsilon>)
```

where the dimension, `n = 200`, `n_iter = 300` and `epsilon = 0.1, 0.001` or `0.00001`.

For more information, type the following:

```MatLab
help q3
```

The script generates 2 plots for each run. For the value of `m` chosen, figure 1 corresponds to Nesterov's vs Nesterov's strongly convex vs Conjugate Gradient Method vs Heavy Ball method, and figure 2 corresponds to monotonically decreasing variant of Nesterov's vs Nesterov's strongly convex vs Conjugate Gradient Method vs Heavy Ball method.

##### Results

`m=1`
<p float="left">
	<img src="nest_vs_all_m=1.png" width="350" />
	<img src="mononest_vs_all_m=1.png" width="350" />
</p>

`m=0.1`
<p float="left">
	<img src="nest_vs_all_m=.1.png" width="350" />
	<img src="mononest_vs_all_m=.1.png" width="350" />
</p>

`m=0.01`
<p float="left">
	<img src="nest_vs_all_m=.01.png" width="350" />
	<img src="mononest_vs_all_m=.01.png" width="350" />
</p>

#### Q4

Run the following in the current directory in a MatLab shell:

```MatLab
q4(iter)
```

where `iter = 100`.

For more information, type the following:

```MatLab
help q4
```

The script generates a plot showing the optimality gap against number of iterations for Nesterov's method for strongly convex `f` and Heavy ball method for `f` defined in Q3.

##### Results

![Nesterov vs Heavy Ball](nest_vs_heavyball.png?raw=true "Nesterov vs Heavy Ball")
