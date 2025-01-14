# CS726 - HW3

## Comparison of Optimization Methods for an Unconstrained Quadratic Minimization

### Usage

#### Q3

Run the following in the current directory in a MatLab shell:

```MatLab
optimization(n, m, iter)
```

where the dimension, `n = 100`, `iter = 1000` and `m = 1, .1 or .01`.

For more information, type the following:

```MatLab
help optimization
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
