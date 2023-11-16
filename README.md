# Cavity_FDM_NumPy2

Updated version of [Cavity_FDM_NumPy](https://github.com/ShotaDeguchi/Cavity_FDM_NumPy)

For a mathematical understanding of finite difference methods, a great resource can be found [here](https://folk.ntnu.no/leifh/teaching/tkt4140/._main000.html). 

See `01_Arakawa_B/04_Kawamura_Kuwahara/main.py` for the clean implementation. 

## Results
Cavity flow is a steady state problem. Consider that the field has reached its steady state when the following condition is satisfied:
```math
\max \left( \frac{\| u^{(n+1)} - u^{(n)} \|_2}{\| u^{(n)} \|_2}, \frac{\| v^{(n+1)} - v^{(n)} \|_2}{\| v^{(n)} \|_2} \right) \le \delta
```
where $\delta$ is the convergence tolerance, set to $\delta = 10^{-8}$. 

The following summarizes the results obtained using different spatial schemes at different Reynolds numbers. 


