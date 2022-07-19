# Algebraic Bethe Circuits
[![DOI](https://zenodo.org/badge/455221135.svg)](https://zenodo.org/badge/latestdoi/number)

This repository contains the code to reproduce the numerical implementations presented in the manuscript ["Algebraic Bethe Circuits"](https://arxiv.org/abs/2202.04673).


## Dependences

- `Pyhton>=3.8`

- `qibo==0.1.7`

- `qibojit==0.04`

## Usage
[`Pk_gates.py`](https://github.com/AlejandroSopena/Algebraic_Bethe_Circuits/blob/main/Pk_gates.py) contains the functions to generate the matrices <img src="https://render.githubusercontent.com/render/math?math=P_k"> (unitaries for <img src="https://render.githubusercontent.com/render/math?math=k<M"> and isometries <img src="https://render.githubusercontent.com/render/math?math=P_k|0\rangle"> for <img src="https://render.githubusercontent.com/render/math?math=k\geq M">).
```python
from Pk_gates import full_pink

nspins=4
roots=[-0.574,0.574]
delta=0.5
  
[Pk,Gk]=full_pink(nspins,roots,delta)
```

[`bethe_circuit.py`](https://github.com/AlejandroSopena/Algebraic-Bethe-Circuits/blob/main/bethe_circuit.py) defines the class `BetheCircuit` which implements the Algebraic Bethe Ansatz for the XXZ model with both the non-unitary matrices <img src="https://render.githubusercontent.com/render/math?math=R"> and the unitary matrices <img src="https://render.githubusercontent.com/render/math?math=P_k">.
```python
nspins = 4
nmagnons = 2
roots = [-0.574,0.574]
delta = 0.5

v = BetheCircuit(nspins, nmagnons, delta)
state1 = v.exact(roots)().numpy()
state1 = [state1[i] for i in range(0, len(state1), 2 ** nmagnons)]
state1 /= np.linalg.norm(state1)
state2 = v.aba(roots)().numpy()
print('overlap: ', np.abs(np.dot(np.conjugate(state1), state2)))
print('norm exact', norm(state1))
print('norm circuit', norm(state2))
