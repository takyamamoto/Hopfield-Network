# Hopfield Network
Hopfield network (Amari-Hopfield network) implemented with Python. Two update rules are implemented: **Asynchronous** & **Synchronous**.

## Requirement
- Python >= 3.5
- numpy
- matplotlib
- skimage
- tqdm
- keras (to load MNIST dataset)

## Usage
Run `train.py` or `train_mnist.py`.

## Demo

### train.py
The following is the result of using **Synchronous** update.
```
Start to data preprocessing...
Start to train weights...
100%|██████████| 4/4 [00:06<00:00,  1.67s/it]
Start to predict...
100%|██████████| 4/4 [00:02<00:00,  1.80it/s]
Show prediction results...
```
<img src="https://github.com/takyamamoto/Hopfield-Network/blob/master/imgs/result.png" width=50%>

```
Show network weights matrix...
````
<img src="https://github.com/takyamamoto/Hopfield-Network/blob/master/imgs/weights.png" width=50%>

### train_mnist.py
The following is the result of using **Asynchronous** update.
```
Start to data preprocessing...
Start to train weights...
100%|██████████| 3/3 [00:00<00:00, 274.99it/s]
Start to predict...
100%|██████████| 3/3 [00:00<00:00, 32.52it/s]
Show prediction results...
```
<img src="https://github.com/takyamamoto/Hopfield-Network/blob/master/imgs/result_mnist.png" width=30%>

```
Show network weights matrix...
````
<img src="https://github.com/takyamamoto/Hopfield-Network/blob/master/imgs/weights_mnist.png" width=50%>

## Reference
- Amari, "Neural theory of association and concept-formation", SI. Biol. Cybernetics (1977) 26: 175. https://doi.org/10.1007/BF00365229
- J. J. Hopfield, "Neural networks and physical systems with emergent collective computational abilities", Proceedings of the National Academy of Sciences of the USA, vol. 79 no. 8 pp. 2554–2558, April 1982.
