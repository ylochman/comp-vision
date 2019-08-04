# Deep Learning for CV. CIFAR-10

- For CIFAR-10 classification problem I trained a CNN with GeForce GTX 1080.
- I slightly modified `cifar10.ipynb` to get insights.
- I put all essentials in `net.py`, `utils.py`, `train.py`.
- **The experimental pipeline is in [`Experiments.ipynb`](./Experiments.ipynb)**
- The top-3-performance models are following:

![Top-3 Configs](assets/top-3_configs.png?raw=true "Top-3 Configs")

![Top-3 TB](assets/top-3_tb.png?raw=true "Top-3 TB")

- The best model gained 76.88% test accuracy. It was trained with Stochastic Gradient Descent (batch size 64, learning rate 0.016, with Nesterov momentum; 5-step learning rate decay by 0.1; cross-entropy objective and regularization on weights with 0.01 multiplier) for 10 epochs, then continued up to 15 epochs and stopped on 13th epoch due to Plateau. The architecture is:
    * 16 conv3x3 - relu - bn - maxpool2x2 ->
    * 32 conv3x3 - relu - bn - maxpool2x2 ->
    * 32 conv3x3 - relu - bn ->
    * 128 fc - relu ->
    * 64 fc - relu ->
    * 10 fc
- For the details see [`Experiments.ipynb`](./Experiments.ipynb)
- I also compared the training time with CPU / GPU. See below the results:
1. **GPU (2 min)**
```bash
time python train.py
```
```
real    0m46.655s
user    1m51.989s
sys     0m9.353s
```

2. **CPU (22 min)**
```bash
time python train.py --no-cuda
```
```
real    3m12.462s
user    22m2.023s
sys     0m14.506s
```
The configurations fot this experiment were:

![GPU/CPU Configs](assets/gpu_cpu_configs.png?raw=true "GPU/CPU Configs")

## Task Description

Зробити серiю експериментiв на базi ноутбука [a3_cifar10.ipynb](https://github.com/lyubonko/ucu2019/blob/master/assignments/a3_cifar10.ipynb)

**Вимоги**:

* Dataset: cifar10 
* Можливi експерименти:

    - Better model, follow guidelines at the end of the notebook
    - Experimenting with different optimization algorithms
    - Data augmentation
    - Experiment with pre-trained model

* Побажання щодо оформлення результатiв:
    - Опишiть що було зроблено i який остаточний результат
    - Зобразiть результати навчання у виглядi графiкiв. Пояснiть, що зображено на графiку та додайте опис
    - Також напишiть, що б можна було ще зробити та як покращити результати в майбутньому
    - Цiкаво побачити частинки вашого коду (код снiпет)
        * якщо Ви змiнювали аугментацiю - цiкаво побачити ваш transformers
        * якщо Ви використовували рiзний лернiнг рейт для рiзних шарiв - цiкаво побачити ваш optimizer
        * якщо Ви писали власну функцiю для змiни лернiнг рейт або використовували scheduler - цiкаво побачити, як ви це робили