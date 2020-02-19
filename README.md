Scalable Inference of Symbolic Adversarial Examples
======
CIFAR10 and MNIST Experiments
------

To download and install the code for CIFAR10 and MNIST experiments execute the following commands:
```
git clone https://github.com/dimy93/SymbolAdex.git
cd SymbolAdex
./install.sh 
```
To run the MNIST convSmall experiments call the ***run\_mnist.sh*** script as follows:
```
cd ERAN/tf_verify/
./run_mnist.sh
```
To run the CIFAR10 convSmall experiments call the ***run\_cifar10.sh*** script as follows:
```
cd ERAN/tf_verify/
./run_cifar10.sh
```
To run the MNIST convBig experiments call the ***run\_mnist\_big.sh*** script as follows:
```
cd ERAN/tf_verify/
./run_mnist_big.sh
```
To run the MNIST 9x200 experiments call the ***run\_mnist\_ffn.sh*** script as follows:
```
cd ERAN/tf_verify/
./run_mnist_ffn.sh
```
The resulting symbolic adversarial examples will appear under ***./ERAN/tf\_verify/NetworkName\_ImgNum\_class\_AdvClass\_it\_Iteration***.

Mortgage Dataset Experiments
------

To download and install the code for the mortgage dataset experiments execute the following commands:
```
git clone https://github.com/dimy93/SymbolAdex.git
cd SymbolAdex
git checkout mortgage
sudo ./install.sh
```
To run the mortgage dataset experiments that use the triangle convex relaxation call the ***mortgage\_lp.sh*** script as follows:
```
cd ERAN/tf_verify/
./mortgage_lp.sh
```
To run the mortgage dataset experiments that use the DeepPoly convex relaxation call the ***mortgage\_deeppoly.sh*** script as follows:
```
cd ERAN/tf_verify/
./mortgage_deeppoly.sh
```
Visualisation of the resuls will appear in ***./ERAN/tf\_verify/mortgage\_ImgNum\_class\_0\_seed\_Seed***.

The resulting symbolic adversarial examples will appear under ***./ERAN/tf\_verify/mortgage\_ImgNum\_class\_0\_seed\_Seed\_it\_Iteration***.
