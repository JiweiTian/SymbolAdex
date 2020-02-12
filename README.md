**Scalable Inference of Symbolic Adversarial Examples**

To download and install the code for this project execute the following commands:
```
git clone https://github.com/dimy93/SymbolAdex.git
cd SymbolAdex
sudo ./install.sh 
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

To execute mortgage dataset experiments:
```
git clone https://github.com/dimy93/SymbolAdex.git
cd SymbolAdex
git checkout mortgage
sudo ./install.sh
```
