# :test_tube: Running Experiments

## Usage

To run the experiments for CIFAR-100 with the FedAvg algorithm, use the following command:

```bash
python example_cifar_10.py --model_name=<model_name> \
                           --dataset_name=CIFAR100 \
                           --add_reg=0 \
                           --unbalanced_sgm=0 \
                           --rule=<partition_rule> \
                           --rule_arg=<dirichlet_value> \
                           --alg_name=FedAvg \
                           --mu_mean=0.0 \
                           --mu_var=0.0 \
                           --epoch=5 \
                           --lr_decay_per_round=0.998 \
                           --learning_rate=0.1 \
                           --com_amount=500
```
For SAM/ASAM experiments, add ```--opt_method=SAM/ASAM``` to the command with FedAvg as the algorithm.

### Parameters:
- **model_name**: cifar100 or ConvNet
- **dataset_name**: CIFAR100 or TinyImageNet
- **add_reg**: Flag for adding MAN regularizer
- **unbalanced_sgm**: Sigma for unbalanced data simulation (0 for none).
- **rule**: The rule for data partitioning (iid or Dirichlet)
- **rule_arg**: The argument for the partition rule, typically the concentration parameter for the Dirichlet distribution for non-iid setups
- **alg_name**: FedAvg/FedDyn/FedDC/FedSpeed
- **mu_mean**: hyperparameter $\zeta$ (we have used 0.15 for CIFAR100 and 0.1 for  Tiny-Imagenet)
- **mu_var**: 0 (not used in our experiments)
- **epoch**: number of epochs per round
- **lr_decay_per_round**: learning rate decay per round
- **learning_rate**: initial learning rate
- **com_amount**: number of communication rounds

### Note:

- Before running experiments for Tiny-Imagenet, dataset has to be installed in the **code** or **man_code_sam** folder depending on the experiments that need to be run. Use the following command to download the dataset:
```bash 
gdown 13xbXPCi1LAXuZRNIy6ArkJp79qpwQiIi
```
- All SAM/ASAM experiments should be run in the **man_code_sam** directory while the other experiments will be run in the **code** directory.

### Data Partitioning
- Experiments were conducted with both **iid** and **non-iid** data setups.
- **Non-iid Data** was generated using the Dirichlet distribution \(Dir(\delta)\).
  - A label distribution vector was sampled for each client.
  - Lower \(\delta\) values result in higher label imbalances.

### Experimental Setup
- **Number of Clients**: 100 clients participated in the experiments.
- **Client Selection**: 10% of clients were selected at random for each communication round.
- **Communication Rounds**: Accuracy was measured at the end of 500 communication rounds.
- **Sensitivity Analysis**: Conducted for hyper-parameter ($\zeta$) on global model accuracy.