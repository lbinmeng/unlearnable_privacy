# Unlearnable privacy
This project aims to make privacy information (e.g., identity) unlearnable for machine learning models.

## Privacy-unlearnable noises
We generate privacy-unlearnable EEG dataset by sample-wise and subject-wise noise:
```
# x_train: raw EEG training data, y_train: task labels, s_train: identity labels
# generated by sample-wise noise 
from unlearnable_gen import unlearnable
u_x_train = unlearnable(x_train, y_train, s_train, args)

# generated by subject-wise noise
from unlearnable_gen import unlearnable_optim
u_x_train = unlearnable_optim(x_train, y_train, s_train, args)
``` 

## Make data unlearnable for all subjects
```
# generate privacy-unlearnable dataset for ERN dataset with lambda 0.5, EEGNet as the feature extractor
python3  main.py --dataset ERN --feature_c EEGNet --alpha 0.5 
```


## Online scenario
```
python3  main_continue.py --dataset MI109 --alpha 0.03 
```
