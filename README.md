This is the repository for ''For Robust Worst-Group Accuracy, Ignore Domain Annotations'' published in [TMLR](https://openreview.net/forum?id=l8E68fD6yp). 

There are two types of scipts in this folder. The 'hparam' scripts run the hyperparameter tuning 10 times and selects 10 different hyperparameter values for each noise level for each dataset. We then set the hyperparameters by choosing the most selected value for each hyperparameter for each noise level for each dataset. The data-augmentation based LLR and vanilla LLR methods run the hyperparameters for all datsets.

For example to run hyperparameter tuning for RAD with L1 regularization on celebA:

```
python RAD_l1_hparams_celebA.py
```

Now to run the final testing for RAD with L1 regularization on celebA:

```
python final_RAD_l1_celebA.py
```

For our SELF implementation, the code expects the base models in the file structure. The code explains where exactly it looks for. The 'self' scripts do both the hyperparameter tuning and the final testing.
