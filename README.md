# CMAC

### Using 'mimo.py'
For problems with multiple input variables, use 'mimo.py'.

Before using the CMAC, make sure to import your dataset, and to set all the parameters.
The variables which may be set at your discretion are weights, learning_rate, rho, g, and epoch. 
in_dim is fixed to the number of input dimensions of your dataset, but needs to be manually updated for different datasets.
D_train is somewhat discretionary, but must make sense within the context of your dataset.

FOR THE MIMO CMAC, in_dim MUST BE GREATER THAN ONE.
For a CMAC with one dimension of input, there are a variety of models available.

Equation numbers are usually references to Wu et al. (2011)
