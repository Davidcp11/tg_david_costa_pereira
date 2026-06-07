# How to Run the Script
This script is part of a modular training pipeline and requires a configuration file to be passed as a parameter.

## Basic Usage
To run the script, use the following command:

``` bash
python -m training.main --config <config_fpath>
```
Where `config_fpath` is the path to your configuration file (.json) that defines the training parameters.

## Parameter Explanation
- `--config`: **Required**. The path to the configuration file that contains model training settings (e.g., hyperparameters, dataset paths, etc.).

### Example Command

```bash 
python -m training.main --config ./configs/exp1.json
```

In this example:

- The script will start training using the parameters defined in `exp1.json`.

## Accessing Help
To see a detailed explanation of all available arguments, you can run:


```bash
python -m training.main --help
```
This will display usage information, including all possible arguments and their descriptions.