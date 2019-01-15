# Deep Summarization Toolkit

Toolkit for simple model training and evaluating.

### Description

`workbench` folder is used to store different model runs.
Use `*.yml` file to configure model launch. Don't forget to describe default config arguments. This toolkit helps avoid duplicate code of automaticaly log training process and evaluating different runs. Describe model, dataset and running config file is all you need.
Run model using next command:
```
$ python main.py my-run-file.yml
```