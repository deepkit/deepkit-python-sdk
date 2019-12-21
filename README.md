# Deepkit Python SDK

This package is necessary to send data from and to Deepkit in Python.

## Requirement

- Python 3.7+

Deepkit provides **Keras Callbacks** that handles many stuff automatically:

 - Keras is fully supported with version >2.2.5.
 - Tensorflow Keras is supported with version >2.
 
Debugger with layer watching is only available in Tensorflow Keras version >2.

Any other library is supported as well, but you need to manually
call the SDK methods ot enrich your experiment with analytical data. 

## Installation

```
$ pip install deepkit

# update
$ pip install deepkit --upgrade
```