# CNN_conversion

This is a package that will convert a dataset into an image and run a CNN model on the image

# Install
This can be installed with
```sh
%pip install git+https://github.com/as17simpson/CNN_conversion.git
```

# Quick Start
Create an object that will automatically convert your dataset into an image and run a CNN on it:

```python
from CNN_conversion import convert

CNN = convert(inputs, target)
```
