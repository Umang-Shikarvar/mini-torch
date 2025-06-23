# Usage Guide for miniTorch

## How to use minitorch in subfolders?
In python scripts, you can import `minitorch` in subfolders by adding the parent directory to the path.
```python
import sys
sys.path.append('../')
```
In Jupyter Notebooks, you can use the following code to add the parent directory to the path:
```python
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
```

## How to run tests?
You can run the tests using `pytest`. Make sure you have `pytest` installed in your environment. You can install it using pip:
```bash
pip install pytest
```
Then, you can run the tests using the following command:
```bash
pytest
# or
run_tests
```
