# mini-torch

**Maintainers:**
1. [Shardul Junagade](https://github.com/ShardulJunagade)
2. [Umang Shikarvar](https://github.com/Umang-Shikarvar)
3. [Soham Gaonkar](https://github.com/Soham-Gaonkar)



### How to use minitorch in subfolders?
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



### Clearing all the pycaches folders


Using Bash:
```bash
find . -type d -name "__pycache__" -exec rm -r {} +
```

Using PowerShell:
```powershell
Get-ChildItem -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force
```

Using Command Prompt:
```cmd
for /d /r . %d in (__pycache__) do @if exist "%d" rd /s /q "%d"
```