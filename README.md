# miniTorch

**Maintainers:**
1. [Shardul Junagade](https://github.com/ShardulJunagade)
2. [Umang Shikarvar](https://github.com/Umang-Shikarvar)
3. [Soham Gaonkar](https://github.com/Soham-Gaonkar)


## Usage

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

### How to run tests?
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


## 🚀 Maintainer Contribution Workflow (No PRs)
1. Create a new branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes and commit them:
   ```bash
    git add .
    git commit -m "Add your commit message here"
    ```
3. Push your changes to your branch:
   ```bash
   git push origin feature/your-feature-name
   ```
4. Merge into the main branch:
   ```bash
   git checkout main
   git pull origin main         # Ensure you're up to date
   git merge feature/your-feature-name
   ```
   > 🔧 Resolve any merge conflicts if necessary.
5. Push the changes to the remote repository:
   ```bash
    git push origin main
    ```
6. Update your branch with the latest changes from the main branch:
   ```bash
   git checkout feature/your-feature-name
   git pull origin main
   ```
