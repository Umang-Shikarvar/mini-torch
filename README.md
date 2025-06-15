# mini-torch

Maintainers:
1. [Shardul Junagade](https://github.com/ShardulJunagade)
2. [Umang Shikarvar](https://github.com/Umang-Shikarvar)
3. [Soham Gaonkar](https://github.com/Soham-Gaonkar)




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