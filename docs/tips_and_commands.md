# Tips and Useful Commands

## Clearing all the pycaches folders

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


