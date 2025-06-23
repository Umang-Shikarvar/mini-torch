# Contributing to miniTorch

## 🚀 Maintainer Contribution Workflow (No PRs)
1. Create a new branch for your changes:
   ```bash
   git checkout -b <branch-name>
   ```
2. Make your changes and commit them:
   ```bash
    git add .
    git commit -m "Add your commit message here"
    ```
3. Push your changes to your branch:
   ```bash
   git push origin <branch-name>
   ```
4. Merge into the main branch:
   ```bash
   git checkout main
   git pull origin main         # Ensure you're up to date
   git merge <branch-name>
   ```
   > 🔧 Resolve any merge conflicts if necessary.
5. Push the changes to the remote repository:
   ```bash
    git push origin main
    ```
6. Update your branch with the latest changes from the main branch:
   ```bash
   git checkout <branch-name>
   git pull origin main
   ```
