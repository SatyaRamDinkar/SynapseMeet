# Git auto-commit and push script for voice-driven prompt engineering
param(
    [string]$CommitMessage = "voice-prompt: automated progress sync"
)

Write-Host "Syncing repository..." -ForegroundColor Cyan

# Check if git is initialized
if (-not (Test-Path ".git")) {
    Write-Host "Initializing git repository..." -ForegroundColor Yellow
    git init
    git branch -M main
}

# Add all changes
git add .

# Check if there are changes to commit
$status = git status --porcelain
if ($status) {
    git commit -m "$CommitMessage"
    Write-Host "Committed changes: $CommitMessage" -ForegroundColor Green
} else {
    Write-Host "No changes to commit." -ForegroundColor Gray
}

# Check if remote exists
$remote = git remote
if ($remote) {
    Write-Host "Pushing to remote origin..." -ForegroundColor Yellow
    git push -u origin main
} else {
    Write-Host "Notice: Remote 'origin' not configured yet. Run 'git remote add origin <your-repo-url>' to push to GitHub." -ForegroundColor Yellow
}
