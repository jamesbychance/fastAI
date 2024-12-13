# Managing FastAI Course Notes on GitHub

## Working with the repository and managing updates

# Syncing Your Local FastAI Repository with GitHub

## Daily Start-up Routine

1. Navigate to your repository:
```bash
cd ~/courses/fastAI
```

2. Check you're on your notes branch:
```bash
git checkout my-course-notes
```

3. Pull the latest changes from GitHub:
```bash
git pull origin my-course-notes
```

## Useful Status Commands

Check your current branch:
```bash
git branch
```

Check if any files are modified or out of sync:
```bash
git status
```

## Common Issues and Solutions

If you get errors about uncommitted changes:
1. Save your current work:
```bash
git stash
```

2. Pull the updates:
```bash
git pull origin my-course-notes
```

3. Reapply your saved work:
```bash
git stash pop
```

# Pushing Local Changes to GitHub

## Basic Push Workflow

1. Check which files you've modified:
```bash
git status
```

2. Stage your changes (add files to be committed):
```bash
# Add specific file:
git add path/to/your/file

# Or add all changed files:
git add .
```

3. Commit your changes with a descriptive message:
```bash
git commit -m "Brief description of what you changed"
```

4. Push to your GitHub branch:
```bash
git push origin my-course-notes
```

## Common Scenarios

### If Push Is Rejected
This usually means changes exist on GitHub that you don't have locally. Solve by:

1. Pull the latest changes first:
```bash
git pull origin my-course-notes
```

2. Then push your changes:
```bash
git push origin my-course-notes
```

### Verifying Your Push

After pushing, you can verify your changes were pushed by:

1. Checking your repository on GitHub
2. Or using git log to see your latest commit:
```bash
git log --oneline -1
```

# Previous version of help for github

## Repository Setup and Best Practices

### Initial Setup
1. Fork the fastai/fastbook repository to your GitHub account. This creates your own copy while maintaining a link to the original source.

2. Clone your forked repository locally:
```bash
git clone https://github.com/YOUR_USERNAME/fastbook.git
```

3. Create a dedicated branch for your notes:
```bash
git checkout -b my-course-notes
```

4. Create a notes directory:
   - Make a new `my_notes/` directory in the repository
   - This keeps your notes separate from course content
   - Organize by chapter or topic as needed

5. When making changes:
```bash
git add my_notes/
git commit -m "Add notes for chapter X"
git push origin my-course-notes
```

### Benefits of This Structure
- Easy to pull updates from original course repository
- Notes are version controlled
- Clear separation between original content and personal additions
- Simple to share notes with others if desired

## Maintaining Your Notes

### Initial Setup Check
1. Ensure you're on your notes branch:
```bash
git checkout my-course-notes
```

### Regular Update Process

#### Before Starting New Notes
1. Sync with the original fastai repository:
```bash
# Add the original repo as upstream (if not done before)
git remote add upstream https://github.com/fastai/fastbook.git

# Fetch updates from upstream
git fetch upstream

# Merge updates from the main fastai repo
git checkout main
git merge upstream/main
git checkout my-course-notes
git merge main
```

#### Adding New Notes
1. Confirm you're on your notes branch:
```bash
git checkout my-course-notes
```

2. Update notes in the `my_notes/` directory:
   - One markdown file per lecture: `lecture_XX_notes.md`
   - Include date and lecture number at top
   - Add code snippets, insights, and questions
   - Link to relevant fastai book chapters

3. Commit your changes:
```bash
# Check modified files
git status

# Add new or modified notes
git add my_notes/lecture_XX_notes.md

# Commit with descriptive message
git commit -m "Add notes for Lecture XX: [Main Topic]"

# Push to GitHub
git push origin my-course-notes
```

### Best Practices

#### Note Structure
- Begin each note file with metadata:
  ```markdown
  # Lecture XX: [Title]
  Date: YYYY-MM-DD
  Related Book Chapter: [Chapter Number]
  ```
- Use proper markdown code blocks for code snippets
- Maintain consistent section headers:
  - Notes
  - Code Examples
  - Questions
  - Resources
- Tag important concepts with `#tags`
- Reference original fastai notebook locations

#### Troubleshooting
If you encounter merge conflicts:
```bash
# Check conflict files
git status

# Resolve conflicts in editor
# Add resolved files
git add <resolved-files>

# Complete merge
git commit -m "Resolve merge conflicts with upstream"
```

### Useful Git Commands
- Check current branch: `git branch`
- View commit history: `git log --oneline`
- Discard local changes: `git checkout -- <file>`
- View changes before commit: `git diff`

## Additional Tips
- Bookmark this workflow document for easy reference
- Consider creating automation scripts for updates
- Use a consistent checklist for each new notes file
- Regularly push changes to GitHub for backup
- Keep commit messages clear and descriptive