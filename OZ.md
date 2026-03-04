# OZ Environment Setup Notes

## Who is Oz?

Oz is an AI agent built into Warp (the Agentic Development Environment).
Oz helps you set up and manage Warp environments — automated cloud workspaces
that clone your repositories, install toolchains, and run setup commands
consistently every time they are triggered.

---

## What was done in this session

### 1. Folder & Repository Created
- Local folder created: `D:\claude\OZ`
- README.md added
- Git initialized and initial commit made

### 2. Git Identity Configured
```
git config --global user.email "krz.pietras@gmail.com"
git config --global user.name "pietras"
```

### 3. GitHub CLI Installed
- Installed via: `winget install --id GitHub.cli`
- Version installed: 2.87.3

### 4. GitHub Authentication
- Authenticated via browser login (`gh auth login`)
- Logged in as: **krzpietras-glitch**
- Protocol: HTTPS

### 5. GitHub Repository Created & Pushed
- Repository: https://github.com/krzpietras-glitch/OZ
- Pushed initial commit with README.md

---

## What still needs to be done

To finish creating the Warp environment, the following steps remain:

### Step 1 — Choose a Docker image
Tell Oz what language/framework you plan to use in this repo
(e.g. Python, Node.js, Go, Rust, .NET).
Oz will recommend an appropriate base image.

### Step 2 — Confirm setup commands
Oz will propose the commands that run automatically after the repo is cloned
(e.g. `npm install`, `pip install -r requirements.txt`).

### Step 3 — Create the Warp environment
Once the above are confirmed, Oz will run:
```
oz environment create --name OZ --docker-image <image> --repo krzpietras-glitch/OZ --setup-command "<command>"
```

---

## How to resume after a terminal restart

1. Open Warp terminal
2. Make sure `gh` and `oz` CLIs are available (they should be after install)
3. Verify GitHub auth is still active:
   ```
   gh auth status
   ```
4. Continue the conversation with Oz in Warp and tell it:
   > "I want to continue setting up my Warp environment for krzpietras-glitch/OZ"

---

## Quick Reference

| Item             | Value                                      |
|------------------|--------------------------------------------|
| Local path       | D:\claude\OZ                               |
| GitHub repo      | https://github.com/krzpietras-glitch/OZ    |
| GitHub user      | krzpietras-glitch                          |
| Git email        | krz.pietras@gmail.com                      |
| Git username     | pietras                                    |
