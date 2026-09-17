# Development shell for code-atlas: the venv, plus this repository's .env in the environment.
#
# Dot-source it from the repo (or a worktree of it):   . scripts/activate.ps1
#
# Atlas itself never reads .env -- the environment is the caller's to provide. This script is
# that caller for local development: it puts the gitignored .env of the MAIN checkout into this
# PowerShell session, so `atlas ...` and `pytest` started here see its ATLAS_* overrides and
# provider API keys. Variables already set in the session win, as they would for any launcher.
#
# A linked worktree (.claude/worktrees/*) has no .env or .venv of its own; both come from the
# main checkout, found through the worktree's `.git` file rather than by calling git.

$checkout = Split-Path -Parent $PSScriptRoot
$main = $checkout
$gitFile = Join-Path $checkout '.git'
if (Test-Path $gitFile -PathType Leaf) {
    # "gitdir: <main>/.git/worktrees/<name>"
    $gitDir = ((Get-Content $gitFile -TotalCount 1) -replace '^gitdir:\s*', '').Trim()
    $main = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $gitDir))
}

$venv = @($checkout, $main) | ForEach-Object { Join-Path $_ '.venv/Scripts/Activate.ps1' } |
    Where-Object { Test-Path $_ } | Select-Object -First 1
if ($venv) { . $venv } else { Write-Warning "No .venv found in $checkout or $main -- run: uv sync --group dev" }

$envFile = Join-Path $main '.env'
if (Test-Path $envFile) {
    $loaded = 0
    foreach ($line in Get-Content $envFile) {
        if ($line -notmatch '^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$') { continue }
        $name, $value = $Matches[1], $Matches[2].Trim()
        if ($value -match '^"(.*)"$' -or $value -match "^'(.*)'$") {
            $value = $Matches[1]
        } else {
            $value = ($value -replace '\s+#.*$', '').Trim()  # an unquoted value may carry a trailing comment
        }
        if ($value -eq '' -or (Test-Path "env:$name")) { continue }
        Set-Item -Path "env:$name" -Value $value
        $loaded++
    }
    Write-Host "code-atlas: loaded $loaded variable(s) from $envFile"
}
