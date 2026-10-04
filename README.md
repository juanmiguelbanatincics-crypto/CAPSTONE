# CDH Pharmaceutical Inventory DSS

## Setup

This project is a Streamlit and DuckDB pharmaceutical inventory Decision Support System.

1. Install WSL2 Ubuntu from an elevated Windows PowerShell prompt:

   ```powershell
   wsl --install
   ```

   Restart Windows if prompted, then open Ubuntu and complete the Linux-user setup.

2. Clone the repository inside the native WSL2 Linux filesystem. Do not clone or create the virtual environment under `/mnt/c/`:

   ```bash
   mkdir -p ~/projects
   cd ~/projects
   git clone https://github.com/juanmiguelbanatincics-crypto/CAPSTONE.git
   cd CAPSTONE
   ```

   Windows Smart App Control currently blocks DuckDB's native Windows binary through a Windows Code Integrity policy. Run this application inside WSL2 Ubuntu, not directly with Windows Python.

3. Create and activate a Linux virtual environment inside the WSL filesystem:

   ```bash
   python3 -m venv .venv-linux
   source .venv-linux/bin/activate
   pip install -r requirements.txt
   ```

## Database Setup

The DuckDB database files are **not included** in this repository. They are excluded by `.gitignore` because the available snapshots total approximately 663 MB.

At minimum, the application requires this file at this exact relative path:

```text
database/cdh_inventory.duckdb
```

This matches the fallback path in `app/data.py` (`DEFAULT_DATABASE_PATH`).

For a governed published database, an optional `database/current.json` pointer may be supplied. When present, the pointer must resolve to a valid versioned database under `database/versions/`, such as:

```text
database/versions/2026.1/cdh_inventory.duckdb
```

If `current.json` is not supplied, the application falls back to `database/cdh_inventory.duckdb`.

Shared database download:

```text
https://drive.google.com/drive/folders/1UWdUdctoPOkdlg6BBmt6O9inJjw8hEN3?usp=sharing
```

The shared drive folder also includes `DATABASE_CHECKSUM.txt` so the downloaded database can be verified for integrity.

After obtaining the shared database package, preserve the relative `database/` layout before starting the app.

## Running the app

From the repository root, activate `.venv-linux`, then run:

```bash
cd app
streamlit run main.py
```

Open the application at:

```text
http://localhost:8501
```

## Project structure

- `app/` - Streamlit user interface and read-only data-access layer.
- `src/` - Data-pipeline scripts that build and validate the analytical database.
- `docs/` - Data-handling rules and project documentation.
- `database/` - Local database files supplied separately and excluded from Git.
