Specify the data path (DRIVE_PATH) before running the code.

This project requires you to set the DRIVE_PATH variable in the code so the scripts can find the dataset directory (for example the folder that contains `data_real_final/`).

How to set DRIVE_PATH
- Edit the Python file where DRIVE_PATH is defined (search for the symbol `DRIVE_PATH`).
- Assign it the absolute path to your dataset folder. Examples:
	- Windows (PowerShell):
		DRIVE_PATH = r"D:\Workspace\Projects\BrainImg_NodeEmb\data_real_final"
	- Linux / macOS:
		DRIVE_PATH = "/home/username/Projects/BrainImg_NodeEmb/data_real_final"

Tips
- Use a raw string (prefix with r) on Windows to avoid backslash escaping, or use forward slashes.
- Verify the directory contains files like `abide1_id.npy`, `class_abide1.npy`, and `graph_abide1.npy`.
- If the code reads DRIVE_PATH from an environment variable or a config file, set that instead (check the code for logic that prefers env vars).

Quick checklist before running
- [ ] DRIVE_PATH points to the correct dataset folder
- [ ] Required files exist in that folder (see `data_real_final/`)
- [ ] Python dependencies are installed (see project requirements or `requirements.txt` if present)
