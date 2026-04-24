from pathlib import Path
import subprocess, shlex

PARENTS = [
    "/home/sourccey/.cache/huggingface/lerobot/sourccey-013/sourccey-013__shirt-fold-blue-a/nickm",
    "/home/sourccey/.cache/huggingface/lerobot/sourccey-013/sourccey-013__shirt-fold-blue-c/nickm",
]

OUTPUT_REPO_ID = "Combination/sourccey-shirt-fold-c-000"
OUTPUT_ROOT = "/home/sourccey/.cache/huggingface/lerobot/Combination/sourccey-shirt-fold-c-000"

def is_dataset_root(p: Path) -> bool:
    return (p / "meta" / "info.json").is_file() and (p / "meta" / "tasks.parquet").is_file() and (p / "data").is_dir()

roots = []
for parent in PARENTS:
    pp = Path(parent)
    if not pp.exists():
        print(f"Missing parent: {pp}")
        continue
    # Allow either:
    # 1) a parent directory containing multiple dataset roots, or
    # 2) a direct path to a dataset root.
    if is_dataset_root(pp):
        roots.append(str(pp))
    else:
        roots.extend(sorted(str(d) for d in pp.iterdir() if d.is_dir() and is_dataset_root(d)))

if not roots:
    raise SystemExit("No dataset roots found.")

repo_ids = [Path(r).name for r in roots]

print(f"Found {len(roots)} dataset roots")
cmd = [
    "lerobot-edit-dataset",
    "--new_repo_id", OUTPUT_REPO_ID,
    "--new_root", OUTPUT_ROOT,
    "--operation.type", "merge",
    "--operation.repo_ids", str(repo_ids),
    "--operation.roots", str(roots),
]
print("Running:", " ".join(shlex.quote(x) for x in cmd[:8]), "...")

subprocess.run(cmd, check=True)
print(f"Merged dataset written to: {OUTPUT_ROOT}")
