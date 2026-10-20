import os

# Chỉ định rõ các thư mục và file gốc cần đóng gói
TARGET_DIRS = ["jaxpi", "porous_pinn", "cases", "scripts"]
ROOT_FILES = ["pyproject.toml", "README.md"]
ALLOWED_EXTENSIONS = (".py", ".toml", ".md")
OUTPUT_FILE = "core_codebase.txt"


def collect_core_codebase():
  with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
    # 1. Ghi cây thư mục của các phần cốt lõi
    out.write("=== CORE PROJECT STRUCTURE ===\n")
    for d in TARGET_DIRS:
      if os.path.exists(d):
        out.write(f"{d}/\n")
        for root, _, files in os.walk(d):
          if "__pycache__" in root:
            continue
          rel_root = os.path.relpath(root, d)
          indent = "    " if rel_root == "." else "        "
          for f in sorted(files):
            if f.endswith(ALLOWED_EXTENSIONS):
              out.write(f"{indent}{f}\n")

    out.write("\nRoot Files:\n")
    for rf in ROOT_FILES:
      if os.path.exists(rf):
        out.write(f"    {rf}\n")

    out.write("\n" + "=" * 80 + "\n")

    # 2. Ghi nội dung chi tiết các file trong TARGET_DIRS
    for d in TARGET_DIRS:
      if not os.path.exists(d):
        continue
      for root, _, files in sorted(os.walk(d)):
        if "__pycache__" in root:
          continue
        for file in sorted(files):
          if file.endswith(ALLOWED_EXTENSIONS):
            file_path = os.path.join(root, file)
            out.write(
                f"\n{'=' * 80}\nFILE: {file_path}\n{'=' * 80}\n"
            )
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
              out.write(f.read().strip() + "\n")

    # 3. Ghi nội dung các file cấu hình / chạy ở root
    for rf in ROOT_FILES:
      if os.path.exists(rf):
        out.write(f"\n{'=' * 80}\nFILE: {rf}\n{'=' * 80}\n")
        with open(rf, "r", encoding="utf-8", errors="ignore") as f:
          out.write(f.read().strip() + "\n")

  file_size_kb = os.path.getsize(OUTPUT_FILE) / 1024
  print(
      f"-> Đã gom xong codebase cốt lõi vào '{OUTPUT_FILE}' ({file_size_kb:.1f}"
      " KB)."
  )


if __name__ == "__main__": 
  collect_core_codebase()