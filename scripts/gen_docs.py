# scripts/gen_ref_pages.py
from pathlib import Path
import mkdocs_gen_files

src = Path("src/tab2seq/")
nav = mkdocs_gen_files.Nav()

print("Generating reference pages...")
for path in sorted(src.rglob("*.py")):
    print(path)  # visible in mkdocs serve terminal output

for path in sorted(src.rglob("*.py")):
    module_path = path.relative_to(src).with_suffix("")
    doc_path = path.relative_to(src).with_suffix(".md")
    full_doc_path = Path("api", doc_path)

    parts = tuple(module_path.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    elif parts[-1].startswith("_"):
        continue

    nav[parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        ident = ".".join(parts)
        fd.write(f"# `{ident}`\n\n::: {ident}\n")

    mkdocs_gen_files.set_edit_path(full_doc_path, path)
    
for path in sorted(src.rglob("*.py")):
    print(path)  # visible in mkdocs serve terminal output

with mkdocs_gen_files.open("api/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())