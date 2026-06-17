from pathlib import Path

import mkdocs_gen_files

src = Path("src")
summary_lines = []

for path in sorted(src.rglob("*.py")):
    module_path = path.relative_to(src).with_suffix("")
    doc_path = path.relative_to(src).with_suffix(".md")
    parts = tuple(module_path.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = Path("api", *parts, "index.md")
    elif parts[-1].startswith("_"):
        continue
    else:
        doc_path = Path("api", *parts).with_suffix(".md")

    ident = ".".join(parts)
    with mkdocs_gen_files.open(doc_path, "w") as doc_file:
        doc_file.write(f"# `{ident}`\n\n::: {ident}\n")
    mkdocs_gen_files.set_edit_path(doc_path, path)

    indent = "    " * (len(parts) - 1)
    summary_lines.append(f"{indent}* [{ident}]({doc_path.relative_to('api')})\n")

# writes the nav file that literate-nav reads to build the API sidebar
with mkdocs_gen_files.open("api/SUMMARY.md", "w") as summary_file:
    summary_file.write("".join(summary_lines))

# summary_lines has one entry per generated .md file (skipping _private modules)
print(f"Generated {len(summary_lines)} API pages in api/")