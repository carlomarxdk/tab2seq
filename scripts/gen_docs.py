from pathlib import Path

src = Path("src")
docs_api = Path("docs/api")
docs_api.mkdir(parents=True, exist_ok=True)
summary_lines = []

for path in sorted(src.rglob("*.py")):
    module_path = path.relative_to(src).with_suffix("")
    doc_path = path.relative_to(src).with_suffix(".md")
    full_doc_path = docs_api / doc_path

    parts = tuple(module_path.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
        full_doc_path = full_doc_path.with_name("index.md")
    elif parts[-1].startswith("_"):
        continue

    full_doc_path.parent.mkdir(parents=True, exist_ok=True)
    ident = ".".join(parts)
    full_doc_path.write_text(f"# `{ident}`\n\n::: {ident}\n")

    indent = "    " * (len(parts) - 1)
    summary_lines.append(f"{indent}* [{ident}]({full_doc_path.relative_to(docs_api)})\n")

# writes the nav file that literate-nav reads to build the API sidebar
(docs_api / "SUMMARY.md").write_text("".join(summary_lines))

# summary_lines has one entry per generated .md file (skipping _private modules)
print(f"Generated {len(summary_lines)} API pages in {docs_api}")