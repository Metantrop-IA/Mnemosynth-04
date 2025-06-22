import tomli
import importlib.metadata
import re
import sys

def parse_package_name(dep):
    # Remove version specifiers and extras
    return re.split(r'[<>=!~\[]', dep.strip())[0]

def main():
    output_lines = []
    output_lines.append(f"Python version: {sys.version.split()[0]}\n")
    with open("pyproject.toml", "rb") as f:
        data = tomli.load(f)
    deps = data["project"]["dependencies"]
    output_lines.append("Dependency versions in current environment:\n")
    for dep in deps:
        pkg = parse_package_name(dep)
        try:
            version = importlib.metadata.version(pkg)
            output_lines.append(f"{pkg}: {version}")
        except importlib.metadata.PackageNotFoundError:
            output_lines.append(f"{pkg}: NOT INSTALLED")
    # Print to console
    print('\n'.join(output_lines))
    # Write to file
    with open("installed_versions.txt", "w") as outf:
        outf.write('\n'.join(output_lines) + '\n')

if __name__ == "__main__":
    main()