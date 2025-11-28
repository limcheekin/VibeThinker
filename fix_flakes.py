import re
import sys


def fix_flake8_errors(flake8_output):
    errors = flake8_output.strip().split("\n")
    file_errors = {}

    for error in errors:
        match = re.match(r"([^:]+):(\d+):(\d+): (\w+) (.+)", error)
        if match:
            filepath, lineno, col, code, message = match.groups()
            lineno = int(lineno)
            if filepath not in file_errors:
                file_errors[filepath] = []
            file_errors[filepath].append((lineno, code, message))

    for filepath, errors in file_errors.items():
        with open(filepath, "r") as f:
            lines = f.readlines()

        # Sort errors by line number in descending order to avoid messing up line numbers
        errors.sort(key=lambda x: x[0], reverse=True)

        for lineno, code, message in errors:
            line_index = lineno - 1
            if code == "F401":  # Unused import
                lines.pop(line_index)
            elif code == "F841":  # Unused local variable
                lines[line_index] = "#" + lines[line_index]
            elif code == "F541":  # f-string is missing placeholders
                lines[line_index] = (
                    lines[line_index].replace("f'", "'", 1).replace('f"', '"', 1)
                )

        with open(filepath, "w") as f:
            f.writelines(lines)


if __name__ == "__main__":
    flake8_output = sys.stdin.read()
    fix_flake8_errors(flake8_output)
