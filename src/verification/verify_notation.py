#!/usr/bin/env python3
"""
Mathematical notation consistency verification script for LaTeX files.
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict


class NotationVerifier:
    """Verifies mathematical notation consistency in LaTeX files"""

    def __init__(self, latex_dir: Path):
        self.latex_dir = latex_dir
        self.math_commands: Dict[str, List[Tuple[str, str]]] = defaultdict(
            list
        )  # command -> [(file, context)]
        self.math_environments: Dict[str, List[Tuple[str, str]]] = defaultdict(
            list
        )  # env -> [(file, content)]
        self.symbols: Dict[str, List[Tuple[str, str]]] = defaultdict(
            list
        )  # symbol -> [(file, context)]
        self.definitions: Dict[str, Tuple[str, str]] = (
            {}
        )  # symbol -> (file, definition)
        self.errors: List[str] = []

    def scan_files(self):
        """Scan all LaTeX files for mathematical notation"""
        for tex_file in self.latex_dir.glob("**/*.tex"):
            self._scan_file(tex_file)

    def _scan_file(self, file_path: Path):
        """Scan a single LaTeX file for mathematical notation"""
        content = file_path.read_text()
        rel_path = file_path.relative_to(self.latex_dir)

        # Find math commands
        for match in re.finditer(r"\\newcommand{\\(\w+)}", content):
            cmd = match.group(1)
            context = self._get_context(content, match.start())
            self.math_commands[cmd].append((str(rel_path), context))

        # Find math environments
        for match in re.finditer(
            r"\\begin{(equation|align|gather|multline)\*?}(.*?)\\end{\1\*?}",
            content,
            re.DOTALL,
        ):
            env = match.group(1)
            env_content = match.group(2)
            self.math_environments[env].append((str(rel_path), env_content))

        # Find mathematical symbols and their definitions
        for match in re.finditer(r"\\def\\(\w+)|\\newcommand{\\(\w+)}", content):
            symbol = match.group(1) or match.group(2)
            context = self._get_context(content, match.start())
            if symbol not in self.definitions:
                self.definitions[symbol] = (str(rel_path), context)
            else:
                self.errors.append(
                    f"Symbol '\\{symbol}' redefined in {rel_path} "
                    f"(originally defined in {self.definitions[symbol][0]})"
                )

        # Find symbol usage
        for match in re.finditer(r"\\(\w+)", content):
            symbol = match.group(1)
            if symbol not in {
                "begin",
                "end",
                "newcommand",
                "def",
                "documentclass",
                "usepackage",
            }:
                context = self._get_context(content, match.start())
                self.symbols[symbol].append((str(rel_path), context))

    def _get_context(self, content: str, pos: int, context_size: int = 50) -> str:
        """Get context around a position in text"""
        start = max(0, pos - context_size)
        end = min(len(content), pos + context_size)
        return content[start:end].replace("\n", " ").strip()

    def verify_consistency(self) -> bool:
        """Verify notation consistency"""
        valid = True

        # Check for undefined symbols
        for symbol, usages in self.symbols.items():
            if symbol not in self.definitions and len(usages) > 1:
                files = {file for file, _ in usages}
                if len(files) > 1:  # Only report if used in multiple files
                    self.errors.append(
                        f"Symbol '\\{symbol}' used without definition in: {', '.join(files)}"
                    )
                    valid = False

        # Check for inconsistent usage
        for env, usages in self.math_environments.items():
            symbols = set()
            for _, content in usages:
                for match in re.finditer(r"\\(\w+)", content):
                    symbols.add(match.group(1))

            for symbol in symbols:
                if symbol in self.definitions:
                    files_used = {file for file, _ in self.symbols[symbol]}
                    def_file = self.definitions[symbol][0]
                    if def_file not in files_used:
                        self.errors.append(
                            f"Symbol '\\{symbol}' defined in {def_file} but used differently in: {', '.join(files_used)}"
                        )
                        valid = False

        return valid

    def verify_theorem_notation(self) -> bool:
        """Verify notation consistency in theorems"""
        valid = True
        theorem_envs = {"theorem", "lemma", "proposition", "corollary"}

        for env in theorem_envs:
            for file, content in self.math_environments.get(env, []):
                # Check for consistent notation within theorems
                symbols = set()
                for match in re.finditer(r"\\(\w+)", content):
                    symbol = match.group(1)
                    if symbol in self.definitions:
                        symbols.add(symbol)

                # Verify symbols are defined before use
                for symbol in symbols:
                    def_file, _ = self.definitions[symbol]
                    if (
                        def_file > file
                    ):  # Simple check if definition appears after usage
                        self.errors.append(
                            f"Symbol '\\{symbol}' used in theorem ({file}) before definition in {def_file}"
                        )
                        valid = False

        return valid

    def generate_report(self) -> str:
        """Generate verification report"""
        report = [
            "# Mathematical Notation Verification Report",
            "",
            "## Statistics",
            f"- Total math commands: {sum(len(usages) for usages in self.math_commands.values())}",
            f"- Total math environments: {sum(len(usages) for usages in self.math_environments.values())}",
            f"- Total symbols defined: {len(self.definitions)}",
            f"- Total symbol usages: {sum(len(usages) for usages in self.symbols.values())}",
            "",
        ]

        if self.errors:
            report.extend(
                ["## Inconsistencies", "", *[f"- {error}" for error in self.errors], ""]
            )
        else:
            report.extend(
                ["## Status", "", "All mathematical notation is consistent.", ""]
            )

        # Add symbol usage summary
        report.extend(
            [
                "## Symbol Usage Summary",
                "",
            ]
        )
        for symbol, usages in sorted(self.symbols.items()):
            if len(usages) > 1:  # Only show symbols used multiple times
                files = {file for file, _ in usages}
                if symbol in self.definitions:
                    def_file = self.definitions[symbol][0]
                    report.append(
                        f"- \\{symbol}: Defined in {def_file}, used in {len(files)} files"
                    )
                else:
                    report.append(
                        f"- \\{symbol}: Used in {len(files)} files (no explicit definition)"
                    )

        return "\n".join(report)


def main():
    """Main entry point"""
    if len(sys.argv) != 2:
        print("Usage: verify_notation.py <latex_dir>")
        sys.exit(1)

    latex_dir = Path(sys.argv[1])
    if not latex_dir.is_dir():
        print(f"LaTeX directory not found: {latex_dir}")
        sys.exit(1)

    verifier = NotationVerifier(latex_dir)
    verifier.scan_files()

    # Verify notation consistency
    notation_valid = verifier.verify_consistency()

    # Verify theorem notation
    theorems_valid = verifier.verify_theorem_notation()

    # Generate and print report
    report = verifier.generate_report()
    print(report)

    # Save report
    report_file = latex_dir / "notation_check.md"
    report_file.write_text(report)
    print(f"\nReport saved to: {report_file}")

    # Exit with appropriate status
    sys.exit(not (notation_valid and theorems_valid))


if __name__ == "__main__":
    main()
