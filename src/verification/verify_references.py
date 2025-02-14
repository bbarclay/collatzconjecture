#!/usr/bin/env python3
"""
Cross-reference and citation verification script for LaTeX files.
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict


class ReferenceVerifier:
    """Verifies cross-references and citations in LaTeX files"""

    def __init__(self, latex_dir: Path):
        self.latex_dir = latex_dir
        self.labels: Dict[str, str] = {}  # label -> file
        self.citations: Dict[str, str] = {}  # citation key -> file
        self.references: Dict[str, List[str]] = defaultdict(list)  # label -> [files]
        self.cite_references: Dict[str, List[str]] = defaultdict(
            list
        )  # citation -> [files]
        self.errors: List[str] = []

    def scan_files(self):
        """Scan all LaTeX files for labels and references"""
        for tex_file in self.latex_dir.glob("**/*.tex"):
            self._scan_file(tex_file)

    def _scan_file(self, file_path: Path):
        """Scan a single LaTeX file"""
        content = file_path.read_text()
        rel_path = file_path.relative_to(self.latex_dir)

        # Find labels
        for match in re.finditer(r"\\label{([^}]+)}", content):
            label = match.group(1)
            if label in self.labels:
                self.errors.append(
                    f"Duplicate label '{label}' in {rel_path} "
                    f"(already defined in {self.labels[label]})"
                )
            self.labels[label] = str(rel_path)

        # Find references
        for match in re.finditer(r"\\ref{([^}]+)}", content):
            label = match.group(1)
            self.references[label].append(str(rel_path))

        # Find citations
        for match in re.finditer(r"\\cite{([^}]+)}", content):
            citations = match.group(1).split(",")
            for cite in citations:
                cite = cite.strip()
                self.cite_references[cite].append(str(rel_path))

    def verify_references(self) -> bool:
        """Verify all references are valid"""
        valid = True

        # Check for undefined labels
        for label, files in self.references.items():
            if label not in self.labels:
                self.errors.append(
                    f"Undefined label '{label}' referenced in: {', '.join(files)}"
                )
                valid = False

        # Check for unused labels
        for label, file in self.labels.items():
            if label not in self.references:
                self.errors.append(f"Unused label '{label}' in {file}")

        return valid

    def verify_citations(self, bib_file: Path) -> bool:
        """Verify all citations exist in bibliography"""
        valid = True
        citations = self._parse_bibliography(bib_file)

        # Check for undefined citations
        for cite, files in self.cite_references.items():
            if cite not in citations:
                self.errors.append(
                    f"Undefined citation '{cite}' used in: {', '.join(files)}"
                )
                valid = False

        # Check for unused citations
        for cite in citations:
            if cite not in self.cite_references:
                self.errors.append(f"Unused citation '{cite}' in bibliography")

        return valid

    def _parse_bibliography(self, bib_file: Path) -> Set[str]:
        """Parse bibliography file for citation keys"""
        citations = set()
        content = bib_file.read_text()

        # Match @type{key, ... } entries
        for match in re.finditer(r"@\w+{([^,]+),", content):
            citations.add(match.group(1))

        return citations

    def verify_theorem_references(self) -> bool:
        """Verify theorem references are consistent"""
        valid = True
        theorem_labels = {label for label in self.labels if label.startswith("thm:")}

        # Check theorem mapping consistency
        theorem_file = self.latex_dir / "../src/theorem_mapper.py"
        if theorem_file.exists():
            content = theorem_file.read_text()
            for label in theorem_labels:
                if f'"{label}"' not in content:
                    self.errors.append(
                        f"Theorem '{label}' not mapped in theorem_mapper.py"
                    )
                    valid = False

        return valid

    def generate_report(self) -> str:
        """Generate verification report"""
        report = [
            "# Cross-Reference Verification Report",
            "",
            "## Statistics",
            f"- Total labels: {len(self.labels)}",
            f"- Total references: {len(self.references)}",
            f"- Total citations: {len(self.cite_references)}",
            "",
        ]

        if self.errors:
            report.extend(
                ["## Errors", "", *[f"- {error}" for error in self.errors], ""]
            )
        else:
            report.extend(
                ["## Status", "", "All references verified successfully.", ""]
            )

        return "\n".join(report)


def main():
    """Main entry point"""
    if len(sys.argv) != 2:
        print("Usage: verify_references.py <latex_dir>")
        sys.exit(1)

    latex_dir = Path(sys.argv[1])
    if not latex_dir.is_dir():
        print(f"LaTeX directory not found: {latex_dir}")
        sys.exit(1)

    verifier = ReferenceVerifier(latex_dir)
    verifier.scan_files()

    # Verify references
    refs_valid = verifier.verify_references()

    # Verify citations if bibliography exists
    bib_file = latex_dir / "references.bib"
    cites_valid = True
    if bib_file.exists():
        cites_valid = verifier.verify_citations(bib_file)

    # Verify theorem references
    theorems_valid = verifier.verify_theorem_references()

    # Generate and print report
    report = verifier.generate_report()
    print(report)

    # Save report
    report_file = latex_dir / "reference_check.md"
    report_file.write_text(report)
    print(f"\nReport saved to: {report_file}")

    # Exit with appropriate status
    sys.exit(not (refs_valid and cites_valid and theorems_valid))


if __name__ == "__main__":
    main()
