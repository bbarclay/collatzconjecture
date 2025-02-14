#!/usr/bin/env python3
"""
Final review script that runs all verification checks and generates a comprehensive report.
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, List, Tuple
import shutil


class FinalReview:
    """Runs all verification checks and generates a comprehensive report"""

    def __init__(self, workspace_dir: Path):
        self.workspace_dir = workspace_dir
        self.latex_dir = workspace_dir / "latexpaper"
        self.results_dir = workspace_dir / "review_results"
        self.results_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def run_verification_scripts(self) -> Dict[str, bool]:
        """Run all verification scripts"""
        results = {}
        scripts = [
            ("Reference Check", "verify_references.py"),
            ("Notation Check", "verify_notation.py"),
            ("Proof Structure Check", "verify_proof_structure.py"),
        ]

        for name, script in scripts:
            print(f"\nRunning {name}...")
            try:
                result = subprocess.run(
                    [sys.executable, script, str(self.latex_dir)],
                    capture_output=True,
                    text=True,
                )
                results[name] = result.returncode == 0

                # Save individual reports
                report_file = self.latex_dir / f"{script.replace('.py', '')}_check.md"
                if report_file.exists():
                    shutil.copy(
                        report_file,
                        self.results_dir
                        / f"{self.timestamp}_{script.replace('.py', '')}_report.md",
                    )

                # Save output
                with open(
                    self.results_dir
                    / f"{self.timestamp}_{script.replace('.py', '')}_output.txt",
                    "w",
                ) as f:
                    f.write(result.stdout)
                    if result.stderr:
                        f.write("\n\nErrors:\n")
                        f.write(result.stderr)

                if result.returncode != 0:
                    self.errors.append(
                        f"{name} failed with exit code {result.returncode}"
                    )
                    if result.stderr:
                        self.errors.append(result.stderr)
            except Exception as e:
                results[name] = False
                self.errors.append(f"Error running {name}: {str(e)}")

        return results

    def run_code_verification(self) -> bool:
        """Run code verification tests"""
        print("\nRunning Code Verification...")
        try:
            result = subprocess.run(
                [sys.executable, "run_tests.py"], capture_output=True, text=True
            )

            # Save test results
            test_results = self.workspace_dir / "test_results"
            if test_results.exists() and test_results.is_dir():
                for result_file in test_results.glob("*.json"):
                    shutil.copy(
                        result_file,
                        self.results_dir / f"{self.timestamp}_test_{result_file.name}",
                    )

            if result.returncode != 0:
                self.errors.append("Code verification tests failed")
                if result.stderr:
                    self.errors.append(result.stderr)
                return False
            return True
        except Exception as e:
            self.errors.append(f"Error running code verification: {str(e)}")
            return False

    def verify_implementation_mapping(self) -> bool:
        """Verify implementation mapping between LaTeX and code"""
        print("\nVerifying Implementation Mapping...")
        valid = True

        try:
            # Check theorem mapper
            theorem_file = self.workspace_dir / "src/theorem_mapper.py"
            if not theorem_file.exists():
                self.errors.append("Theorem mapper file not found")
                return False

            # Extract theorem labels from LaTeX
            latex_theorems = set()
            for tex_file in self.latex_dir.glob("**/*.tex"):
                content = tex_file.read_text()
                import re

                for match in re.finditer(
                    r"\\begin{theorem}.*?\\label{([^}]+)}", content, re.DOTALL
                ):
                    latex_theorems.add(match.group(1))

            # Check if all theorems are mapped
            mapper_content = theorem_file.read_text()
            for theorem in latex_theorems:
                if f'"{theorem}"' not in mapper_content:
                    self.warnings.append(
                        f"Theorem '{theorem}' not found in theorem mapper"
                    )
                    valid = False

            return valid
        except Exception as e:
            self.errors.append(f"Error verifying implementation mapping: {str(e)}")
            return False

    def check_documentation_completeness(self) -> bool:
        """Check documentation completeness"""
        print("\nChecking Documentation Completeness...")
        required_files = [
            "README.md",
            "latexpaper/main.tex",
            "src/theorem_mapper.py",
            "tests/test_collatz.py",
            "verify_proof.py",
            "generate_report.py",
        ]

        missing_files = []
        for file in required_files:
            if not (self.workspace_dir / file).exists():
                missing_files.append(file)

        if missing_files:
            self.errors.append("Missing required files: " + ", ".join(missing_files))
            return False
        return True

    def generate_report(self, verification_results: Dict[str, bool]) -> str:
        """Generate comprehensive review report"""
        report = [
            "# Final Review Report",
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Verification Results",
            "",
        ]

        # Add verification results
        for name, passed in verification_results.items():
            status = "✅ Passed" if passed else "❌ Failed"
            report.append(f"### {name}: {status}")
            report.append("")

        # Add code verification results
        code_status = "✅ Passed" if self.run_code_verification() else "❌ Failed"
        report.append(f"### Code Verification: {code_status}")
        report.append("")

        # Add implementation mapping results
        mapping_status = (
            "✅ Passed" if self.verify_implementation_mapping() else "❌ Failed"
        )
        report.append(f"### Implementation Mapping: {mapping_status}")
        report.append("")

        # Add documentation status
        docs_status = (
            "✅ Passed" if self.check_documentation_completeness() else "❌ Failed"
        )
        report.append(f"### Documentation Completeness: {docs_status}")
        report.append("")

        # Add errors and warnings
        if self.errors:
            report.extend(
                ["## Errors", "", *[f"- {error}" for error in self.errors], ""]
            )

        if self.warnings:
            report.extend(
                ["## Warnings", "", *[f"- {warning}" for warning in self.warnings], ""]
            )

        # Add summary
        total_checks = len(verification_results) + 3  # +3 for code, mapping, and docs
        passed_checks = sum(1 for result in verification_results.values() if result)
        passed_checks += (
            (1 if code_status == "✅ Passed" else 0)
            + (1 if mapping_status == "✅ Passed" else 0)
            + (1 if docs_status == "✅ Passed" else 0)
        )

        report.extend(
            [
                "## Summary",
                "",
                f"- Total checks: {total_checks}",
                f"- Passed: {passed_checks}",
                f"- Failed: {total_checks - passed_checks}",
                f"- Success rate: {(passed_checks / total_checks) * 100:.1f}%",
                "",
                "## Recommendation",
                "",
            ]
        )

        if passed_checks == total_checks:
            report.append("✅ All checks passed. The proof is ready for submission.")
        else:
            report.append(
                "❌ Some checks failed. Please address the issues before submission."
            )

        return "\n".join(report)


def main():
    """Main entry point"""
    if len(sys.argv) != 2:
        print("Usage: final_review.py <workspace_dir>")
        sys.exit(1)

    workspace_dir = Path(sys.argv[1])
    if not workspace_dir.is_dir():
        print(f"Workspace directory not found: {workspace_dir}")
        sys.exit(1)

    reviewer = FinalReview(workspace_dir)

    # Run verification scripts
    verification_results = reviewer.run_verification_scripts()

    # Generate and save report
    report = reviewer.generate_report(verification_results)
    report_file = reviewer.results_dir / f"{reviewer.timestamp}_final_review.md"
    report_file.write_text(report)

    print("\nFinal Review Report:")
    print(report)
    print(f"\nReport saved to: {report_file}")

    # Exit with appropriate status
    all_passed = (
        all(verification_results.values())
        and not reviewer.errors
        and not reviewer.warnings
    )
    sys.exit(not all_passed)


if __name__ == "__main__":
    main()
