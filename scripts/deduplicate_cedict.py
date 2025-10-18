#!/usr/bin/env python3
"""
Interactive script to manually remove duplicates from cedict_processed.txt

Usage:
    python scripts/deduplicate_cedict.py --mode english  # Remove English duplicates
    python scripts/deduplicate_cedict.py --mode chinese  # Remove Chinese duplicates
"""

import json
import os
import sys
import signal
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

# File paths
PROJECT_ROOT = Path(__file__).parent.parent
DICT_FILE = PROJECT_ROOT / "dictionaries" / "cedict_processed.txt"
BACKUP_FILE = PROJECT_ROOT / "dictionaries" / "cedict_processed.txt.backup"


class DictionaryEntry:
    """Represents a single dictionary entry"""
    def __init__(self, original_line_num: int, chinese: str, english: str):
        self.original_line_num = original_line_num  # Line number from file
        self.chinese = chinese
        self.english = english

    def __repr__(self):
        return f"DictionaryEntry(line={self.original_line_num}, chinese='{self.chinese}', english='{self.english}')"


class DeduplicationTool:
    """Main tool for handling dictionary deduplication"""

    def __init__(self, mode: str):
        self.mode = mode  # 'english' or 'chinese'
        self.entries: List[DictionaryEntry] = []
        self.duplicates: Dict[str, List[DictionaryEntry]] = {}
        self.progress: Dict[str, int] = {}
        self.should_exit = False

        # Progress file depends on mode
        self.progress_file = PROJECT_ROOT / "scripts" / f"deduplication_progress_{mode}.json"

        # Setup signal handler for graceful exit
        signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, sig, frame):
        """Handle Ctrl+C gracefully"""
        print("\n\n🛑 Exit signal received. Saving progress...")
        self.save_progress()
        print("✓ Progress saved. You can resume later by running this script again.")
        sys.exit(0)

    def parse_dictionary(self) -> None:
        """Parse the dictionary file into DictionaryEntry objects"""
        print(f"📖 Reading dictionary from: {DICT_FILE}")

        if not DICT_FILE.exists():
            print(f"❌ Error: Dictionary file not found at {DICT_FILE}")
            sys.exit(1)

        self.entries = []
        with open(DICT_FILE, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                original_line = line
                line = line.strip()
                if not line:
                    continue

                # Parse format: "chinese\tenglish"
                if '\t' in line:
                    parts = line.split('\t', 1)
                    if len(parts) == 2:
                        chinese = parts[0]
                        english = parts[1]
                        entry = DictionaryEntry(line_num, chinese, english)
                        self.entries.append(entry)
                else:
                    # Skip lines without tab separator
                    print(f"⚠ Warning: Skipping line {line_num} (no tab separator): {original_line.strip()[:50]}")

        print(f"✓ Loaded {len(self.entries)} entries")

        if len(self.entries) == 0:
            print("❌ Error: No entries were parsed. Please check the file format.")
            print("Expected format: 'chinese\\tenglish'")
            sys.exit(1)

    def find_duplicates(self) -> None:
        """Find duplicates based on the selected mode"""
        duplicate_map = defaultdict(list)

        if self.mode == 'english':
            # Find English words with multiple Chinese translations
            for entry in self.entries:
                duplicate_map[entry.english].append(entry)
            mode_label = "English words"
        else:  # chinese mode
            # Find Chinese words with multiple English translations
            for entry in self.entries:
                duplicate_map[entry.chinese].append(entry)
            mode_label = "Chinese words"

        # Filter only duplicates
        self.duplicates = {
            key: entries
            for key, entries in duplicate_map.items()
            if len(entries) > 1
        }

        print(f"✓ Found {len(self.duplicates)} {mode_label} with duplicates")
        total_duplicate_entries = sum(len(entries) for entries in self.duplicates.values())
        print(f"  Total duplicate entries: {total_duplicate_entries}")

    def load_progress(self) -> None:
        """Load previously saved progress"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                self.progress = json.load(f)
            print(f"✓ Loaded progress: {len(self.progress)} duplicate sets already resolved")
        else:
            self.progress = {}
            print("ℹ No previous progress found. Starting fresh.")

    def save_progress(self) -> None:
        """Save current progress to JSON file"""
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(self.progress, f, ensure_ascii=False, indent=2)

    def prompt_choice(self, key: str, entries: List[DictionaryEntry]) -> int:
        """Prompt user to choose which duplicate to keep"""
        print("\n" + "="*70)

        if self.mode == 'english':
            # English mode: show English word and Chinese translations
            print(f"📝 English word: \"{key}\"")
            print(f"   Found {len(entries)} Chinese translations:\n")
            for idx, entry in enumerate(entries, 1):
                print(f"   [{idx}] {entry.chinese} (original line {entry.original_line_num})")
        else:  # chinese mode
            # Chinese mode: show Chinese word and English translations
            print(f"📝 Chinese word: \"{key}\"")
            print(f"   Found {len(entries)} English translations:\n")
            for idx, entry in enumerate(entries, 1):
                print(f"   [{idx}] {entry.english} (original line {entry.original_line_num})")

        print(f"\n   [0] Keep NONE of these\n")

        while True:
            try:
                choice = input(f"👉 Choose which to keep (0-{len(entries)}), or 'q' to quit: ").strip()

                if choice.lower() == 'q':
                    self.should_exit = True
                    return -1

                choice_num = int(choice)
                if 0 <= choice_num <= len(entries):
                    return choice_num
                else:
                    print(f"❌ Invalid choice. Please enter a number between 0 and {len(entries)}")
            except ValueError:
                print("❌ Invalid input. Please enter a number or 'q' to quit")
            except EOFError:
                # Handle EOF (Ctrl+D)
                print("\n")
                self.should_exit = True
                return -1

    def interactive_resolution(self) -> None:
        """Interactively resolve all duplicates"""
        total = len(self.duplicates)
        resolved = len(self.progress)

        print(f"\n{'='*70}")
        print(f"🎯 Starting interactive deduplication ({self.mode} mode)")
        print(f"   Progress: {resolved}/{total} duplicate sets resolved")
        print(f"   Press Ctrl+C at any time to save progress and exit")
        print(f"{'='*70}\n")

        for idx, (key, entries) in enumerate(sorted(self.duplicates.items()), 1):
            # Skip already resolved
            if key in self.progress:
                continue

            print(f"\n📊 Progress: {idx}/{total} duplicate sets")
            choice = self.prompt_choice(key, entries)

            if self.should_exit:
                break

            self.progress[key] = choice
            self.save_progress()
            print(f"✓ Choice saved")

        if not self.should_exit:
            print(f"\n{'='*70}")
            print("🎉 All duplicates have been resolved!")
            print(f"{'='*70}\n")

    def apply_changes(self) -> List[DictionaryEntry]:
        """Apply the deduplication choices to create a new dictionary"""
        print("\n📝 Applying changes to dictionary...")

        # Build set of entries to remove
        entries_to_remove = set()

        for key, choice in self.progress.items():
            if key not in self.duplicates:
                continue

            duplicate_entries = self.duplicates[key]

            if choice == 0:
                # Remove all duplicates
                for entry in duplicate_entries:
                    entries_to_remove.add(id(entry))
            elif 1 <= choice <= len(duplicate_entries):
                # Keep only the chosen one, remove the rest
                for idx, entry in enumerate(duplicate_entries, 1):
                    if idx != choice:
                        entries_to_remove.add(id(entry))

        # Filter entries
        kept_entries = [e for e in self.entries if id(e) not in entries_to_remove]

        removed_count = len(self.entries) - len(kept_entries)
        print(f"✓ Will remove {removed_count} duplicate entries")
        print(f"✓ Will keep {len(kept_entries)} entries")

        return kept_entries

    def write_dictionary(self, entries: List[DictionaryEntry]) -> None:
        """Write the deduplicated dictionary"""
        # Create backup
        if DICT_FILE.exists():
            import shutil
            shutil.copy(DICT_FILE, BACKUP_FILE)
            print(f"✓ Backup created at: {BACKUP_FILE}")

        # Write new file
        with open(DICT_FILE, 'w', encoding='utf-8') as f:
            for entry in entries:
                # Format: "chinese\tenglish"
                formatted_line = f"{entry.chinese}\t{entry.english}\n"
                f.write(formatted_line)

        print(f"✓ Dictionary updated: {DICT_FILE}")
        print(f"✓ Total entries: {len(entries)}")

    def cleanup_progress(self) -> None:
        """Remove the progress file after successful completion"""
        if self.progress_file.exists():
            self.progress_file.unlink()
            print("✓ Progress file cleaned up")

    def run(self) -> None:
        """Main execution flow"""
        print("\n" + "="*70)
        print(f"  📚 CEDICT Deduplication Tool ({self.mode.upper()} mode)")
        print("="*70 + "\n")

        # Parse and analyze
        self.parse_dictionary()
        self.find_duplicates()

        if not self.duplicates:
            mode_label = "English" if self.mode == 'english' else "Chinese"
            print(f"\n✨ No {mode_label} duplicates found! Dictionary is clean for this mode.")
            return

        self.load_progress()

        # Interactive resolution
        self.interactive_resolution()

        if self.should_exit:
            return

        # Confirm before applying changes
        print("\n" + "="*70)
        confirm = input("🔄 Apply changes to dictionary file? (yes/no): ").strip().lower()
        if confirm in ['yes', 'y']:
            kept_entries = self.apply_changes()
            self.write_dictionary(kept_entries)
            self.cleanup_progress()
            print("\n✅ Deduplication complete!")

            # Remind about the other mode
            other_mode = 'chinese' if self.mode == 'english' else 'english'
            print(f"\n💡 Tip: Don't forget to run with --mode {other_mode} to handle {other_mode} duplicates!")
        else:
            print("\n❌ Changes not applied. Progress is saved and you can run this script again.")


def main():
    """Entry point"""
    parser = argparse.ArgumentParser(
        description='Interactive tool to remove duplicates from cedict_processed.txt',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/deduplicate_cedict.py --mode english   # Remove English duplicates
  python scripts/deduplicate_cedict.py --mode chinese   # Remove Chinese duplicates

Two-pass workflow:
  1. First pass:  python scripts/deduplicate_cedict.py --mode english
  2. Second pass: python scripts/deduplicate_cedict.py --mode chinese
        """
    )

    parser.add_argument(
        '--mode',
        choices=['english', 'chinese'],
        required=True,
        help='Deduplication mode: "english" for same English word with different Chinese, '
             '"chinese" for same Chinese word with different English'
    )

    args = parser.parse_args()

    tool = DeduplicationTool(mode=args.mode)
    tool.run()


if __name__ == "__main__":
    main()
