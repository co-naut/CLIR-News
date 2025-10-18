# CLIR-News

## Project Proposal

[project proposal](./SI650_Project_Proposal_G9.pdf)

## Setup

### Prerequisites

- Python 3.x
- pip (Python package manager)

### Download Pre-trained Word2Vec Embeddings

run:

```bash
chmod +x ./get_resources.sh

./get_resources.sh
```

and wait for the script to finish.

## Dictionary Deduplication (Help Needed!)

We have **2,538 English duplicates** and **35 Chinese duplicates** in our dictionary that need manual review.

### Quick Start

```bash
# Remove English duplicates (2,538 sets to review)
python scripts/deduplicate_cedict.py --mode english

# Remove Chinese duplicates (35 sets to review)
python scripts/deduplicate_cedict.py --mode chinese
```

### How it works

- For each duplicate, choose which translation to keep (1-N), or 0 to remove all
- Press Ctrl+C anytime to save progress and resume later
- Progress saved in `scripts/deduplication_progress_{mode}.json`
- Backup created automatically before applying changes

### Which duplicate to keep?

**Goal:** Create a bijective (1-to-1) dictionary for vector mapping alignment.

**Decision Rules:**

1. **Polysemous words → Choose 0 (remove all)**
   - If the Chinese word has multiple distinct meanings, remove all (even if not all meanings are shown)
   - If the English word has multiple distinct meanings, remove all
   - Example: 东北 → "Manchuria" vs "northeast" (different concepts) → Choose **0**

2. **Archaic/rare words → Don't keep**
   - Avoid classical Chinese (古文) and rare characters (生僻字)
   - Prefer modern, commonly-used words

3. **Synonyms → Keep one**
   - When duplicates are just synonyms (近义词), pick any one
   - Example: "catwalk" → T台 vs T型台 (same meaning) → Choose **1 or 2**

4. **When in doubt → Remove all (choose 0)**
   - If unsure whether meanings are truly synonymous, it's safer to remove

**目标：** 为向量映射准备双射（一对一）词典。

**选择规则：**

1. **多义词 → 选 0（全删）** - 中文或英文有多个不同含义的都删掉
2. **古文/生僻字 → 不要留** - 选择现代、常用的词
3. **近义词 → 选一个** - 多个选项意思相同时，任选一个即可
4. **不确定时 → 选 0** - 拿不准就删掉

