"""
Vector Mapping for Cross-Lingual Word Embedding Alignment
Architecture for aligning Chinese and English word embeddings using Procrustes analysis.
"""

import numpy as np
from gensim.models import KeyedVectors
from typing import Dict, Set, List, Optional, Tuple
import re
from opencc import OpenCC


class EmbeddingLoader:
    """
    Loads and processes word embeddings from various formats.

    This class maintains state and ensures only one embedding file is loaded per instance.
    Supports Word2Vec binary and text formats with automatic compression handling.

    Examples:
        >>> # Load embeddings
        >>> loader = EmbeddingLoader()
        >>> loader.load_word2vec('GoogleNews.bin.gz')
        >>> print(len(loader))  # Vocabulary size

        >>> # Filter and process
        >>> loader.filter_vocabulary(common_words)
        >>> matrix = loader.get_embedding_matrix(['cat', 'dog'])

        >>> # Check word existence
        >>> if 'cat' in loader:
        >>>     vec = loader.get_embedding('cat')
    """

    def __init__(self):
        """Initialize the loader with empty state."""
        self.embeddings: Optional[Dict[str, np.ndarray]] = None
        self.source_file: Optional[str] = None
        self.embedding_dim: Optional[int] = None

    def _ensure_not_loaded(self):
        """
        Check that no embeddings have been loaded yet.

        Raises:
            RuntimeError: If embeddings have already been loaded
        """
        if self.embeddings is not None:
            raise RuntimeError(
                f"Embeddings already loaded from '{self.source_file}'. "
                f"Create a new EmbeddingLoader instance to load different embeddings."
            )

    def _ensure_loaded(self):
        """
        Check that embeddings have been loaded.

        Raises:
            RuntimeError: If no embeddings have been loaded yet
        """
        if self.embeddings is None:
            raise RuntimeError(
                "No embeddings loaded. Call load_word2vec() first."
            )

    def load_word2vec(
        self,
        filepath: str,
        format: str = 'auto',
        max_vocab: Optional[int] = None
    ) -> 'EmbeddingLoader':
        """
        Load pre-trained word embeddings from a Word2Vec format file.

        Args:
            filepath: Path to the embedding file (supports .bin, .bin.gz, .vec, .txt, .bz2)
            format: Loading format - 'auto' (default), 'binary', or 'text'
                   'auto': Auto-detect from file extension
                   'binary': Force Word2Vec binary format
                   'text': Force text format
            max_vocab: Maximum number of words to load (useful for testing). None = load all

        Returns:
            self (for method chaining)

        Raises:
            RuntimeError: If embeddings already loaded
            FileNotFoundError: If file doesn't exist

        Note:
            Vectors are stored in their original form without preprocessing.
            Use VectorPreprocessor for normalization, centering, or other transformations.

        Examples:
            >>> # Auto-detect format (recommended)
            >>> loader = EmbeddingLoader()
            >>> loader.load_word2vec('GoogleNews.bin.gz')
            >>>
            >>> # Load subset for testing
            >>> loader = EmbeddingLoader()
            >>> loader.load_word2vec('big.bin', max_vocab=10000)
            >>>
            >>> # Explicit format
            >>> loader = EmbeddingLoader()
            >>> loader.load_word2vec('custom.vec', format='text')
        """
        self._ensure_not_loaded()

        # Auto-detect format from file extension
        if format == 'auto':
            lower_path = filepath.lower()
            if any(ext in lower_path for ext in ['.bin.gz', '.bin']):
                binary = True
            else:
                # Default to text for .vec, .txt, .bz2, etc.
                binary = False
        elif format == 'binary':
            binary = True
        elif format == 'text':
            binary = False
        else:
            raise ValueError(f"Unknown format '{format}'. Use 'auto', 'binary', or 'text'")

        try:
            # Load using gensim (handles compression automatically)
            kv = KeyedVectors.load_word2vec_format(
                filepath,
                binary=binary,
                limit=max_vocab,
                no_header=False if binary else None  # Text format may or may not have header
            )

            # Convert to simple dict for easier downstream usage
            embeddings = {word: kv[word].copy() for word in kv.key_to_index}

            # Store results
            self.embeddings = embeddings
            self.source_file = filepath
            self.embedding_dim = kv.vector_size if embeddings else None

            return self

        except FileNotFoundError:
            raise FileNotFoundError(f"Embedding file not found: {filepath}")
        except Exception as e:
            raise RuntimeError(f"Error loading embeddings from {filepath}: {str(e)}")

    def filter_vocabulary(
        self,
        words: Set[str]
    ) -> 'EmbeddingLoader':
        """
        Filter stored embeddings to keep only words in the specified vocabulary.

        Args:
            words: Set of words to keep

        Returns:
            self (for method chaining)

        Raises:
            RuntimeError: If no embeddings have been loaded

        Examples:
            >>> loader = EmbeddingLoader()
            >>> loader.load_word2vec('large.bin')
            >>> # Keep only words that have dictionary translations
            >>> dict_words = {'cat', 'dog', 'house'}
            >>> loader.filter_vocabulary(dict_words)
            >>> print(len(loader))  # 3 (or fewer if some words not in embeddings)
        """
        self._ensure_loaded()

        self.embeddings = {word: vec for word, vec in self.embeddings.items() if word in words}

        return self

    def get_embedding_matrix(
        self,
        words: List[str]
    ) -> np.ndarray:
        """
        Convert stored embeddings to aligned 2D matrix for specified words.

        Creates a matrix where row i corresponds to words[i], ensuring
        alignment for parallel training data (e.g., source-target pairs).

        Args:
            words: Ordered list of words (defines row order in output matrix)

        Returns:
            Matrix of shape (len(words), embedding_dim) where row i is the
            embedding for words[i]. Words not in embeddings are skipped.

        Raises:
            RuntimeError: If no embeddings have been loaded
            ValueError: If none of the words are found in embeddings

        Examples:
            >>> loader = EmbeddingLoader()
            >>> loader.load_word2vec('embeddings.bin')
            >>> words = ['cat', 'dog']
            >>> matrix = loader.get_embedding_matrix(words)
            >>> matrix.shape  # (2, 300) - 2 words, 300-dim embeddings
            >>> matrix[0]  # Embedding for 'cat'
        """
        self._ensure_loaded()

        # Filter to words that exist in embeddings
        valid_words = [w for w in words if w in self.embeddings]

        if len(valid_words) == 0:
            raise ValueError("None of the provided words found in embeddings")

        # Stack vectors in order
        vectors = np.vstack([self.embeddings[word] for word in valid_words])

        return vectors

    def get_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Get the loaded embeddings dictionary.

        Returns:
            Dictionary mapping words to embedding vectors

        Raises:
            RuntimeError: If no embeddings have been loaded

        Examples:
            >>> loader = EmbeddingLoader()
            >>> loader.load_word2vec('embeddings.bin')
            >>> emb_dict = loader.get_embeddings()
            >>> print(emb_dict['cat'])  # numpy array
        """
        self._ensure_loaded()
        return self.embeddings

    def get_vocabulary(self) -> Set[str]:
        """
        Get the set of words in the loaded embeddings.

        Returns:
            Set of words (vocabulary)

        Raises:
            RuntimeError: If no embeddings have been loaded

        Examples:
            >>> loader = EmbeddingLoader()
            >>> loader.load_word2vec('embeddings.bin')
            >>> vocab = loader.get_vocabulary()
            >>> 'cat' in vocab  # True
        """
        self._ensure_loaded()
        return set(self.embeddings.keys())

    def get_embedding(self, word: str) -> np.ndarray:
        """
        Get the embedding vector for a specific word.

        Args:
            word: The word to look up

        Returns:
            Embedding vector for the word

        Raises:
            RuntimeError: If no embeddings have been loaded
            KeyError: If word not found in embeddings

        Examples:
            >>> loader = EmbeddingLoader()
            >>> loader.load_word2vec('embeddings.bin')
            >>> vec = loader.get_embedding('cat')
            >>> vec.shape  # (300,)
        """
        self._ensure_loaded()
        return self.embeddings[word]

    def __len__(self) -> int:
        """
        Return the vocabulary size.

        Returns:
            Number of words in the loaded embeddings

        Raises:
            RuntimeError: If no embeddings have been loaded

        Examples:
            >>> loader = EmbeddingLoader()
            >>> loader.load_word2vec('embeddings.bin')
            >>> print(len(loader))  # 100000
        """
        self._ensure_loaded()
        return len(self.embeddings)

    def __contains__(self, word: str) -> bool:
        """
        Check if a word exists in the loaded embeddings.

        Args:
            word: The word to check

        Returns:
            True if word exists, False otherwise

        Raises:
            RuntimeError: If no embeddings have been loaded

        Examples:
            >>> loader = EmbeddingLoader()
            >>> loader.load_word2vec('embeddings.bin')
            >>> 'cat' in loader  # True
            >>> 'asdfghjkl' in loader  # False
        """
        self._ensure_loaded()
        return word in self.embeddings


class DictionaryParser:
    """
    Parses bilingual dictionaries from various formats (CC-CEDICT, MUSE)
    with support for traditional/simplified Chinese conversion.

    This class maintains state and ensures only one dictionary is loaded per instance.
    Use OpenCC for automatic traditional/simplified Chinese conversion.

    Examples:
        >>> # Parse CEDICT with only simplified Chinese
        >>> parser = DictionaryParser()
        >>> parser.parse_cedict('cedict_ts.u8', include_traditional=False)
        >>> print(len(parser.get_pairs()))  # Number of translation pairs

        >>> # Parse with both traditional and simplified
        >>> parser2 = DictionaryParser()
        >>> parser2.parse_cedict('cedict_ts.u8', include_traditional=True)
        >>> parser2.filter_by_vocabulary(zh_vocab, en_vocab)
        >>> parser2.save('filtered_dict.txt')

        >>> # Parse MUSE format
        >>> parser3 = DictionaryParser()
        >>> parser3.parse_muse_format('zh-en.txt', include_traditional=False)
    """

    def __init__(self):
        """Initialize the parser with OpenCC converters and empty state."""
        self.pairs: Optional[List[Tuple[str, str]]] = None
        self.source_file: Optional[str] = None
        self.format_type: Optional[str] = None

        # Initialize OpenCC converters
        try:
            self._t2s_converter = OpenCC('t2s')  # Traditional to Simplified
            self._s2t_converter = OpenCC('s2t')  # Simplified to Traditional
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenCC converters: {str(e)}")

    def _ensure_not_loaded(self):
        """
        Check that no dictionary has been loaded yet.

        Raises:
            RuntimeError: If a dictionary has already been loaded
        """
        if self.pairs is not None:
            raise RuntimeError(
                f"Dictionary already loaded from '{self.source_file}' "
                f"(format: {self.format_type}). Create a new DictionaryParser instance "
                f"to load a different dictionary."
            )

    def _convert_with_variants(
        self,
        chinese_text: str,
        include_traditional: bool
    ) -> List[str]:
        """
        Convert Chinese text to simplified (and optionally traditional) forms.

        Args:
            chinese_text: Original Chinese text
            include_traditional: If True, return both forms; if False, only simplified

        Returns:
            List of Chinese variants (1 or 2 elements). Duplicates are removed.

        Examples:
            >>> parser = DictionaryParser()
            >>> parser._convert_with_variants('中国', False)
            ['中国']
            >>> parser._convert_with_variants('中國', True)
            ['中国', '中國']  # Both forms
            >>> parser._convert_with_variants('中国', True)
            ['中国']  # Only one form if already simplified
        """
        # Always convert to simplified
        simplified = self._t2s_converter.convert(chinese_text)

        if not include_traditional:
            return [simplified]

        # Also get traditional form
        traditional = self._s2t_converter.convert(chinese_text)

        # Return both if different, otherwise just simplified
        if simplified != traditional:
            return [simplified, traditional]
        else:
            return [simplified]

    def parse_cedict(
        self,
        filepath: str,
        include_traditional: bool = False
    ) -> 'DictionaryParser':
        """
        Parse CC-CEDICT format dictionary file.

        CC-CEDICT format:
            Traditional Simplified [pin1 yin1] /English 1/English 2/...
            Example: 中國 中国 [Zhong1 guo2] /China/Middle Kingdom/

        Args:
            filepath: Path to CC-CEDICT file (usually .u8 extension)
            include_traditional: If True, generate pairs for both simplified and
                               traditional forms (when different)

        Returns:
            self (for method chaining)

        Raises:
            RuntimeError: If dictionary already loaded
            FileNotFoundError: If file doesn't exist

        Examples:
            >>> parser = DictionaryParser()
            >>> parser.parse_cedict('cedict_ts.u8')
            >>> pairs = parser.get_pairs()
            >>> pairs[0]  # ('中国', 'China')
        """
        self._ensure_not_loaded()

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except FileNotFoundError:
            raise FileNotFoundError(f"CEDICT file not found: {filepath}")
        except Exception as e:
            raise RuntimeError(f"Error reading CEDICT file {filepath}: {str(e)}")

        pairs = []

        # CEDICT format: Traditional Simplified [pinyin] /def1/def2/.../
        # Example: 中國 中国 [Zhong1 guo2] /China/Middle Kingdom/
        cedict_pattern = re.compile(
            r'^(\S+)\s+(\S+)\s+\[([^\]]+)\]\s+/(.+)/$'
        )

        for line in lines:
            line = line.strip()

            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue

            match = cedict_pattern.match(line)
            if not match:
                continue

            traditional, simplified, pinyin, definitions = match.groups()

            # Split definitions by '/'
            def_list = [d.strip() for d in definitions.split('/') if d.strip()]

            # Use the first English definition (most common meaning)
            if not def_list:
                continue

            english = def_list[0]

            # Generate variants based on include_traditional
            chinese_variants = self._convert_with_variants(simplified, include_traditional)

            # Add a pair for each variant
            for chinese in chinese_variants:
                pairs.append((chinese, english))

        # Store results
        self.pairs = pairs
        self.source_file = filepath
        self.format_type = 'cedict'

        return self

    def parse_muse_format(
        self,
        filepath: str,
        include_traditional: bool = False
    ) -> 'DictionaryParser':
        """
        Parse MUSE (Facebook) format dictionary file.

        MUSE format (tab or space separated):
            source_word target_word
            Example: 中国 China

        Args:
            filepath: Path to MUSE format dictionary file
            include_traditional: If True, generate pairs for both simplified and
                               traditional forms (when different)

        Returns:
            self (for method chaining)

        Raises:
            RuntimeError: If dictionary already loaded
            FileNotFoundError: If file doesn't exist

        Examples:
            >>> parser = DictionaryParser()
            >>> parser.parse_muse_format('zh-en.txt')
            >>> pairs = parser.get_pairs()
        """
        self._ensure_not_loaded()

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except FileNotFoundError:
            raise FileNotFoundError(f"MUSE dictionary file not found: {filepath}")
        except Exception as e:
            raise RuntimeError(f"Error reading MUSE file {filepath}: {str(e)}")

        pairs = []

        for line_num, line in enumerate(lines, 1):
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue

            # Split by tab or space
            parts = re.split(r'\s+', line)

            if len(parts) < 2:
                # Skip malformed lines
                continue

            chinese = parts[0]
            english = parts[1]

            # Generate variants based on include_traditional
            chinese_variants = self._convert_with_variants(chinese, include_traditional)

            # Add a pair for each variant
            for chinese_variant in chinese_variants:
                pairs.append((chinese_variant, english))

        # Store results
        self.pairs = pairs
        self.source_file = filepath
        self.format_type = 'muse'

        return self

    def filter_by_vocabulary(
        self,
        zh_vocab: Set[str],
        en_vocab: Set[str]
    ) -> 'DictionaryParser':
        """
        Filter stored translation pairs to only include words present in both vocabularies.

        This is essential for training: ensures every word pair has embeddings available.

        Args:
            zh_vocab: Set of Chinese words that have embeddings
            en_vocab: Set of English words that have embeddings

        Returns:
            self (for method chaining)

        Raises:
            RuntimeError: If no dictionary has been loaded yet

        Examples:
            >>> parser = DictionaryParser()
            >>> parser.parse_cedict('cedict.u8')
            >>> print(len(parser.get_pairs()))  # 100000
            >>> parser.filter_by_vocabulary(zh_vocab, en_vocab)
            >>> print(len(parser.get_pairs()))  # 50000 (after filtering)
        """
        if self.pairs is None:
            raise RuntimeError(
                "No dictionary loaded. Call parse_cedict() or parse_muse_format() first."
            )

        # Filter pairs where both words have embeddings
        filtered_pairs = [
            (zh, en) for zh, en in self.pairs
            if zh in zh_vocab and en in en_vocab
        ]

        self.pairs = filtered_pairs

        return self

    def save(self, filepath: str) -> None:
        """
        Save parsed dictionary pairs to a tab-separated file.

        Output format:
            chinese<tab>english<newline>

        Args:
            filepath: Path where to save the dictionary

        Raises:
            RuntimeError: If no dictionary has been loaded

        Examples:
            >>> parser = DictionaryParser()
            >>> parser.parse_cedict('cedict.u8')
            >>> parser.filter_by_vocabulary(zh_vocab, en_vocab)
            >>> parser.save('filtered_dict.txt')
        """
        if self.pairs is None:
            raise RuntimeError(
                "No dictionary loaded. Call parse_cedict() or parse_muse_format() first."
            )

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                for chinese, english in self.pairs:
                    f.write(f"{chinese}\t{english}\n")
        except Exception as e:
            raise RuntimeError(f"Error saving dictionary to {filepath}: {str(e)}")

    def get_pairs(self) -> List[Tuple[str, str]]:
        """
        Get the loaded translation pairs.

        Returns:
            List of (chinese, english) tuples

        Raises:
            RuntimeError: If no dictionary has been loaded

        Examples:
            >>> parser = DictionaryParser()
            >>> parser.parse_cedict('cedict.u8')
            >>> pairs = parser.get_pairs()
            >>> print(pairs[0])  # ('中国', 'China')
        """
        if self.pairs is None:
            raise RuntimeError(
                "No dictionary loaded. Call parse_cedict() or parse_muse_format() first."
            )

        return self.pairs
