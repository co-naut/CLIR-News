"""
Vector Mapping for Cross-Lingual Word Embedding Alignment
Architecture for aligning Chinese and English word embeddings using Procrustes analysis.
"""

import numpy as np
from gensim.models import KeyedVectors
from typing import Dict, Set, List, Optional


class EmbeddingLoader:
    """
    Loads and processes word embeddings from various formats.

    Supports Word2Vec binary and text formats with automatic compression handling.
    """

    @staticmethod
    def load_word2vec(
        filepath: str,
        format: str = 'auto',
        max_vocab: Optional[int] = None,
        normalize: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Load pre-trained word embeddings from a Word2Vec format file.

        Args:
            filepath: Path to the embedding file (supports .bin, .bin.gz, .vec, .txt, .bz2)
            format: Loading format - 'auto' (default), 'binary', or 'text'
                   'auto': Auto-detect from file extension
                   'binary': Force Word2Vec binary format
                   'text': Force text format
            max_vocab: Maximum number of words to load (useful for testing). None = load all
            normalize: If True, L2-normalize vectors on load

        Returns:
            Dictionary mapping words (str) to embedding vectors (np.ndarray)

        Examples:
            >>> # Auto-detect format (recommended)
            >>> embeddings = EmbeddingLoader.load_word2vec('GoogleNews.bin.gz')
            >>>
            >>> # Load subset for testing
            >>> small_emb = EmbeddingLoader.load_word2vec('big.bin', max_vocab=10000)
            >>>
            >>> # Explicit format
            >>> text_emb = EmbeddingLoader.load_word2vec('custom.vec', format='text')
        """
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

            # Normalize if requested
            if normalize:
                kv.init_sims(replace=True)

            # Convert to simple dict for easier downstream usage
            embeddings = {word: kv[word].copy() for word in kv.key_to_index}

            return embeddings

        except FileNotFoundError:
            raise FileNotFoundError(f"Embedding file not found: {filepath}")
        except Exception as e:
            raise RuntimeError(f"Error loading embeddings from {filepath}: {str(e)}")

    @staticmethod
    def filter_vocabulary(
        embeddings: Dict[str, np.ndarray],
        words: Set[str]
    ) -> Dict[str, np.ndarray]:
        """
        Filter embeddings to keep only words in the specified vocabulary.

        Args:
            embeddings: Dictionary of word embedding vector
            words: Set of words to keep

        Returns:
            Filtered dictionary containing only words in the vocabulary set

        Examples:
            >>> embeddings = EmbeddingLoader.load_word2vec('large.bin')
            >>> # Keep only words that have dictionary translations
            >>> dict_words = {'cat', 'dog', 'house'}
            >>> filtered = EmbeddingLoader.filter_vocabulary(embeddings, dict_words)
            >>> len(filtered)  # 3 (or fewer if some words not in embeddings)
        """
        return {word: vec for word, vec in embeddings.items() if word in words}

    @staticmethod
    def get_embedding_matrix(
        embeddings: Dict[str, np.ndarray],
        words: List[str]
    ) -> np.ndarray:
        """
        Convert embeddings dictionary to aligned 2D matrix.

        Creates a matrix where row i corresponds to words[i], ensuring
        alignment for parallel training data (e.g., source-target pairs).

        Args:
            embeddings: Dictionary of word embedding vector
            words: Ordered list of words (defines row order in output matrix)

        Returns:
            Matrix of shape (len(words), embedding_dim) where row i is the
            embedding for words[i]. Words not in embeddings are skipped.

        Examples:
            >>> embeddings = {'cat': np.array([1, 2, 3]), 'dog': np.array([4, 5, 6])}
            >>> words = ['cat', 'dog']
            >>> matrix = EmbeddingLoader.get_embedding_matrix(embeddings, words)
            >>> matrix.shape  # (2, 3)
            >>> matrix[0]  # array([1, 2, 3]) - embedding for 'cat'
        """
        # Filter to words that exist in embeddings
        valid_words = [w for w in words if w in embeddings]

        if len(valid_words) == 0:
            raise ValueError("None of the provided words found in embeddings")

        # Stack vectors in order
        vectors = np.vstack([embeddings[word] for word in valid_words])

        return vectors
