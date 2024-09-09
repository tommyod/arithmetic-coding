#!/usr/bin/env python3
"""
Cumulative sum objects and Fenwick trees for fast operations
============================================================

Fenwick tree and CumulativeSum classes designed to work with adaptive models.

In an adaptive model we do not use frequencies like

>>> frequencies = {'a': 4, 'b': 1, 'c': 3}

in the arithmetic coder. Instead we provide a list of symbols like

>>> symbols = ['a', 'b', 'c']

and set each count to one.

>>> frequencies = {symbol:1 for symbol in frequencies}

A cumulative sum object is updated as the model see more of each symbol.

>>> cumsum = CumulativeSum(frequencies)
>>> cumsum.get_low_high('a')
(0, 1)
>>> cumsum.add_count('a', 1)
>>> cumsum.get_low_high('a')
(0, 2)
>>> cumsum.get_low_high('b')
(2, 3)

By using a Fenwick tree we can get O(log n) time operations for getting and
setting counts as symbols are seen. Since the access pattern of the encoder
is to alternate between getting symbol counts and updating them, this gives
O(log n) performance instead of O(n). In practice n is the unique number of
symbols, which is not a large value, so this does not matter that much.
Still nice to use a data structure with good asymptotic performance though.


"""


class FenwickTree:
    """A data structure for maintaining cumulative (prefix) sums.
    All operations are O(log n).

    This implementation is based on: https://github.com/dstein64/fenwick

    Examples
    --------
    >>> frequencies = [1, 0, 2, 1, 1, 3, 0, 4]
    >>> ft = FenwickTree(frequencies)
    """

    def __init__(self, frequencies):
        """Initializes n frequencies to zero."""
        self._v = list(frequencies)

        # Initialize in O(n) with specified frequencies.
        for idx in range(1, len(self) + 1):
            parent_idx = idx + (idx & -idx)  # parent in update tree
            if parent_idx <= len(self):
                self._v[parent_idx - 1] += self._v[idx - 1]

    def __len__(self):
        return len(self._v)

    def prefix_sum(self, stop):
        """Returns sum of first elements (sum up to *stop*, exclusive).

        Examples
        --------
        >>> ft = FenwickTree([1, 0, 2, 1, 1, 3, 0, 4])
        >>> ft.prefix_sum(1) == 1
        True
        >>> ft.prefix_sum(2) == 1 + 0
        True
        >>> ft.prefix_sum(3) == 1 + 0 + 2
        True
        >>> ft.prefix_sum(4) == 1 + 0 + 2 + 1
        True
        """
        if stop <= 0 or stop > len(self):
            raise IndexError("index out of range")
        _sum = 0
        while stop > 0:
            _sum += self._v[stop - 1]
            stop &= stop - 1
        return _sum

    def range_sum(self, start, stop):
        """Returns sum from start (inclusive) to stop (exclusive).

        Examples
        --------
        >>> ft = FenwickTree([1, 0, 2, 1, 1, 3])
        >>> ft.range_sum(0, 3) == 1 + 0 + 2
        True
        >>> ft.range_sum(0, 5) == 1 + 0 + 2 + 1 + 1
        True

        """
        if start < 0 or start >= len(self):
            raise IndexError("index out of range")
        if stop <= start or stop > len(self):
            raise IndexError("index out of range")
        result = self.prefix_sum(stop)
        if start > 0:
            result -= self.prefix_sum(start)
        return result

    def __getitem__(self, idx):
        """Get item value (not cumsum) at index.

        Examples
        --------
        >>> ft = FenwickTree([1, 0, 2, 1, 1, 3, 0, 4])
        >>> ft[0], ft[1], ft[2], ft[3]
        (1, 0, 2, 1)
        >>> ft[-1]
        4
        """
        if isinstance(idx, int):
            idx = idx % len(self)
            return self.range_sum(idx, idx + 1)
        else:
            raise IndexError(f"Indexing only works with integers, got {idx}")

    def frequencies(self):
        """Retrieves all frequencies in O(n).

        Examples
        --------
        >>> ft = FenwickTree([1, 0, 2, 1, 1])
        >>> ft.frequencies()
        [1, 0, 2, 1, 1]
        """
        _frequencies = [0] * len(self)
        for idx in range(1, len(self) + 1):
            _frequencies[idx - 1] += self._v[idx - 1]
            parent_idx = idx + (idx & -idx)
            if parent_idx <= len(self):
                _frequencies[parent_idx - 1] -= self._v[idx - 1]
        return _frequencies

    def add(self, idx, k):
        """Adds k to idx'th element (0-based indexing).

        Examples
        --------
        >>> ft = FenwickTree([1, 0, 2, 1, 1])
        >>> ft.add(0, 2)
        >>> ft.add(3, -4)
        >>> ft.frequencies()
        [3, 0, 2, -3, 1]
        >>> ft.range_sum(0, 4) == 3 + 0 + 2 - 3
        True
        """
        if idx < 0 or idx >= len(self):
            raise IndexError("index out of range")
        idx += 1
        while idx <= len(self):
            self._v[idx - 1] += k
            idx += idx & -idx

    def __setitem__(self, idx, value):
        # It's more efficient to use add directly, as opposed to
        # __setitem__, since the latter calls __getitem__.
        self.add(idx, value - self[idx])

    def bisect_left(self, value):
        """
        Returns the smallest index i such that the cumulative sum up to i is >= value.
        If no such index exists, returns len(self).
        This operation is O(log n).

        Examples
        --------
        >>> ft = FenwickTree([1, 3, 5, 10])
        >>> ft.prefix_sum(4)
        19
        >>> ft.bisect_left(2)
        1
        >>> ft.bisect_left(9)
        3
        >>> ft.bisect_left(1)
        1
        >>> ft.bisect_left(0.5)
        0
        >>> ft.bisect_left(99)
        4
        """
        # https://stackoverflow.com/questions/34699616/fenwick-trees-to-determine-which-interval-a-point-falls-in
        j = 2 ** len(self)

        i = -1
        while j > 0:
            if i + j < len(self) and value >= self._v[i + j]:
                value -= self._v[i + j]
                i += j
            j >>= 1
        return i + 1

    def __eq__(self, other):
        return isinstance(other, FenwickTree) and self._v == other._v


class NaiveCumulativeSum:
    """Cumulative sum with slow asymptotic performance."""

    def __init__(self, frequencies, update=True):
        """Create cumulative sum in O(n) time."""
        self.frequencies = dict(frequencies)
        self.ranges = dict(self.ranges_from_frequencies(self.frequencies))
        self.update = update

    def get_low_high(self, symbol):
        """Get (low, high) for symbol in O(1) time."""
        return self.ranges[symbol]

    def add_count(self, symbol, value):
        """Update count in O(n) time."""
        if self.update:
            self.frequencies[symbol] += value
            self.ranges = dict(self.ranges_from_frequencies(self.frequencies))

    def total_count(self):
        """Get sum of all frequencies in O(n) time."""
        return sum(self.frequencies.values())

    def reset(self):
        """Set all frequency counts to one."""
        self.frequencies = {frequency: 1 for frequency in self.frequencies}
        self.ranges = dict(self.ranges_from_frequencies(self.frequencies))

    @staticmethod
    def ranges_from_frequencies(frequencies):
        """Build a dictionary of ranges from a dictionary of frequencies.

        Examples
        --------
        >>> freq = {'a': 5, 'b': 3, 'c': 2}
        >>> dict(NaiveCumulativeSum.ranges_from_frequencies(freq))
        {'a': (0, 5), 'b': (5, 8), 'c': (8, 10)}
        """
        cumsum = 0
        for symbol, frequency in sorted(frequencies.items()):
            yield (symbol, (cumsum, cumsum + frequency))
            cumsum += frequency

    def search_ranges(self, value):
        """Find symbol such that low <= value < high in O(n) time.

        Examples
        --------
        >>> cumsum = NaiveCumulativeSum({'a': 5, 'b': 3, 'c': 2})
        >>> cumsum.search_ranges(2)
        'a'
        >>> cumsum.search_ranges(5)
        'b'
        """
        for symbol, (low, high) in self.ranges.items():
            if low <= value < high:
                return symbol
        raise ValueError("Could not locate value in ranges.")


class CumulativeSum:
    """Cumulative sum with fast asymptotic performance."""

    def __init__(self, frequencies, update=True):
        """Create cumulative sum in O(n) time."""
        symbols = sorted(frequencies.keys())
        self.idx_to_symbol = dict(enumerate(symbols))
        self.symbol_to_idx = {s: i for (i, s) in self.idx_to_symbol.items()}
        self.fenwick_tree = FenwickTree([frequencies[s] for s in symbols])
        self.update = update

    def get_low_high(self, symbol):
        """Get (low, high) for symbol in O(log n) time.

        Examples
        --------
        >>> cumsum = CumulativeSum({'a': 2, 'b': 3, 'c': 4})
        >>> cumsum.get_low_high('a')
        (0, 2)
        >>> cumsum.get_low_high('b')
        (2, 5)
        >>> cumsum.get_low_high('c')
        (5, 9)
        """
        idx = self.symbol_to_idx[symbol]
        if idx == 0:
            return (0, self.fenwick_tree[idx])

        sum_upto = self.fenwick_tree.prefix_sum(idx)
        return (sum_upto, sum_upto + self.fenwick_tree[idx])

    def add_count(self, symbol, value):
        """Update count in O(log n) time.

        Examples
        --------
        >>> cumsum = CumulativeSum({'a': 2, 'b': 3, 'c': 4})
        >>> cumsum.add_count('b', 2)
        >>> cumsum.get_low_high('a')
        (0, 2)
        >>> cumsum.get_low_high('b')
        (2, 7)
        >>> cumsum.get_low_high('c')
        (7, 11)
        """
        if self.update:
            idx = self.symbol_to_idx[symbol]
            self.fenwick_tree.add(idx, value)

    def total_count(self):
        """Get sum of all frequencies in O(log n) time.

        Examples
        --------
        >>> cumsum = CumulativeSum({'a': 2, 'b': 3, 'c': 4})
        >>> cumsum.total_count()
        9
        >>> cumsum.add_count('c', 2)
        >>> cumsum.total_count()
        11
        """
        return self.fenwick_tree.prefix_sum(len(self.fenwick_tree))

    def reset(self):
        """Set all frequency counts to one."""
        self.fenwick_tree = FenwickTree([1] * len(self.fenwick_tree))

    def search_ranges(self, value):
        """Find symbol such that low <= value < high in O(n) time.

        Examples
        --------
        >>> cumsum = CumulativeSum({'a': 5, 'b': 3, 'c': 2})
        >>> cumsum.search_ranges(2)
        'a'
        >>> cumsum.search_ranges(5)
        'b'
        """
        idx = self.fenwick_tree.bisect_left(value)
        return self.idx_to_symbol[idx]


if __name__ == "__main__":
    import doctest

    doctest.testmod()
