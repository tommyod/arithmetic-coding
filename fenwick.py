#!/usr/bin/env python3


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


if __name__ == "__main__":
    import doctest

    doctest.testmod()
