"""

References
----------

https://marknelson.us/posts/2014/10/19/data-compression-with-arithmetic-coding.html
https://marknelson.us/posts/1991/02/01/arithmetic-coding-statistical-modeling-data-compression
https://marknelson.us/posts/2014/10/19/data-compression-with-arithmetic-coding.html

https://www.cs.cmu.edu/~aarti/Class/10704/Intro_Arith_coding.pdf




0xFFFF has 16**4 = (2**4)**4 = 2**16 = 16 bits
There are 8 bits in one byte, so it has 2 bytes.



"""

import itertools


def ranges_from_frequencies(frequencies):
    """Build a dictionary of ranges from a dictionary of frequencies.

    Examples
    --------
    >>> dict(ranges_from_frequencies({'a':5, 'b':3, 'c':2}))
    {'a': (0, 5), 'b': (5, 8), 'c': (8, 10)}
    """
    cumsum = 0
    for symbol, frequency in sorted(frequencies.items()):
        yield (symbol, (cumsum, cumsum + frequency))
        cumsum += frequency


def search_ranges(i, ranges):
    """Search the range for i and return the symbol.

    Examples
    --------
    >>> ranges = {'a': (0, 5), 'b': (5, 8), 'c': (8, 10)}
    >>> search_ranges(2, ranges)
    'a'
    >>> search_ranges(5, ranges)
    'b'
    """
    for symbol, (low, high) in ranges.items():
        if low <= i < high:
            return symbol


def print_table(low, high, bits):
    for number in reversed(range(low, high + 1)):
        print(f" 0b{number:0{bits}b} ({number})")


class ArithmeticEncoder:
    def __init__(self, frequencies, *, bits=6, verbose=0, EOM="<EOM>"):
        self.EOM = EOM
        assert self.EOM in frequencies.keys()
        assert all(isinstance(freq, int) for freq in frequencies.values())
        self.frequencies = frequencies
        self.bits = bits
        self.verbose = verbose

        # Build ranges from frequencies
        self.ranges = dict(ranges_from_frequencies(self.frequencies))
        self.total_count = sum(self.frequencies.values())

        # The total range. Examples in comments are with 4 bits
        self.TOP_VALUE = (1 << self.bits) - 1  # 0b1111 = 15
        self.FIRST_QUARTER = (self.TOP_VALUE >> 2) + 1  # 0b0100 = 4
        self.HALF = self.FIRST_QUARTER * 2  # 0b1000 = 8
        self.THIRD_QUARTER = self.FIRST_QUARTER * 3  # 0b1100 = 12
        self.bits_to_follow = 0  # Counter

        if self.verbose > 0:
            print("Initialized with:")
            print(
                f" TOP_VALUE     = 0b{self.TOP_VALUE:0{self.bits}b} ({self.TOP_VALUE})"
            )
            print(
                f" THIRD_QUARTER = 0b{self.THIRD_QUARTER:0{self.bits}b} ({self.THIRD_QUARTER})"
            )
            print(f" HALF          = 0b{self.HALF:0{self.bits}b} ({self.HALF})")
            print(
                f" FIRST_QUARTER = 0b{self.FIRST_QUARTER:0{self.bits}b} ({self.FIRST_QUARTER})"
            )
            print(f" total_count   = {self.total_count}")

    def print_state(self, low, high, prefix=" "):
        if self.verbose > 0:
            range_ = high - low + 1
            print(prefix + f"High value: 0b{high:0{self.bits}b} ({high})")
            print(prefix + f"Low value:  0b{low:0{self.bits}b} ({low})")
            print(prefix + f"Range: [{low}, {high + 1}) Width: {range_}")

    def bit_plus_follow(self, bit):
        """Yield the bit, then yield opposite bits `self.bits_to_follow` times."""
        yield bit
        yield from itertools.repeat(int(not bit), times=self.bits_to_follow)
        self.bits_to_follow = 0  # Reset the counter

    def decode(self, iterable):
        # iterable = iter(iterable)

        # Set up low and high values
        low = 0
        high = self.TOP_VALUE

        # Consume the first `self.bits` into the `value` variable.
        # For instance, if iterable = [0, 1, 0, 1] and self.bits = 6,
        # then value = 0b010100 after this step
        iterable = itertools.chain(iter(iterable), itertools.repeat(0))
        value = 0
        first_bits = itertools.islice(iterable, self.bits)
        for i, input_bit in enumerate(first_bits, 1):
            if self.verbose:
                print(f"\nProcessing bit {i}: {input_bit}")
                print("-" * 32)

            value = (value << 1) + input_bit
            print(f"Value: 0b{value:0{self.bits}b} ({value})")

        # General loop
        while True:
            # Current range and current scaled value
            range_ = high - low + 1
            scaled_value = ((value - low + 1) * self.total_count - 1) / range_
            print(f"{range_=}")
            print(f"{scaled_value=}")

            symbol = search_ranges(scaled_value, self.ranges)
            yield symbol
            if symbol == self.EOM:
                break

            symbol_low, symbol_high = self.ranges[symbol]
            high = low + int(range_ * symbol_high / self.total_count) - 1
            low = low + int(range_ * symbol_low / self.total_count)

            self.print_state(low, high, " ")

            while True:
                if high < self.HALF:
                    print("In bottom half of interval")
                    pass
                elif low >= self.HALF:
                    print("In top half of interval")
                    value -= self.HALF
                    low -= self.HALF
                    high -= self.HALF
                elif low >= self.FIRST_QUARTER and high < self.THIRD_QUARTER:
                    print("In middle half of interval")
                    value -= self.FIRST_QUARTER
                    low -= self.FIRST_QUARTER
                    high -= self.FIRST_QUARTER
                else:
                    break

                low = 2 * low
                high = 2 * high + 1
                try:
                    value = 2 * value + next(iterable)
                except StopIteration:
                    value = 2 * value
                    break
                self.print_state(low, high, " ")
                print()

    def encode(self, iterable):
        """Encode an iterable of symbols, yielding bits (0/1).

        Examples
        --------
        >>> message = iter(['A', 'B', '<EOM>'])
        >>> frequencies = {'A': 5, 'B':2, '<EOM>':1}
        >>> encoder = ArithmeticEncoder(frequencies=frequencies)
        >>> list(encoder.encode(message))
        [1, 0, 0, 1, 1, 0, 1]
        """
        iterable = iter(iterable)

        # Initial low and high values for the range [low, high)
        low = 0
        high = self.TOP_VALUE

        assert self.total_count <= ((high + 1) / 4) + 1  # Equation on page 533

        # Loop over every symbol in the input stream `iterable`
        for i, symbol in enumerate(iterable, 1):
            if self.verbose > 0:
                print(f"\nProcessing symbol number {i}: {repr(symbol)}")
                print("-" * 32)

            range_ = high - low + 1
            assert range_ >= self.total_count, "Not enough precision"

            # Algorithm invariants
            assert low <= high
            assert 0 <= low <= (1 << self.bits) - 1
            assert 0 <= high <= (1 << self.bits) - 1

            if self.verbose > 0:
                print("Initial range:")
                self.print_state(low, high, "")

            symbol_low, symbol_high = self.ranges[symbol]

            # Transform the range
            range_ = high - low + 1
            high = low + int((symbol_high / self.total_count) * range_) - 1
            low = low + int((symbol_low / self.total_count) * range_)

            prob = (symbol_high - symbol_low) / self.total_count
            if self.verbose > 0:
                print(f"\nTransformed range (prob. of symbol '{symbol}': {prob:.4f}):")
                self.print_state(low, high, "")
                print("Going through bits.")
            while True:
                if high < self.HALF:
                    if self.verbose > 0:
                        print(" Range in lower half")
                    # Since HALF > `high` > `low`, both `high` and `low` have
                    # 0 in the first bit. We output this 0 bit.
                    yield from self.bit_plus_follow(bit=0)
                elif low >= self.HALF:
                    if self.verbose > 0:
                        print(" Range in upper half")

                    # Since `high` > `low` >= HALF, both `high` and `low` have
                    # 0 as the first bit. We output this 1 bit.
                    yield from self.bit_plus_follow(bit=1)

                    # HALF is 0b1000..., and we remove the first bit from
                    # both `low` and `high`. An example:
                    # low : 0b10110 => 0b00110
                    # high: 0b11100 => 0b01100
                    low -= self.HALF
                    high -= self.HALF
                elif low >= self.FIRST_QUARTER and high < self.THIRD_QUARTER:
                    if self.verbose > 0:
                        print(" Range in middle half")

                    # At this point we know that `low` is in the second quarter
                    # and `high` is in the third quarter. Therefore the first
                    # two bits in `low` must be 01 and the first two bits in
                    # high must be 10.

                    # The lines below ...

                    self.bits_to_follow += 1
                    low -= self.FIRST_QUARTER
                    high -= self.FIRST_QUARTER
                else:
                    break

                # Scale up bits by shifting every bit to the left, then adding
                # a 0 to `low` and a 1 to `high`. Here is an example:
                # low : 0b00110 => 0b01100
                # high: 0b01100 => 0b11001
                low = 2 * low
                high = 2 * high + 1
                if self.verbose > 0:
                    self.print_state(low, high, " ")
                    print()

        # Check that the last symbol was EOM
        if symbol != self.EOM:
            raise ValueError("Last symbol must be {repr(self.EOM)}, got {repr(symbol)}")

        # Finish encoding
        self.bits_to_follow += 1

        # If low < FIRST_QUARTER, then yield 0, else yield 1
        yield from self.bit_plus_follow(int(low >= self.FIRST_QUARTER))


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v", "--doctest-modules"])


if __name__ == "__main__":
    import random

    message = ["A", "B", "B", "B", "A", "<EOM>"]
    message = random.choices(["A", "B"], k=3) + ["<EOM>"]
    frequencies = {"A": 5, "B": 2, "<EOM>": 1}

    encoder = ArithmeticEncoder(frequencies=frequencies, verbose=1)

    bits = list(encoder.encode(message))

    print("Final output:", bits)

    for symbol in encoder.decode(bits):
        print(symbol)

    decoded = list(encoder.decode(bits))
    print(decoded)

    assert decoded == message
