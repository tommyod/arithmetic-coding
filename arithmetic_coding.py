# -*- coding: utf-8 -*-
"""
This module implements a ArithmeticEncoder class for encoding and decoding.



Minimal example
===============

Create a message, which is an iterable consisting of hashable symbols.

>>> message = ['A', 'B', 'B', 'B', '<EOM>']

Create frequency counts - the model needs to know how common each symbol is.
The essential compression idea is that high-frequency symbols get fewers bits.

>>> frequencies = {'A': 1, 'B': 3, '<EOM>': 1}

Now create the encoder and encoe the message.

>>> encoder = ArithmeticEncoder(frequencies=frequencies)
>>> bits = list(encoder.encode(message))
>>> bits
[0, 1, 0, 1, 1, 0, 0, 1]

Verify that decoding brings back the original message.

>>> list(encoder.decode(bits))
['A', 'B', 'B', 'B', '<EOM>']


Compression of infrequent symbols
=================================

Here is an example with many common letters. In 'Crime and Punishment' by 
Fyodor Dostoyevsky the symbol 'e' is around 136 times more frequent than 'q'.

>>> import random
>>> rng = random.Random(42)
>>> message = rng.choices(['e', 'q'], weights=[136, 1], k=10_000) + ["<EOM>"]
>>> frequencies = {'e': 13600, 'q': 100, '<EOM>': 1}

The 10_000 symbols are compressed to a small number of bits

>>> encoder = ArithmeticEncoder(frequencies=frequencies, bits=16)
>>> bits = list(encoder.encode(message))
>>> len(bits)
676

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


def search_ranges(value, ranges):
    """Find symbol such that low <= value < high.

    Examples
    --------
    >>> ranges = {'a': (0, 5), 'b': (5, 8), 'c': (8, 10)}
    >>> search_ranges(2, ranges)
    'a'
    >>> search_ranges(5, ranges)
    'b'
    """
    for symbol, (low, high) in ranges.items():
        if low <= value < high:
            return symbol
    raise ValueError("Could not locate value in ranges.")


def print_table(low, high, bits):
    """Print binary representation of numbers in range [low, high]."""
    for number in reversed(range(low, high + 1)):
        print(f" 0b{number:0{bits}b} ({number})")


class BitQueue:
    """A queue to keep track of bits to follow.

    Examples
    --------
    >>> bitqueue = BitQueue()
    >>> bitqueue += 3
    >>> list(bitqueue.bit_plus_follow(0))
    [0, 1, 1, 1]
    >>> bitqueue += 2
    >>> list(bitqueue.bit_plus_follow(1))
    [1, 0, 0]
    """

    bits_to_follow = 0  # Initialize the counter

    def __add__(self, bits):
        self.bits_to_follow += bits  # Add to the counter
        return self

    def bit_plus_follow(self, bit):
        yield bit  # Yield the bit, then `bits_to_follow` of the opposite bit
        yield from itertools.repeat(int(not bit), times=self.bits_to_follow)
        self.bits_to_follow = 0  # Reset the counter


class ArithmeticEncoder:
    """An implementation of arithmetic coding based on:

    - Ian H. Witten, Radford M. Neal, and John G. Cleary. 1987.
      Arithmetic coding for data compression.
      Commun. ACM 30, 6 (June 1987), 520–540.
      https://doi.org/10.1145/214762.214771
    - Data Compression With Arithmetic Coding
      https://marknelson.us/posts/2014/10/19/data-compression-with-arithmetic-coding.html

    This implementation pedagogical, not production ready code.
    You should probably not implement this in Python for real-world use
    cases, since the language is too slow and too high-level.
    """

    def __init__(self, frequencies, *, bits=6, verbose=0, EOM="<EOM>"):
        """Initialize an arithmetic encoder/decoder.

        Parameters
        ----------
        frequencies : dict
            A dictionary mapping symbols to frequencies, e.g. {'A':3, 'B':2}.
        bits : int, optional
            The number of bits to use in the buffer. The default is 6.
        verbose : int, optional
            How much information to print. The default is 0.
        EOM : str, optional
            An End Of Message (OEM) symbol. The default is '<EOM>'.

        Examples
        --------
        >>> message = ['A', 'B', 'B', 'B', '<EOM>']
        >>> frequencies = {'A': 1, 'B': 3, '<EOM>': 1}
        >>> encoder = ArithmeticEncoder(frequencies=frequencies)
        >>> bits = list(encoder.encode(message))
        >>> bits
        [0, 1, 0, 1, 1, 0, 0, 1]
        >>> list(encoder.decode(bits))
        ['A', 'B', 'B', 'B', '<EOM>']
        """
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

        # Equation on page 533 - check if there is enough precision
        if self.total_count > int((self.TOP_VALUE + 1) / 4) + 1:
            msg = "Insufficient precision to encode low-probability symbols."
            msg += "\nIncrease the value of `bits` in the encoder."
            raise Exception(msg)

        if self.verbose > 0:
            print("Initialized with:")
            print(f" bits          = {self.bits}")
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

    def _print_state(self, low, high, value=None, *, prefix=" ", end="\n"):
        range_ = high - low + 1
        print(prefix + f"High value: 0b{high:0{self.bits}b} ({high})")
        if value is not None:
            print(prefix + f"Value:      0b{value:0{self.bits}b} ({value})")
        print(prefix + f"Low value:  0b{low:0{self.bits}b} ({low})")
        print(prefix + f"Range: [{low}, {high + 1}) Width: {range_}", end=end)

    def encode(self, iterable):
        """Encode an iterable of symbols, yielding bits (0/1).

        Examples
        --------
        >>> message = iter(['A', 'B', '<EOM>'])
        >>> frequencies = {'A': 5, 'B': 2, '<EOM>': 1}
        >>> encoder = ArithmeticEncoder(frequencies=frequencies)
        >>> list(encoder.encode(message))
        [1, 0, 0, 1, 1, 0, 1]
        """
        if self.verbose:
            print("------------------------ ENCODING ------------------------")

        bit_queue = BitQueue()  # Keep track of bits to follow

        # Initial low and high values for the range [low, high)
        low = 0
        high = self.TOP_VALUE

        # Loop over every symbol in the input stream `iterable`
        for i, symbol in enumerate(iterable, 1):
            if self.verbose > 0:
                print(f"\nProcessing symbol number {i}: {repr(symbol)}")
                print("-" * 32)

            # Current range
            range_ = high - low + 1

            # Algorithm invariants
            assert 0 <= low <= high <= self.TOP_VALUE
            assert low < self.HALF <= high
            assert high - low > self.FIRST_QUARTER
            assert range_ >= self.total_count, "Not enough precision"

            # Print current state of the low and high values
            if self.verbose > 0:
                self._print_state(low, high, prefix="")

            # Get the symbol counts (non-normalized cumulative probabilities)
            symbol_low, symbol_high = self.ranges[symbol]

            # Transform the range [low, high) based on probability of symbol.
            # Note: due to floating point issues, even the order of operations
            # must match EXACTLY between the encoder and decoder here.
            high = low + int(range_ * symbol_high / self.total_count) - 1
            low = low + int(range_ * symbol_low / self.total_count)

            # Print state of low and high after transforming
            if self.verbose > 0:
                prob = (symbol_high - symbol_low) / self.total_count
                print(f"\nTransformed range (prob. of symbol '{symbol}': {prob:.4f}):")
                self._print_state(low, high, prefix="", end="\n\n")

            # This loop will run as long as one of the three cases below happen
            # (1) The first bit in `low` and `high` are both 0 (high < HALF)
            # (2) The first bit in  `low` and `high` are both 1 (low >= HALF)
            # (3) The first two bits in `low` and `high` are opposites
            while True:
                # Case (1): The first bits are both 0
                if high < self.HALF:
                    if self.verbose > 0:
                        print(" Range in lower half - both start with 0")
                        self._print_state(low, high, prefix="   ")
                    # Since HALF > `high` > `low`, both `high` and `low` have
                    # 0 in the first bit. We output this 0 bit.
                    yield from bit_queue.bit_plus_follow(bit=0)

                # Case (2): The first bits are both 0
                elif low >= self.HALF:
                    if self.verbose > 0:
                        print(" Range in upper half  - both start with 1")
                        self._print_state(low, high, prefix="   ")

                    # Since `high` > `low` >= HALF, both `high` and `low` have
                    # 1 as the first bit. We output this 1 bit.
                    yield from bit_queue.bit_plus_follow(bit=1)

                    # HALF is 0b1000..., and we remove the first bit from
                    # both `low` and `high`. An example:
                    # low : 0b10110 => 0b00110
                    # high: 0b11100 => 0b01100
                    low -= self.HALF
                    high -= self.HALF

                # Case (3): The first two bits are opposite
                elif low >= self.FIRST_QUARTER and high < self.THIRD_QUARTER:
                    if self.verbose > 0:
                        print(" Range in middle half - first 2 bits are opposite")
                        self._print_state(low, high, prefix="   ")

                    # At this point we know that `low` is in the second quarter
                    # and `high` is in the third quarter (since the other IF-
                    # statements did not trigger). Therefore the first
                    # two bits in `low` must be 01 and the first two bits in
                    # high must be 10.

                    # FIRST_QUARTER is 0b01000..., so these lines set the first
                    # two bits to 00 in `low` and set 01 in `high`. Example:
                    # low : 0b01xxx => 0b00xxx
                    # high: 0b10xxx => 0b01xxx
                    low -= self.FIRST_QUARTER
                    high -= self.FIRST_QUARTER
                    # The scaling of the bits outside of the IF-statement will
                    # then transform these to
                    # low : 0b01xxx => 0b00xxx => 0b0xxx0
                    # high: 0b10xxx => 0b01xxx => 0b1xxx1
                    # The overall effect is to get rid of the second largest
                    # bit. We don't know the value of this removed bit is untill
                    # the first bit converges to a value. Once the first value
                    # converges and we yield it, we must follow with an opposite
                    # bit. The number of opposite bits are now incremented.
                    bit_queue += 1
                else:
                    break  # Skip the bit shifting below the IF-statement

                # In all three cases above, we scale up bits by shifting every
                # bit to the left, then adding a 0 to `low` and a 1 to `high`.
                # Here is an example:
                # low : 0b00110 => 0b01100
                # high: 0b01100 => 0b11001
                low = 2 * low
                high = 2 * high + 1
                if self.verbose > 0:
                    print("  New values for high and low")
                    self._print_state(low, high, prefix="   ")

        # Check that the last symbol was the End Of Message (EOM) symbol
        if symbol != self.EOM:
            raise ValueError("Last symbol must be {repr(self.EOM)}, got {repr(symbol)}")

        # Finish encoding. Since low < HALF, we resolve ambiguity by yielding
        # bits [0, 1] if low < FIRST_QUARTER, else [1, 0].
        assert low < self.HALF
        bit_queue += 1
        yield from bit_queue.bit_plus_follow(int(low >= self.FIRST_QUARTER))

    def decode(self, iterable):
        """Decode an iterable of bits (0/1), yielding symbols.

        Examples
        --------
        >>> bits = [1, 0, 0, 1, 1, 0, 1]
        >>> frequencies = {'A': 5, 'B': 2, '<EOM>': 1}
        >>> encoder = ArithmeticEncoder(frequencies=frequencies)
        >>> list(encoder.decode(bits))
        ['A', 'B', '<EOM>']
        """
        if self.verbose:
            print("------------------------ DECODING ------------------------")

        # Set up low, current value and high
        low = 0
        value = 0
        high = self.TOP_VALUE

        # Consume the first `self.bits` into the `value` variable.
        # For instance, if iterable = [0, 1, 0, 1] and self.bits = 6,
        # then value = 0b010100 after this step
        iterable = enumerate(itertools.chain(iter(iterable), itertools.repeat(0)), 1)
        first_bits = itertools.islice(iterable, self.bits)
        for i, input_bit in first_bits:
            value = (value << 1) + input_bit

        if self.verbose:
            print(f"Consumed the initial {i} bits: 0b{value:0{self.bits}b} ")

        # General loop
        while True:
            if self.verbose:
                print("Current state:")
                self._print_state(low, high, value, prefix=" ", end="\n")

            # Current range and current scaled value
            range_ = high - low + 1
            scaled_value = ((value - low + 1) * self.total_count - 1) / range_
            symbol = search_ranges(scaled_value, self.ranges)
            yield symbol

            # Scale high and low. This mimicks (reverses) the encoder process
            symbol_low, symbol_high = self.ranges[symbol]
            high = low + int(range_ * symbol_high / self.total_count) - 1
            low = low + int(range_ * symbol_low / self.total_count)

            if self.verbose:
                print(f"After yielding symbol '{symbol}' and scaling:")
                self._print_state(low, high, value, prefix=" ", end="\n\n")

            # The symbol was the End Of Message (EOM) symbol and we are done.
            if symbol == self.EOM:
                break

            while True:
                if high < self.HALF:
                    # All of `high`, `low` and `value` have 0 as the first bit.
                    if self.verbose > 0:
                        print("  Range in lower half - both start with 0")
                    pass
                elif low >= self.HALF:
                    # All of `high`, `low` and `value` have 1 as the first bit.
                    if self.verbose > 0:
                        print("  Range in upper half - both start with 1")
                    value -= self.HALF
                    low -= self.HALF
                    high -= self.HALF
                elif low >= self.FIRST_QUARTER and high < self.THIRD_QUARTER:
                    # Low is in the `second` quarter and `high` is in the third.
                    if self.verbose > 0:
                        print("  Range in middle half - first 2 bits are opposite")

                    value -= self.FIRST_QUARTER
                    low -= self.FIRST_QUARTER
                    high -= self.FIRST_QUARTER
                else:
                    break

                if self.verbose > 0:
                    self._print_state(low, high, value, prefix="   ", end="\n")

                # Shift all bits one to the left, add 0 to low and 1 to high.
                # From the input bit stream (iterable) we read the next bit,
                # and default to 0 if the generator is exhausted.
                low = 2 * low
                high = 2 * high + 1
                i, input_bit = next(iterable, 1)
                value = 2 * value + input_bit
                assert low <= value <= high

                if self.verbose > 0:
                    print(f"  Consumed bit {i}: {input_bit}")
                    self._print_state(low, high, value, prefix="   ", end="\n\n")


if __name__ == "__main__":
    
    # An example
    message = ["B", "A", "A", "A", "<EOM>"]
    frequencies = {"A": 20, "B": 28, "<EOM>": 1}
    encoder = ArithmeticEncoder(frequencies=frequencies, bits=8, verbose=1)
    bits = list(encoder.encode(message))
    decoded = list(encoder.decode(bits))
    assert decoded == message


if __name__ == "__main__":
    import doctest

    doctest.testmod()
