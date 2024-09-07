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
    for number in reversed(range(low, high+1)):
        print(f" 0b{number:0{bits}b} ({number})")




class ArithmeticEncoder:
    
    def __init__(self, frequencies, *, bits=6, verbose=0, EOM="<EOM>"):
        self.EOM = EOM
        assert self.EOM in frequencies.keys()
        assert all(isinstance(freq, int) for freq in frequencies.values())
        self.frequencies = frequencies
        self.bits = bits
        self.verbose=verbose
        
        # Build ranges from frequencies
        self.ranges = dict(ranges_from_frequencies(self.frequencies))
        self.total_count  = sum(self.frequencies.values())
        
        # The total range. Examples in comments are with 4 bits
        self.TOP_VALUE = (1 << self.bits) - 1 # 0b1111 = 15
        self.FIRST_QUARTER = (self.TOP_VALUE >> 2) + 1 # 0b0100 = 4
        self.HALF = self.FIRST_QUARTER * 2 # 0b1000 = 8
        self.THIRD_QUARTER = self.FIRST_QUARTER * 3 # 0b1100 = 12
        self.bits_to_follow = 0 # Counter
        
        if self.verbose > 0:
            print("Initialized with:")
            print(f" TOP_VALUE     = 0b{self.TOP_VALUE:0{self.bits}b} ({self.TOP_VALUE})")
            print(f" THIRD_QUARTER = 0b{self.THIRD_QUARTER:0{self.bits}b} ({self.THIRD_QUARTER})")
            print(f" HALF          = 0b{self.HALF:0{self.bits}b} ({self.HALF})")
            print(f" FIRST_QUARTER = 0b{self.FIRST_QUARTER:0{self.bits}b} ({self.FIRST_QUARTER})")
            print(f" total_count   = {self.total_count}")
        
        
    def print_state(self, low, high, prefix=" "):
        if self.verbose > 0:
            range_ = (high - low + 1)
            print(prefix + f"High value: 0b{high:0{self.bits}b} ({high})")
            print(prefix + f"Low value:  0b{low:0{self.bits}b} ({low})")
            print(prefix + f"Range: [{low}, {high + 1}) Width: {range_}")
        
    def bit_plus_follow(self, bit):
        yield bit
        for _ in range(self.bits_to_follow):
            yield int(not bit)
        self.bits_to_follow = 0 # Reset
        
        
    def mask_to_bits(self, integer):
        """Mask so only the lowest bits are kept. This is identical to modulus
        and keeps e.g. 4 bits means that mask_to_bits(0b10110) == 0b0110."""
        return integer & self.TOP_VALUE
    
    def decode(self, iterable):
        
        
        
        #iterable = iter(iterable)
        iterable = itertools.chain(iter(iterable), itertools.repeat(0))
        low = 0
        high = self.TOP_VALUE
        
        # Consume the first bits
        # TODO: consume extra dummy bits if loop does not go long enough
        value = 0
        first_bits = itertools.islice(iterable, self.bits)
        for i, input_bit in enumerate(first_bits, 1):
            if self.verbose:
                print(f"\nProcessing bit {i}: {input_bit}")
                print("-"*32)
                
            value = value * 2 + input_bit
            print(f"Value: 0b{value:0{self.bits}b} ({value})")
                
        # General loop
        while True:
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
            
            self.print_state(low, high,  " ")
            
            
            while True:
                if high < self.HALF:
                    print("In bottom half of interval")
                    pass
                elif low>= self.HALF:
                    print("In top half of interval")
                    value -= self.HALF 
                    low -= self.HALF
                    high -= self.HALF
                elif (low >= self.FIRST_QUARTER and high < self.THIRD_QUARTER):
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
        
        
        # Initial low and high values for the range [low, high)
        low = 0
        high = self.TOP_VALUE
        
        assert self.total_count <= ((high + 1) / 4) + 1 # Equation on page 533
        
        for i, symbol in enumerate(iterable, 1):
            if self.verbose > 0:
                print(f"\nProcessing symbol {i}: {symbol}")
                print("-"*32)

        
            range_ = (high - low + 1)
            assert range_ >= self.total_count, "Not enough precision"
            
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
                    yield from self.bit_plus_follow(bit=0)
                elif low >= self.HALF:
                    if self.verbose > 0:
                        print(" Range in upper half")
                    yield from self.bit_plus_follow(bit=1)
                    low -= self.HALF
                    high -= self.HALF
                elif (low >= self.FIRST_QUARTER and high < self.THIRD_QUARTER):
                    if self.verbose > 0:
                        print(" Range in middle half")
                    self.bits_to_follow += 1
                    low -= self.FIRST_QUARTER
                    high -= self.FIRST_QUARTER
                else:
                    break 
                
                # Scale up bits
                low = 2 * low
                high = 2 * high + 1
                if self.verbose > 0:
                    self.print_state(low, high, " ")
                    print()
                
        # Check that the last symbol was EOM
        assert symbol == self.EOM
        
        # Finish encoding
        self.bits_to_follow += 1
        
        if low < self.FIRST_QUARTER:
            yield from self.bit_plus_follow(bit=0)
        else:
            yield from self.bit_plus_follow(bit=1)    
    
if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "--doctest-modules"])
    
    
if __name__ == "__main__":
    
    import random
    

    message = ["A", "B", "B", "B", "A", "<EOM>"]
    message = random.choices(["A", "B"], k=3) + ["<EOM>"]
    frequencies = {"A": 5, "B":2, "<EOM>":1}
    
    encoder = ArithmeticEncoder(frequencies=frequencies)
    
    bits = list(encoder.encode(message))
    
    print("Final output:", bits)
    
    for symbol in encoder.decode(bits):
        print(symbol)
        
    decoded = list(encoder.decode(bits))
    print(decoded)
    
    assert decoded == message
    
    