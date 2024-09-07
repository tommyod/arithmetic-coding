# -*- coding: utf-8 -*-
"""
Tests for arithmetic encoder.
"""

import pytest
import random
from arithmetic_coding import ArithmeticEncoder


class TestArithmeticEncoder:
    def test_encoding_on_simple_example_1(self):
        message = ["A", "<EOM>"]
        frequencies = {"A": 2, "B": 2, "C": 2, "<EOM>": 1}
        encoder = ArithmeticEncoder(frequencies=frequencies)
        bits = list(encoder.encode(message))
        assert bits == [0, 0, 1, 0, 1, 0]

    def test_encoding_on_simple_example_2(self):
        message = ["C", "<EOM>"]
        frequencies = {"A": 2, "B": 2, "C": 2, "<EOM>": 1}
        encoder = ArithmeticEncoder(frequencies=frequencies)
        bits = list(encoder.encode(message))
        assert bits == [1, 0, 1, 1, 1, 0]

    def test_on_very_infrequent_symbol(self):
        message = ["A"] * 1000 + ["B"] + ["<EOM>"]
        frequencies = {"A": 1000, "B": 1, "<EOM>": 1}
        encoder = ArithmeticEncoder(frequencies=frequencies, bits=12)
        bits = list(encoder.encode(message))
        assert len(bits) == 24  # Encodes ~1000 symbols in 24 bits only
        assert list(encoder.decode(bits)) == message

    @pytest.mark.parametrize("seed", range(99))
    @pytest.mark.parametrize("bits", [6, 7, 8, 12, 16])
    def test_encoding_and_decoding_random_messages(self, seed, bits):
        random_generator = random.Random(seed * 1000 + bits)
        k = random_generator.randint(0, 99)

        # Generate a random message of random length
        message = random_generator.choices(["A", "B", "C"], k=k) + ["<EOM>"]
        frequencies = {"A": 5, "B": 2, "C": 1, "<EOM>": 1}

        # Create encoder
        encoder = ArithmeticEncoder(frequencies=frequencies, bits=bits)

        # Encode to bits
        bits = list(encoder.encode(message))

        # Decode from bits to message
        decoded = list(encoder.decode(bits))

        assert decoded == message


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
