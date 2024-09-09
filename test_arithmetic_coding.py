# -*- coding: utf-8 -*-
"""
Tests for arithmetic encoder.
"""

import pytest
import random
from arithmetic_coding import ArithmeticEncoder
import string


class TestArithmeticEncoder:
    def test_encoding_only_EOM(self):
        message = ["<EOM>"]
        frequencies = {"<EOM>": 1}
        encoder = ArithmeticEncoder(frequencies=frequencies, bits=6)
        bits = list(encoder.encode(message))
        decoded = list(encoder.decode(bits))
        assert decoded == message

    def test_encoding_only_EOM_with_other_frequencies(self):
        message = ["<EOM>"]
        frequencies = {"<EOM>": 1, "A": 3, "B": 5}
        encoder = ArithmeticEncoder(frequencies=frequencies, bits=6)
        bits = list(encoder.encode(message))
        decoded = list(encoder.decode(bits))
        assert decoded == message

    def test_encoding_on_simple_example_1(self):
        message = ["A", "<EOM>"]
        frequencies = {"A": 2, "B": 2, "C": 2, "<EOM>": 1}
        encoder = ArithmeticEncoder(frequencies=frequencies, bits=6)
        bits = list(encoder.encode(message))
        assert bits == [0, 0, 1, 0, 1, 0]

    def test_encoding_on_simple_example_2(self):
        message = ["C", "<EOM>"]
        frequencies = {"A": 2, "B": 2, "C": 2, "<EOM>": 1}
        encoder = ArithmeticEncoder(frequencies=frequencies, bits=6)
        bits = list(encoder.encode(message))
        assert bits == [1, 0, 1, 1, 1, 0]

    def test_on_very_infrequent_symbol(self):
        message = ["A"] * 1000 + ["B"] + ["<EOM>"]
        frequencies = {"A": 1000, "B": 1, "<EOM>": 1}
        encoder = ArithmeticEncoder(frequencies=frequencies, bits=12)
        bits = list(encoder.encode(message))
        assert len(bits) == 24  # Encodes ~1000 symbols in 24 bits only
        assert list(encoder.decode(bits)) == message

    def test_on_regression_case1(self):
        message = ["C", "C", "<EOM>"]
        frequencies = {"A": 1, "B": 13, "C": 40, "<EOM>": 1}
        encoder = ArithmeticEncoder(frequencies=frequencies, bits=8)
        bits = list(encoder.encode(message))
        decoded = list(encoder.decode(bits))
        assert decoded == message

    def test_on_regression_case2(self):
        message = ["B", "C", "<EOM>"]
        frequencies = {"A": 6, "B": 7, "C": 6, "<EOM>": 1}
        encoder = ArithmeticEncoder(frequencies=frequencies, bits=7)
        bits = list(encoder.encode(message))
        decoded = list(encoder.decode(bits))
        assert decoded == message

    def test_on_regression_case3(self):
        message = ["C", "B", "B", "B", "C", "<EOM>"]
        frequencies = {"A": 8, "B": 6, "C": 7, "<EOM>": 1}
        encoder = ArithmeticEncoder(frequencies=frequencies, bits=7)
        bits = list(encoder.encode(message))
        decoded = list(encoder.decode(bits))
        assert decoded == message

    def test_on_regression_case4(self):
        message = ["B", "A", "<EOM>"]
        frequencies = {"A": 20, "B": 28, "<EOM>": 1}
        encoder = ArithmeticEncoder(frequencies=frequencies, bits=8)
        bits = list(encoder.encode(message))
        decoded = list(encoder.decode(bits))
        assert decoded == message

    def test_static_model_on_a_long_message(self):
        rng = random.Random(42)

        message = rng.choices(string.ascii_letters, k=10**5) + ["<EOM>"]
        frequencies = {letter: rng.randint(1, 99) for letter in string.ascii_letters}
        frequencies["<EOM>"] = 1

        encoder = ArithmeticEncoder(frequencies=frequencies, bits=14)
        decoded = list(encoder.decode(encoder.encode(message)))
        assert decoded == message

    def test_dynamic_model_on_a_long_message(self):
        random_generator = random.Random(42)

        message = random_generator.choices(string.ascii_letters, k=10**5) + ["<EOM>"]
        frequencies = list(set(message))

        encoder = ArithmeticEncoder(frequencies=frequencies, bits=20)

        # Test that encoder/decoder works when we FIRST encode, THEN decode
        bits = list(encoder.encode(message))
        decoded = list(encoder.decode(bits))
        assert decoded == message

        # Test that they work in lockstep too
        decoded = list(encoder.decode(encoder.encode(message)))
        assert decoded == message

    def test_that_different_number_of_bits_in_buffer_can_give_different_output_bits(
        self
    ):
        message = ["A", "<EOM>"]
        frequencies = {"A": 6, "<EOM>": 1}
        enc_6bit = ArithmeticEncoder(frequencies=frequencies, bits=6)
        enc_7bit = ArithmeticEncoder(frequencies=frequencies, bits=7)

        # Different bits in the encoder can yield different sequences of bits
        assert list(enc_6bit.encode(message)) == [0, 0, 1, 0, 1]
        assert list(enc_7bit.encode(message)) == [0, 0, 1, 1]

        # But they decode to the same value
        assert list(enc_6bit.decode(enc_6bit.encode(message))) == message
        assert list(enc_7bit.decode(enc_7bit.encode(message))) == message

    @pytest.mark.parametrize("seed", range(99))
    @pytest.mark.parametrize("bits", [4, 6, 8, 7, 12, 16, 32])
    @pytest.mark.parametrize("dynamic", [True, False])
    def test_encoding_and_decoding_random_messages(self, seed, bits, dynamic):
        random_generator = random.Random(seed + bits * 1_000)
        k = 2 ** random_generator.randint(1, 14)

        # Generate a random message of random length
        message = random_generator.choices(["A", "B", "C"], k=k) + ["<EOM>"]
        freq_A, freq_B, freq_C = random_generator.choices(range(1, 99), k=3)

        # Whether or not to test using a dynamic probability model
        if dynamic:
            frequencies = list(set(message))
        else:
            frequencies = {"A": freq_A, "B": freq_B, "C": freq_C, "<EOM>": 1}

        # Create encoder, but skip test if there are not enough bits
        try:
            encoder = ArithmeticEncoder(frequencies=frequencies, bits=bits)
            bits = list(encoder.encode(message))
        except Exception as exception:
            if "Insufficient precision" in exception.args[0]:
                return

        # Decode from bits to message
        decoded = list(encoder.decode(bits))

        assert decoded == message


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
