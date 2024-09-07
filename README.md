# arithmetic-coding

Python implementation of arithmetic encoding/decoding for lossless data compression.

- I wrote the code to understand the algorithm after reading the classic 1987 paper "Arithmetic coding for data compression."
- My focus was a clean, correct and modern Python implementation - not a state-of-the-art implementation.
- Python is too slow and too high level to make this code production ready.
- There are thousands of parametrized tests verifying the correctness of this implementation.

## Example

Here's an example showing how to use the `ArithmeticEncoder`.

```pycon
>>> from arithmetic_coding import ArithmeticEncoder
>>> message = ['A', 'B', 'B', 'B', '<EOM>']
>>> frequencies = {'A': 1, 'B': 3, '<EOM>': 1}
>>> encoder = ArithmeticEncoder(frequencies=frequencies)
>>> bits = list(encoder.encode(message))
>>> bits
[0, 1, 0, 1, 1, 0, 0, 1]
>>> list(encoder.decode(bits))
['A', 'B', 'B', 'B', '<EOM>']

```

## References

The two main references that I used were:

- Ian H. Witten, Radford M. Neal, and John G. Cleary. 1987. Arithmetic coding for data compression. Commun. ACM 30, 6 (June 1987), 520–540. [https://doi.org/10.1145/214762.214771](https://doi.org/10.1145/214762.214771)
- The 2014 blog post [Data Compression With Arithmetic Coding](https://marknelson.us/posts/2014/10/19/data-compression-with-arithmetic-coding.html).

Other useful references to deep-dive into the material:

- [github.com/nayuki/Reference-arithmetic-coding](https://github.com/nayuki/Reference-arithmetic-coding)
