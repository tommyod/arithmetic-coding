# arithmetic-coding

Python implementation of arithmetic encoding/decoding for lossless data compression.

[**Read a blog post about this implementation here.**](https://tommyodland.com/articles/2024/arithmetic-coding-in-python/)

## Overview

This project provides a clean, correct, and modern Python implementation of the arithmetic coding algorithm. 
It is based on the classic 1987 paper "[Arithmetic coding for data compression](https://dl.acm.org/doi/10.1145/214762.214771)" by Witten, Neal, and Cleary.

Key features:

- Clear and readable Python code
- Focused on correctness and educational value
- Extensive parametrized tests for verification

**Note**: This implementation prioritizes clarity and correctness over performance.
For production use, consider more optimized implementations in lower-level languages.

## Example

Here's an example showing how to use the `ArithmeticEncoder`.

```pycon
>>> from arithmetic_coding import ArithmeticEncoder
>>> message = ['A', 'B', 'B', 'B', '<EOM>']
>>> frequencies = {'A': 1, 'B': 3, '<EOM>': 1}
>>> encoder = ArithmeticEncoder(frequencies=frequencies)
>>> bits = list(encoder.encode(message))
>>> bits
[0, 1, 0, 1, 1, 1, 0, 0]
>>> list(encoder.decode(bits))
['A', 'B', 'B', 'B', '<EOM>']

```

## Testing

Run the test suite to verify the implementation.

```bash
pytest . --doctest-modules
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
Stay within the scope of the project: clean, readable Python implementation, no low-level optimizations.

## References

The two main references that I used were:

- Arithmetic coding for data compression. Ian H. Witten, Radford M. Neal, and John G. Cleary. 1987. Commun. ACM 30, 6 (June 1987), 520–540. [https://doi.org/10.1145/214762.214771](https://doi.org/10.1145/214762.214771)
- The 2014 blog post [Data Compression With Arithmetic Coding](https://marknelson.us/posts/2014/10/19/data-compression-with-arithmetic-coding.html).

Other useful references to deep-dive into the material:

- [github.com/nayuki/Reference-arithmetic-coding](https://github.com/nayuki/Reference-arithmetic-coding)
- The book [Information Theory, Inference and Learning Algorithms](https://www.amazon.com/Information-Theory-Inference-Learning-Algorithms/dp/0521642981) by David J. C. MacKay
- [The Data Compression Book](https://marknelson.us/pages/tdcb) by Mark Nelson and Jean-loup Gailly
