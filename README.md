# arithmetic-coding
Python implementation of arithmetic encoding/decoding for lossless data compression.









```pycon
>>> from arithmetic_coding import ArithmeticEncoder
>>> message = ['A', 'B', 'B', 'B', '<EOM>']
>>> frequencies = {'A': 1, 'B':3, '<EOM>':1}
>>> encoder = ArithmeticEncoder(frequencies=frequencies)
>>> bits = list(encoder.encode(message))
>>> bits
[0, 1, 0, 1, 1, 0, 0, 1]
>>> list(encoder.decode(bits))
['A', 'B', 'B', 'B', '<EOM>']

```
