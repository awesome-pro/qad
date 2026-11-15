# Generation-quality diagnostics

- model `Qwen/Qwen3-0.6B` @ `c1899de289a04d12100db370d81485cdf75e47ca`
- 8 fixed prompts, T=0.6, top_p=0.95, top_k=20, max_new_tokens=256, seed=42
- git `48b3b6293abb`

| config | mean tokens | repetition(4g) | premature EOS | hit cap | empty | EOS rate |
|---|---:|---:|---:|---:|---:|---:|
| w4a16_g128 | 256 | 0.050 | 0/8 | 8/8 | 0/8 | 0.0% |

## Examples

### w4a16_g128
- prompt: 'Generate an executable Python function generated from the given prompt. The function should take stdin as input and print the output. Simply call the function a'
  - "<think>\nOkay, let's try to figure out how to solve this problem. The task is to determine if the Chef, head server, and sous-chef can communicate with each other using the transceivers, even if two are too far apart. \n\nFirst, I need to understand the problem. The key points are:\n\n- Each person has two transceivers. But they can communicate directly if they are within R meters. If they are too far apart, they need another one to act as an intermediate. But how do we check if such a scenario is po"
- prompt: 'Generate an executable Python function generated from the given prompt. The function should take stdin as input and print the output. Simply call the function a'
  - "<think>\nOkay, let's see. The task is to write a Python function that reads input numbers A and B, computes their GCD, and prints the result for each pair. The input has two lines, and for each line, we need to output the GCD.\n\nFirst, I remember that the Euclidean algorithm is efficient, but when dealing with very large integers (like up to 250 digits), we can't just use the built-in GCD function because that's not efficient. Wait, but in Python, the built-in GCD function is implemented with the "

