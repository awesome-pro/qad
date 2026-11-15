# Generation-quality diagnostics

- model `Qwen/Qwen3-0.6B` @ `c1899de289a04d12100db370d81485cdf75e47ca`
- 8 fixed prompts, T=0.6, top_p=0.95, top_k=20, max_new_tokens=256, seed=42
- git `48b3b6293abb`

| config | mean tokens | repetition(4g) | premature EOS | hit cap | empty | EOS rate |
|---|---:|---:|---:|---:|---:|---:|
| w4a16_g128 | 256 | 0.053 | 0/8 | 8/8 | 0/8 | 0.0% |

## Examples

### w4a16_g128
- prompt: 'Generate an executable Python function generated from the given prompt. The function should take stdin as input and print the output. Simply call the function a'
  - "<think>\nOkay, let's try to figure out how to solve this problem. The task is to determine if the Chef, head server, and sous-chef can communicate with each other using the transceivers, even if they are too far apart. The key here is understanding the communication rules and then checking if the given positions satisfy the necessary conditions.\n\nFirst, let's restate the problem. The three people are at positions (X1, Y1), (X2, Y2), and (X3, Y3). They can communicate directly if their positions a"
- prompt: 'Generate an executable Python function generated from the given prompt. The function should take stdin as input and print the output. Simply call the function a'
  - "<think>\nOkay, let's see. The problem is to write a Python function that takes input lines of numbers A and B, computes their GCD, and prints the result. The function should be called after defining it. The function given is the Euclidean algorithm. \n\nFirst, I need to understand the problem. The task is to create an efficient code that can handle cases where A and B can be up to 40000 and B is up to 10^250, which is a huge number. So, the straightforward approach of using the Euclidean algorithm "

