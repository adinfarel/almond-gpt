Almond-GPT is an end-to-end, educational GPT project built entirely from scratch using PyTorch.
The goal of this project is to deeply understand how a GPT-style language model works — from tokenization and training a decoder-only transformer, all the way to serving and monitoring the model as a real system.

Unlike production-grade LLMs such as ChatGPT or LLaMA, Almond-GPT does not use RLHF, massive datasets, or pretrained weights.
Instead, it focuses on first principles: pure next-token prediction trained on small, carefully structured text data.

The model is trained on question–answer–style sequences, allowing it to appear to answer questions by learning how text continues — not by explicit reasoning or alignment mechanisms. This makes Almond-GPT ideal for learning how conversational behavior emerges naturally from sequence modeling.
