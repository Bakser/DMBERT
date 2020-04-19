# DMBERT
A temporary repo to share the DMBERT code for Event Detection

This repo includes a recent implementation of the DMBERT model in [the NAACL work](https://github.com/thunlp/Adv-ED) with the support of the newest [transformers](https://github.com/huggingface/transformers) library.

Just a temporary repo, not well-documented, will be merged into some to-be-announced works in the future.

Most of the codes are changed from the examples in [transformers](https://github.com/huggingface/transformers), you can refer to that repo for most of your questions.

# Requirments

transformers

torch >= 1.2.0

# Usage

To run this code, you need to:

- preprocess your ACE2005 dataset as the json format specified in [the HMEAE repo](https://github.com/thunlp/HMEAE)
- change the parameters in ```run.sh`` as your environment needs and run it

# Cite
If the codes help you, please cite our paper:

**Adversarial Training for Weakly Supervised Event Detection.** *Xiaozhi Wang, Xu Han, Zhiyuan Liu, Maosong Sun, Peng Li.* NAACL-HLT 2019.

