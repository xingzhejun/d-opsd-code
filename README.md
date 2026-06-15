<div align="center">
    <h1>Learning from the Self-future: On-policy Self-distillation for dLLMs</h1>
    <p>We introduce <strong>d-OPSD</strong>, the first OPSD framework tailored for dLLMs</p>
</div>


<div align="center">
  <hr width="100%">
</div>

**Updates:**

* 15-06-2026: We released d-OPSD code.
<!-- * 04-11-2025: We released [our paper](https://dllm-reasoning.github.io/media/preprint.pdf) and [project page](https://dllm-reasoning.github.io). Additionally, the SFT code was open-sourced. -->

<div align="center">
  <hr width="100%">
</div>


## d-OPSD Environment

The environment configuration of d-OPSD is almost the same as the RLVR baseline [diffu-GRPO](https://github.com/dllm-reasoning/d1). However, there are some minor but important differences.

To set up the environment, first run (pay attention to the **trl version**):
```
cd d-opsd-code
conda env create -f env.yml
conda activate dOPSD
```

**Second, very important**, please go to your environment `/path/to/env/trl/trainer/grpo_trainer.py`, and modify line 424 to the followings:
```
# modify
# possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
possible_values = [n_gen for n_gen in range(1, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
```

Or you can choose to simply replace the original `/path/to/env/trl/trainer/grpo_trainer.py` with what we offered in this repo.

Finally, we give the real environment configuration we used for all experiments in `used-env.txt`, for debugging convenience. This configuration works like an alarm for Python 3.10 and CUDA 12.9, with A100 / H100 / B200 GPUs.


## d-OPSD Training

All training code is inside the `d-opsd` directory. To reproduce the training, run:
```
cd d-opsd-code
bash run/gsm/opsd.sh
bash run/math/opsd.sh
bash run/countdown/opsd.sh
bash run/sudoku/opsd.sh
```

Note: **Very important**, for A100 / H100 GPUs, the `BATCH_DIVIDE` in the script should be set to 8 to prevent OOM. For B200, the existing setting `BATCH_DIVIDE=4` works well.


## d-OPSD Evaluation

All evaluation code is inside the `eval` directory. First replace the checkpoint path in the scripts with your own, and run:
```
cd d-opsd-code
bash run/gsm/opsd.sh
bash run/math/opsd.sh
bash run/countdown/opsd.sh
bash run/sudoku/opsd.sh
```

This evaluation saves the generations. Second, replace the generation directory in `eval/parse_and_get_acc.py` with your owns, and run the following to obtain the accuracy:
```
cd d-opsd-code/eval
python parse_and_get_acc.py
```