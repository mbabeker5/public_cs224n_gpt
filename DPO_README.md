# Improved Direct Preference Optimization (DPO) for Paraphrase Detection

This README explains how to use the improved DPO implementation for paraphrase detection. The implementation is designed to be fully compatible with the original code and autograder.

## Overview

Direct Preference Optimization (DPO) is an advanced training technique that can improve model performance by teaching the model to prefer "winning" examples over "losing" examples. Our implementation includes:

1. **Curriculum Learning**: Gradually transitions from standard cross-entropy loss to DPO loss
2. **Improved Sample Generation**: Creates challenging examples that help the model learn better distinctions
3. **Mixed Loss Function**: Combines standard training with DPO for more stable learning
4. **Pretraining Phase**: Establishes good fundamentals before applying DPO

## Files

- `run_improved_dpo.py`: Script to train a new model from scratch using improved DPO
- `finetune_with_improved_dpo.py`: Script to fine-tune an existing model using improved DPO

## Usage

### Training a New Model with Improved DPO

```bash
python run_improved_dpo.py --use_gpu --model_size gpt2-medium
```

Key parameters:
- `--epochs`: Number of DPO training epochs (default: 4)
- `--pretrain_epochs`: Number of standard training epochs before DPO (default: 2)
- `--dpo_strategy`: Strategy for generating losing samples ("heuristic" or "model_based")
- `--lr`: Learning rate (default: 1e-5)
- `--batch_size`: Batch size for training (default: 8)
- `--model_size`: Model size to use (default: "gpt2")

### Fine-tuning an Existing Model with Improved DPO

```bash
python finetune_with_improved_dpo.py --model_path <path_to_model> --use_gpu
```

Key parameters:
- `--model_path`: Path to the pre-trained model to fine-tune (required)
- `--epochs`: Number of DPO fine-tuning epochs (default: 2)
- `--dpo_strategy`: Strategy for generating losing samples ("heuristic" or "model_based")
- `--lr`: Learning rate (default: 5e-6, should be lower than original training)

## Compatibility with Autograder

Both scripts maintain compatibility with the autograder by:

1. Using the same output file paths as the original code
2. Maintaining the same model architecture and evaluation methods
3. Ensuring the model saves in the expected format

After running either script, you should find:
- The trained model file (e.g., `4-1e-5-8layers-para.pt`)
- Prediction files in the `predictions` directory:
  - `predictions/para-dev-output.csv`
  - `predictions/para-test-output.csv`

## Recommended Settings

For best results, we recommend:

1. **Training a new model**:
   ```bash
   python run_improved_dpo.py --use_gpu --model_size gpt2-medium --pretrain_epochs 3 --epochs 4 --dpo_strategy heuristic --lr 1e-5 --batch_size 8
   ```

2. **Fine-tuning an existing model**:
   ```bash
   python finetune_with_improved_dpo.py --model_path <your_best_model.pt> --use_gpu --epochs 2 --dpo_strategy heuristic --lr 5e-6
   ```

## Troubleshooting

If you encounter issues:

1. **Memory errors**: Reduce batch size or increase gradient accumulation steps
   ```bash
   python run_improved_dpo.py --use_gpu --batch_size 4 --gradient_accumulation_steps 8
   ```

2. **Low accuracy**: Try different DPO strategies or adjust the learning rate
   ```bash
   python run_improved_dpo.py --use_gpu --dpo_strategy model_based --lr 8e-6
   ```

3. **Compatibility issues**: Ensure you're using the scripts with the original code structure intact

## References

This implementation is based on the paper "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" by Rafailov et al. and the reference implementation at https://github.com/eric-mitchell/direct-preference-optimization. 