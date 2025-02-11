# Data Generator Using GReaT

## Overview
This project demonstrates how to generate synthetic data using **GReaT (Generative Realistic Tabular data)** with **distilgpt2**. The notebook leverages **be-great**, Pandas, and NumPy to create high-quality synthetic datasets suitable for various machine learning and research applications.

## Features
- **Synthetic Data Generation** using GReaT
- **Uses distilgpt2** as the language model
- **Configurable batch size and epochs** for fine-tuning
- **Ensures reproducibility** with fixed random seeds

## Installation
To run this notebook, install the required dependencies:
```bash
pip install be-great pandas numpy rich
```

## Usage
1. **Import dependencies**
   ```python
   import pandas as pd
   import random
   import numpy as np
   from rich import print
   from be_great import GReaT
   ```
   
2. **Set random seed for reproducibility**
   ```python
   random.seed(42)
   np.random.seed(42)
   ```
   
3. **Initialize GReaT Model**
   ```python
   model = GReaT(llm='distilgpt2', batch_size=32, epochs=5, save_steps=400000)
   ```
   
4. **Generate Data** (Replace `your_dataframe` with your dataset)
   ```python
   model.fit(your_dataframe)
   synthetic_data = model.sample(n_samples=5000)
   ```
   
## Output
The script generates **5,000 rows of synthetic data**, maintaining statistical properties similar to the original dataset.

## Notes
- Ensure `be-great` is installed before running.
- Adjust hyperparameters (`batch_size`, `epochs`) for optimal performance.

## License
This project is released under the MIT License.

