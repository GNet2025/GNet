# G-Net

This repository contains the code and experiments for our paper **"G-Net: A Provably Easy Construction of High-Accuracy Random Binary Neural Networks"**, which introduces a two-step approach using GNet and its Embedded High-Dimensional Representation EHDG-Net.

---

## 📁 Project Structure

- `Quick Demo/Step1_GNet_Training.ipynb`: Trains the base GNet model on MNIST.
- `Quick Demo/Step2_EHDGNet.ipynb`: Loads the trained GNet and performs inference using its EHD representation.
- `requirements.txt`: Lists all required Python packages.
- `Comparison With Other HDC Methods/`: Contains the experiments for comparing EHDGNet with other Hyperdimensional Computing Methods.
- `RASU vs. TASU/`: Contains the experiments for comparing Gaussian and Rademacher RASU with Gaussian and Rademacher TASU frameworks in EHDGNet.
- `Robustness Experiments`: Contains the experiments for checking the robustness of RASU vs TASU and a comparison with other HDC methods under bit flip perturbations from 0% to 50%. 

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
