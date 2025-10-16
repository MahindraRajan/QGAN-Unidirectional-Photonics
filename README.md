# QGAN Unidirectional Photonics

**Quantum-Enhanced Inverse Design of Angle-Robust Metasurfaces**

LaSt-QGAN (Latent-Space Quantum Generative Adversarial Network) is a **hybrid quantum-classical architecture** for inverse design of all-dielectric metasurfaces that achieve **unidirectional, angle-independent light transmission** for advanced photovoltaic applications.

---

## 🌞 Overview

Metasurface-enabled photovoltaics offer powerful ways to control light—enhancing trapping, spectral shaping, and directionality. However, achieving **stable unidirectional transmission independent of incidence angle** has been a major challenge.

LaSt-QGAN bridges this gap by combining a **Variational Autoencoder (VAE)** with a **Quantum GAN (QGAN)** to directly generate metasurface patterns from desired far-field profiles. This approach drastically reduces data requirements and computational cost while maintaining optical accuracy and manufacturability.

---

## ⚙️ Key Features

- **Unidirectional Transmission:** Robust across −60° to 60° incidence angles  
- **Data Efficiency:** Achieves results with only 500 training samples (40× fewer than classical GANs)  
- **Fast Training:** ~2.5 hours on a single Tesla A100 GPU  
- **Hybrid Quantum-Classical Architecture:** Combines VAE encoding with quantum generator circuits  
- **Real-Material Mapping:** Integrates refractive index database for manufacturable designs (<10⁻⁴ MSE)  
- **Performance Gain:** 95% boost in simulated Perovskite solar-cell efficiency  

---

## 🧠 Architecture


The **VAE** compresses high dimensional metasurface image into low dimensional latent space. The **QGAN** generator explores this space using quantum circuits to produce candidate metasurface geometries that satisfy far-field design targets. A **classical discriminator** refines convergence, ensuring physical realizability.

---

## 📊 Results

- Generated **bow-tie all-dielectric metasurfaces** exhibiting angle-independent transmission.  
- Demonstrated **robust efficiency** under wide angular variation.  
- Achieved **fabrication-compatible** designs via material lookup mapping.  

---

## 🧩 Applications

- High-efficiency Perovskite and Si solar cells  
- Quantum-assisted nanophotonic inverse design  
- Angle-robust optical coatings and sensors

## 📘 Citation

If you use **LaSt-QGAN** in your work, please cite:

Sreeraj Rajan Warrier, Jayasri Dontabhaktuni. *Hybrid Quantum Generative Adversarial Networks To Inverse Design Metasurfaces For Incident Angle-Independent Unidirectional Transmission*. arXiv:2507.03518 [physics.optics], 2025.
### BibTeX

```bibtex
@article{WarrierDontabhaktuni2025,
  title        = {Hybrid Quantum Generative Adversarial Networks To Inverse Design Metasurfaces For Incident Angle-Independent Unidirectional Transmission},
  author       = {Warrier, Sreeraj Rajan and Dontabhaktuni, Jayasri},
  year         = {2025},
  eprint       = {2507.03518},
  archivePrefix = {arXiv},
  primaryClass = {physics.optics}
}
```
