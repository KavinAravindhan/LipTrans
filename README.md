# LipTrans - Real-Time Lip Transcription

![Build Status](https://img.shields.io/badge/build-stable-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-yellow)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)

![Platform](https://img.shields.io/badge/platform-Android-yellow)
![Flutter](https://img.shields.io/badge/Flutter-Framework-blue?logo=flutter)
![Firebase](https://img.shields.io/badge/Firebase-Database-orange?logo=firebase)

LipTrans is an advanced real-time lip transcription tool that utilizes a sophisticated neural network to translate lip movements into text. Designed to improve accessibility, it directly maps mouth movements to sentences, providing a significant enhancement in live captioning accuracy and accessibility for the hearing impaired.

### Key Features 🌟
- Real-time lip transcription with impressive reductions in Word Error Rate (WER) and Character Error Rate (CER).
- Utilizes a combination of 3D CNN, MaxPooling, Bi-LSTM, and CTC Loss for robust spatial and temporal feature extraction.
- Achieves 4.5x better performance than human lipreading baselines.

### Technologies Used 🛠️
- **Python** (3.8+)
- **TensorFlow** (2.x)
- **OpenCV**
- **Flask** (for the web app)
- **Jupyter Notebook**

### Evaluation Metrics 📊
- **Word Error Rate (WER):** Reduced by 50% to 0.5
- **Character Error Rate (CER):** Reduced by 90% to 0.04164
- **Word Accuracy:** Improved from 0% to 83.33% by epoch 15
- **F1 Score:** Improved to 0.909 by epoch 15

### Installation 🖥️
1. Clone the repository:
   ```bash
   git clone https://github.com/username/LipTrans.git
   ```
2. Navigate to the project directory:
   ```bash
   cd LipTrans
   ```
3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Load the GRID Corpus data by running `data_load.ipynb` in Jupyter Notebook.

### File Details 📁
- **init.ipynb:** Install all required libraries.
- **data_load.ipynb:** Download and load the GRID Corpus.
- **LipTrans_Initial_Setup.ipynb:** Data preprocessing and initial training.
- **single_run_file.ipynb:** Test run on a subset of data.
- **LipTrans_Train_Test-With_Metrics.ipynb:** Neural network design, training, and evaluation.
- **LipTrans_Epochs_Model_Analysis.ipynb:** Model analysis at different epochs.
- **LipTrans_Final.ipynb:** Comprehensive file for training, testing, and prediction.
- **app.py, static, and templates:** Web app files using Flask to upload video and output transcribed text.
- **model_checkpoints:** Contains checkpoint files for models trained at various epochs.

### License 📄
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

### Application Images 🖼️
![App Interface](link_to_app_screenshot.png)
![Model Performance Graph](link_to_performance_graph.png)

### Links 🔗
- [Project Paper](link)
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs/python/tf/data)

### Acknowledgements 👥
Special thanks to the development team and all contributors for their valuable insights and dedication in making LipTrans a reality!

--- 

This template is ready for you to modify as needed! Let me know if you want any more customization.
