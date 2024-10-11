# LipTrans - Real-Time Lip Transcription

![Build Status](https://img.shields.io/badge/build-stable-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/Python-3.0%2B-purple)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)

LipTrans is an advanced real-time lip transcription tool that utilizes a sophisticated neural network to translate lip movements into text. Designed to improve accessibility, it directly maps mouth movements to sentences, providing a significant enhancement in live captioning accuracy and accessibility for the hearing impaired.

## Key Features
- Real-time lip transcription with impressive reductions in Word Error Rate (WER) and Character Error Rate (CER).
- Utilizes a combination of 3D CNN, MaxPooling, Bi-LSTM, and CTC Loss for robust spatial and temporal feature extraction.
- Achieves 4.5x better performance than human lipreading baselines.

## Technologies Used 🛠️
- **Python** (3.8+)
- **TensorFlow** (2.x)
- **OpenCV**
- **Flask** (for the web app)
- **Jupyter Notebook**

## Evaluation Metrics and Comparison 📈
- **Word Error Rate (WER):** Decreased by 50% to 0.5, with LipTrans achieving a 4.8% WER, which is 2.8x lower than the state-of-the-art on the GRID corpus, underscoring its significant improvement over baseline models.
- **Character Error Rate (CER):** Reduced by 90% to 0.04164, demonstrating superior accuracy in recognizing lip movements.
- **Word Accuracy:** Increased from 0% to 83.33% by epoch 15, reflecting a notable advancement over human lipreading capabilities, with LipTrans outperforming by a factor of 4.5x.
- **F1 Score:** Achieved 0.909 by epoch 15, indicating a highly refined ability to translate visual lip cues into accurate text.
- **BLEU Score:** Despite consistently low average scores (below 0.1), the model continues to improve in text similarity with each epoch.

## Dataset 📊
LipTrans utilizes the **GRID Corpus**, a widely used audiovisual dataset for lip reading and speech recognition research. The GRID Corpus is essential for training the model to recognize and interpret lip movements accurately.

### Key Features of the GRID Corpus:
- **Size:** Contains over 34,000 video recordings.
- **Participants:** 34 speakers (18 male and 16 female) reciting fixed vocabulary sentences.
- **Format:** Each video includes synchronized audio and video streams of the speaker’s mouth area.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/KavinAravindhan/LipTrans.git
   ```
2. Navigate to the project directory:
   ```bash
   cd LipTrans
   ```
3. Install required dependencies:
   ```bash
   pip3 install -r requirements.txt
   ```
4. Load the GRID Corpus data by running `data_load.ipynb` in Jupyter Notebook.

## File Details
- **init.ipynb:** Install all required libraries.
- **data_load.ipynb:** Download and load the GRID Corpus.
- **LipTrans_Initial_Setup.ipynb:** Data preprocessing and initial training.
- **single_run_file.ipynb:** Test run on a subset of data.
- **LipTrans_Train_Test-With_Metrics.ipynb:** Neural network design, training, and evaluation.
- **LipTrans_Epochs_Model_Analysis.ipynb:** Model analysis at different epochs.
- **LipTrans_Final.ipynb:** Comprehensive file for training, testing, and prediction.
- **app.py, static, and templates:** Web app files using Flask to upload video and output transcribed text.
- **model_checkpoints:** Contains checkpoint files for models trained at 50 and 100 epochs.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Links 🔗
- [Project Paper](https://github.com/KavinAravindhan/LipTrans/blob/master/paper/Real_Time_Lip_Transcription.pdf)
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs/python/tf/data)

## Team Acknowledgment 🙌

A special thanks to our amazing team for their dedication and hard work. Despite the challenges, their commitment to learning new technologies and collaborating effectively made this project a success.
