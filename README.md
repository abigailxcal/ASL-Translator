# ASL Alphabet Detection using Computer Vision

**Project Team Members**: 
- **Abigail Calderon**
- **Matthew Manning**
- **Rane Dy**

## 📖 Purpose  

The ASL Alphabet Detection project aims to improve accessibility and communication for individuals in the Deaf and Hard of Hearing community by enabling real-time recognition of American Sign Language (ASL) alphabets using computer vision techniques, as well as apending these letters to create sentences.
This project focused on:  
- Preprocessing Dataset from Kaggle: https://www.kaggle.com/datasets/lexset/synthetic-asl-alphabet
- Extracting hand keypoints from gesture images using Mediapipe
- Converting keypoint data into a structured dataset suitable for training
- Designing and training a neural network model for alphabet classification
- Utilizing the webcam and integrating the model into an interactive application to display recognized letters and form words/sentences
 
---
## 🔨 Solution Overview 
We first needed to understand the ASL alphabet hand gestures! Furthermore:
- Real-Time Gesture Recognition: We utilized Mediapipe’s advanced hand tracking technology to enable real-time recognition of ASL gestures through a device’s camera. This allowed for accurate and responsive tracking of hand movements and positions, supporting seamless gesture interpretation. 

- Text Translation: To translate recognized ASL gestures into readable text, we employed TensorFlow to train a robust machine learning model. This model was designed to accurately interpret a wide range of hand signs, ensuring reliable and high-precision gesture-to-text conversion. 

- Accessible Interface: We utilized customtkinter to create a user-friendly interface!
---

## 📈 Results and Key Findings  

### Dataset Results:
-**Samples**: 20,000 Samples

### Key Insights:  
- Accuracy Score: 98%
---

## 🚀 Potential Next Steps  

- In the future, we would love to improve the model to detect ASL phrases and facial expressions

---

## 💻 Installation  

### Prerequisites:  
- Python 3.12.2 

### Installation Steps:  
1. Clone the repository:  
   ```bash  
   git clone https://github.com/abigailxcal/ASL-Translator.git
2. #Install modules
pip install -r requirements.txt

3. #Start the application
python app.py
  
---

 
