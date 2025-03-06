Emotion Detection  

Overview  
This project is a real-time emotion detection system using OpenCV and TensorFlow. It captures video from a webcam, detects faces, and classifies emotions using a pre-trained deep learning model. The detected emotions are displayed on the video feed in real-time.  

Features  
- Detects faces using Haar Cascade Classifier  
- Classifies emotions into categories such as Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral  
- Displays real-time predictions on the video feed  
- Uses a trained deep learning model for accurate emotion recognition  

Installation  
1. Ensure Python is installed (Python 3.x recommended).  
2. Install the required dependencies using:  
  pip install opencv-python numpy tensorflow  
3. Place the pre-trained model (emotion_model.h5) in the specified directory.  
4. Run the script:  
  python emotion_detection.py  

Usage  
- The program captures video using the webcam.  
- It detects faces, processes them, and predicts emotions.  
- Press 'q' to exit the application.  

Dependencies  
- OpenCV for face detection and video processing  
- TensorFlow for loading and running the deep learning model  
- NumPy for numerical computations  

License  
This project is open-source and available under the MIT License.  

Author  
Developed by Sujeet Kumar  
Email: vishwakarma.sujeet1626@gmail.com