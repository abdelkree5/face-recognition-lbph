<h1 align="center">🧑‍🤝‍🧑 Face Recognition (LBPH) — OpenCV</h1>
<h3 align="center">Capture → Train → Live Recognition (LBPH)</h3>
<hr/>

<h2>Overview</h2>
<p>This repository contains a simple production-style face recognition pipeline using OpenCV's LBPH face recognizer. The system includes:</p>
<ul>
  <li>Image capture utility (from webcam)</li>
  <li>Training script that builds an LBPH model</li>
  <li>Real-time recognition script using the trained model</li>
</ul>

<h2>Files</h2>
<ul>
  <li><b>generateTrainingData.py</b> — capture face images from webcam into <code>trainingData/</code>. :contentReference[oaicite:6]{index=6}</li>
  <li><b>trainModel.py</b> — train LBPH recognizer and save <code>face_model.yml</code> and <code>names.npy</code>. :contentReference[oaicite:7]{index=7}</li>
  <li><b>testModel.py</b> — run webcam recognition using the saved model. :contentReference[oaicite:8]{index=8}</li>
  <li><b>cascades/</b> — haarcascade XML files (e.g., <code>haarcascade_frontalface_default.xml</code>).</li>
</ul>

<h2>Quick Start</h2>
<ol>
  <li>Install dependencies: <code>pip install -r requirements.txt</code></li>
  <li>Capture images: <code>python generateTrainingData.py</code></li>
  <li>Train model: <code>python trainModel.py</code></li>
  <li>Test live: <code>python testModel.py</code></li>
</ol>

<h2>Notes & Improvements</h2>
<ul>
  <li>Ensure file naming consistency: prefer <code>PersonName_1.jpg</code> format.</li>
  <li>Use <code>argparse</code> to pass camera index, image count, data path, and model path.</li>
  <li>Augment training data (flip/rotate/brightness) to improve robustness.</li>
  <li>Use a train/test split and evaluate recognition metrics.</li>
  <li>Consider migrating to deep learning (FaceNet / ArcFace) for higher accuracy.</li>
</ul>

<hr/>
<p align="center">Developed by Mostafa Sharqawy — AI Engineer | Computer Vision</p>
