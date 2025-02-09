Face Recognition System with Tkinter
This project is a simple face recognition application built using Python, Tkinter, OpenCV, and the face_recognition library. It allows users to register faces with custom names, verify faces using their webcam, and even delete registered faces. Registered faces are stored persistently using Python’s pickle module so that the data remains available across sessions.

Features
Face Registration:

Capture a face from the webcam and register it.
Prompt the user to enter a custom name for the registered face.
Persistently save face encodings and names in a file (known_faces.pkl).
Face Verification:

Real-time face detection and recognition using your webcam.
Compare detected faces against the stored face encodings.
Label recognized faces with their saved names, while unknown faces are labeled as "Unknown".
Face Deletion:

A dedicated GUI window allows users to view and delete registered faces.
Update the persistent storage once a face is deleted.
Persistent Storage:

Uses Python’s pickle module to store and load registered face data, ensuring your data is maintained between sessions.
Dependencies
Make sure you have Python 3.x installed. The following Python libraries are required:

Tkinter (usually comes with Python)
OpenCV (opencv-python)
face_recognition
Pillow
numpy
Install the necessary packages using pip:

bash
Copy
Edit
pip install opencv-python face_recognition Pillow numpy
Usage
Clone the repository:

bash
Copy
Edit
git clone https://github.com/your-username/your-repo-name.git
Navigate to the project directory:

bash
Copy
Edit
cd your-repo-name
Run the application:

bash
Copy
Edit
python face_recognition_app.py
Note: Ensure your webcam is connected and accessible.

How It Works
Register Face: Click on the "Register Face" button, face the camera, and then enter your name when prompted. The face encoding along with the name is stored in a persistent file.
Start/Stop Verification: Click on "Start Verification" to begin real-time recognition. Detected faces will be compared with the registered ones and labeled accordingly.
Delete Face: Use the "Delete Face" button to open a window listing all registered faces. Select a face and confirm deletion to remove it from both the display and the persistent storage.
Contributing
Contributions are welcome! Feel free to fork the repository, make improvements, and open a pull request with your changes.
