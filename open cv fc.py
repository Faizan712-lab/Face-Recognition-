import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import cv2
from PIL import Image, ImageTk
import face_recognition
import numpy as np
import os
import pickle

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition System")
        
        # Initialize variables for known faces
        self.known_face_encodings = []
        self.known_face_names = []
        self.verification_active = False
        self.register_mode = False
        self.process_this_frame = True
        
        # Load stored face data if available
        self.load_known_faces()
        
        # Create GUI components
        self.create_widgets()
        
        # Initialize video capture
        self.video_capture = cv2.VideoCapture(0)
        if not self.video_capture.isOpened():
            messagebox.showerror("Error", "Unable to access webcam")
            self.root.destroy()
            return
        
        # Start video preview
        self.update_video()
    
    def load_known_faces(self):
        """Load face encodings and names from storage if the file exists."""
        if os.path.exists("known_faces.pkl"):
            try:
                with open("known_faces.pkl", "rb") as f:
                    data = pickle.load(f)
                self.known_face_encodings = data.get("encodings", [])
                self.known_face_names = data.get("names", [])
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load known faces: {str(e)}")
                self.known_face_encodings = []
                self.known_face_names = []
        else:
            self.known_face_encodings = []
            self.known_face_names = []
    
    def save_known_faces(self):
        """Save face encodings and names to storage."""
        data = {"encodings": self.known_face_encodings, "names": self.known_face_names}
        try:
            with open("known_faces.pkl", "wb") as f:
                pickle.dump(data, f)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save known faces: {str(e)}")
    
    def create_widgets(self):
        # Video display label
        self.video_label = tk.Label(self.root)
        self.video_label.pack(padx=10, pady=10)
        
        # Control buttons frame
        control_frame = ttk.Frame(self.root)
        control_frame.pack(pady=10)
        
        self.register_btn = ttk.Button(
            control_frame, 
            text="Register Face", 
            command=self.register_face
        )
        self.register_btn.pack(side=tk.LEFT, padx=5)
        
        self.verify_btn = ttk.Button(
            control_frame, 
            text="Start Verification", 
            command=self.toggle_verification
        )
        self.verify_btn.pack(side=tk.LEFT, padx=5)
        
        # New Delete Face Button
        self.delete_btn = ttk.Button(
            control_frame,
            text="Delete Face",
            command=self.delete_face
        )
        self.delete_btn.pack(side=tk.LEFT, padx=5)
        
        self.exit_btn = ttk.Button(
            control_frame, 
            text="Exit", 
            command=self.cleanup
        )
        self.exit_btn.pack(side=tk.LEFT, padx=5)
    
    def register_face(self):
        """Activate registration mode."""
        self.register_mode = True
        self.register_btn.config(state=tk.DISABLED)
        messagebox.showinfo("Info", "Face the camera and click OK", parent=self.root)
    
    def toggle_verification(self):
        """Toggle the verification mode on/off."""
        self.verification_active = not self.verification_active
        btn_text = "Stop Verification" if self.verification_active else "Start Verification"
        self.verify_btn.config(text=btn_text)
    
    def update_video(self):
        ret, frame = self.video_capture.read()
        if ret:
            # Resize frame for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Process registration if active
            if self.register_mode:
                self.process_registration(rgb_small_frame)
            
            # Process verification if active and not in registration mode
            if self.verification_active and not self.register_mode:
                frame = self.process_verification(rgb_small_frame, frame)
            
            # Convert the processed frame to a Tkinter-compatible image
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        
        self.root.after(10, self.update_video)
    
    def process_registration(self, small_frame):
        """Register a new face by capturing its encoding and saving it permanently."""
        face_locations = face_recognition.face_locations(small_frame)
        
        if len(face_locations) == 0:
            messagebox.showwarning("Warning", "No face detected")
        else:
            face_encodings = face_recognition.face_encodings(small_frame, face_locations)
            if face_encodings:
                face_encoding = face_encodings[0]
                # Ask the user to input a name for the face
                name = simpledialog.askstring("Input", "Enter name for this face:", parent=self.root)
                if not name:
                    name = f"User {len(self.known_face_names) + 1}"
                self.known_face_encodings.append(face_encoding)
                self.known_face_names.append(name)
                # Save the updated face data permanently
                self.save_known_faces()
                messagebox.showinfo("Success", f"Face registered as {name}")
        
        self.register_mode = False
        self.register_btn.config(state=tk.NORMAL)
    
    def process_verification(self, small_frame, original_frame):
        """Verify detected faces against the stored known faces."""
        face_locations = face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Scale face locations back to the original frame size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            
            # Compare detected face to known faces
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"
            
            if True in matches:
                first_match_index = matches.index(True)
                name = self.known_face_names[first_match_index]
            
            # Draw rectangle around the face and put the name label
            cv2.rectangle(original_frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(original_frame, name, (left + 6, bottom - 6), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
        
        return original_frame
    
    def delete_face(self):
        """Open a window to select and delete a registered face."""
        if not self.known_face_names:
            messagebox.showinfo("Info", "No faces registered to delete.")
            return
        
        # Create a new window for deletion
        delete_window = tk.Toplevel(self.root)
        delete_window.title("Delete Face")
        
        label = ttk.Label(delete_window, text="Select a face to delete:")
        label.pack(padx=10, pady=10)
        
        # Listbox to show registered faces
        listbox = tk.Listbox(delete_window, width=30, height=10)
        listbox.pack(padx=10, pady=10)
        
        # Insert face names into the listbox
        for name in self.known_face_names:
            listbox.insert(tk.END, name)
        
        def confirm_deletion():
            selection = listbox.curselection()
            if not selection:
                messagebox.showwarning("Warning", "No face selected.")
                return
            index = selection[0]
            name = self.known_face_names[index]
            # Confirm deletion
            confirm = messagebox.askyesno("Confirm", f"Are you sure you want to delete {name}?")
            if confirm:
                del self.known_face_names[index]
                del self.known_face_encodings[index]
                self.save_known_faces()
                messagebox.showinfo("Deleted", f"Deleted face {name}.")
                delete_window.destroy()
        
        delete_button = ttk.Button(delete_window, text="Delete", command=confirm_deletion)
        delete_button.pack(padx=10, pady=5)
        
        cancel_button = ttk.Button(delete_window, text="Cancel", command=delete_window.destroy)
        cancel_button.pack(padx=10, pady=5)
    
    def cleanup(self):
        if self.video_capture.isOpened():
            self.video_capture.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()
