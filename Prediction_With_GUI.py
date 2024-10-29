import tkinter as tk
from tkinter import messagebox
import cv2
import operator
import os
import numpy as np
from keras.models import load_model

# Change directory to where your model files are located
os.chdir(r'C:\Users\Ajaya\Downloads\Hand Gesture Recognition\CAPSTONE PROJECT 1')

# Load the gesture recognition model and weights
classifier = load_model('sign_language_gesture_recognition.h5')
classifier.load_weights('sign_language_categorical_weights.h5')

# Create a Tkinter window for login
window = tk.Tk()
window.geometry("240x100")
window.title('Sign Language Gesture Recognition Login')
window.resizable(0, 0)

window.columnconfigure(0, weight=1)
window.columnconfigure(1, weight=3)

# Add username and password labels and entries
username_label = tk.Label(window, text='Username')
username_label.grid(column=0, row=0, sticky=tk.W, padx=5, pady=5)
username_entry = tk.Entry(window)
username_entry.grid(column=1, row=0, sticky=tk.E, padx=5, pady=5)

password_label = tk.Label(window, text='Password')
password_label.grid(column=0, row=1, sticky=tk.W, padx=5, pady=5)
password_entry = tk.Entry(window, show='*')
password_entry.grid(column=1, row=1, sticky=tk.E, padx=5, pady=5)


def slgr():
    if (username_entry.get() == 'arv' and password_entry.get() == '1234'):
        messagebox.showinfo('Result', 'Welcome ' + str(username_entry.get()))
        cap = cv2.VideoCapture(0)  # Open the camera

        while True:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)  # Flip the frame horizontally

            # Define region of interest (ROI) where the hand will be detected
            roi = frame[120:400, 320:620]

            # Convert the ROI to HSV color space
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # Define skin color range in HSV
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)

            # Create a binary mask where the skin color is detected
            mask = cv2.inRange(hsv, lower_skin, upper_skin)

            # Apply Gaussian blur to the mask to reduce noise
            mask = cv2.GaussianBlur(mask, (5, 5), 0)

            # Find contours in the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Only proceed if contours are found
            if contours:
                # Get the largest contour (which should be the hand)
                max_contour = max(contours, key=cv2.contourArea)
                contour_area = cv2.contourArea(max_contour)

                # Set a minimum area threshold for hand detection
                if contour_area > 5000:
                    # Draw the contour on the ROI for visualization
                    cv2.drawContours(roi, [max_contour], -1, (0, 255, 0), 3)

                    # Resize the ROI to match the input size of the model (64x64)
                    roi_resized = cv2.resize(mask, (64, 64))

                    # Copy of the original frame for drawing the rectangle and text
                    copy = frame.copy()
                    cv2.rectangle(copy, (320, 120), (620, 400), (255, 0, 0), 5)

                    # Make a prediction using the pre-trained model
                    result = classifier.predict(roi_resized.reshape(1, 64, 64, 1))
                    prediction = {
                        'DONE': result[0][0],
                        'HELLO': result[0][1],
                        'LEFT': result[0][2],
                        'NO': result[0][3],
                        'RIGHT': result[0][4],
                        'THANK YOU': result[0][5],
                        'YES': result[0][6]
                    }
                    predicted = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
                    res = predicted[0][0]

                    # Display the prediction result on the frame
                    cv2.rectangle(copy, (25, 45), (590, 115), (255, 255, 255), -1)
                    cv2.putText(copy, 'Please Put Up Your Right Hand', (30, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
                    cv2.putText(copy, 'In The Box', (30, 110), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
                    cv2.putText(copy, "User Response: " + res, (10, 420), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.imshow('Sign Language Gesture Recognition', copy)

            # Display the ROI with the detected hand contour
            cv2.imshow('ROI', roi)

            # Exit if 'Enter' key is pressed
            if cv2.waitKey(1) == 13:
                break

        # Release the camera and close all OpenCV windows
        cap.release()
        cv2.destroyAllWindows()
    else:
        messagebox.showinfo('Error', 'Authorized Personnel Only')


# Create login and exit buttons
login_button = tk.Button(window, text='Login', command=slgr)
login_button.grid(column=1, row=3, sticky=tk.E, padx=5, pady=5)

exit_button = tk.Button(window, text='Exit', command=window.destroy)
exit_button.grid(column=2, row=3, sticky=tk.W, padx=5, pady=5)

# Run the Tkinter event loop
window.mainloop()
