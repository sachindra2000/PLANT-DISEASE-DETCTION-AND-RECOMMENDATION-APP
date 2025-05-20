"""
THIS IS THE CODE FOR THE ROBOT CAR THAT WAS USED FOR THE FINAL PROJECT, THE ROBOT CAR BASE IS THE PICAR-X.
THE ORIGINAL CODE WAS MODIFIED TO INCLUDE THE TFLITE MODEL FOR PLANT DISEASE DETECTION.
THE CODE WAS TESTED ON THE PICAR-X ROBOT CAR AND WORKS AS EXPECTED.
"""



from robot_hat.utils import reset_mcu
from picarx import Picarx
from vilib import Vilib
from time import sleep, time, strftime, localtime
import readchar
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import os

# Get user directory for saving photos
user = os.getlogin()
user_home = os.path.expanduser(f'~{user}')

reset_mcu()
sleep(0.2)

manual = '''
Press key to call the function(non-case sensitive):

    O: speed up
    P: speed down
    W: forward  
    S: backward
    A: turn left
    D: turn right
    F: stop
    T: take photo & analyze

    Ctrl+C: quit
'''

px = Picarx()

# Load the TFLite model
interpreter = tflite.Interpreter(model_path="plant_disease_model.tflite")
interpreter.allocate_tensors()

# Get model input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define plant disease labels (modify according to your dataset)
disease_labels = ['Aphid', 'Black Rust', 'Blast', 'Brown Rust', 'Common Root Rot', 'Fusarium Head Blight', 'Healthy', 'Leaf Blight', 'Mildew', 'Mite', 'Septoria', 'Smut', 'Stem Fly', 'Tan Spot', 'Yellow Rust']

def take_photo_and_analyze():
    """Capture an image, save it, and run plant disease detection."""
    _time = strftime('%Y-%m-%d-%H-%M-%S', localtime(time()))
    name = f'photo_{_time}'
    path = f"{user_home}/Pictures/picar-x/"
    os.makedirs(path, exist_ok=True)  # Ensure directory exists

    # Capture the image
    Vilib.take_photo(name, path)
    img_path = f"{path}{name}.jpg"
    print(f'\nðŸ“¸ Photo saved as {img_path}')

    # Load the captured image
    img = cv2.imread(img_path)
    if img is None:
        print("âš ï¸ Error: Could not read the captured image.")
        return

    # Get the expected input shape from the model
    model_input_shape = input_details[0]['shape']
    print(f"âœ… Model expects input shape: {model_input_shape}")  # Debugging info

    # Extract required dimensions
    _, height, width, channels = model_input_shape

    # Resize the image correctly
    img = cv2.resize(img, (width, height))  # Resize to expected size
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img.astype(np.float32) / 255.0  # Normalize pixel values

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Get prediction result
    predicted_label = disease_labels[np.argmax(output_data)]
    confidence = np.max(output_data)

    print(f"ðŸ” Detected Disease: {predicted_label} ({confidence:.2f})")
    os.system(f"espeak '{predicted_label}' 2>/dev/null")

def move(operate: str, speed):
    """Control the movement of the Picar-X."""
    if operate == 'stop':
        px.stop()  
    else:
        if operate == 'forward':
            px.set_dir_servo_angle(0)
            px.forward(speed)
        elif operate == 'backward':
            px.set_dir_servo_angle(0)
            px.backward(speed)
        elif operate == 'turn left':
            px.set_dir_servo_angle(-30)
            px.forward(speed)
        elif operate == 'turn right':
            px.set_dir_servo_angle(30)
            px.forward(speed)

def main():
    speed = 0
    status = 'stop'

    Vilib.camera_start(vflip=False, hflip=False)
    Vilib.display(local=True, web=True)
    sleep(2)  # Wait for startup
    print(manual)
    
    while True:
        print("\rstatus: %s , speed: %s    " % (status, speed), end='', flush=True)
        
        # Read keyboard input
        key = readchar.readkey().lower()
        
        # Movement operations
        if key in ('wsadfop'):
            if key == 'o':
                if speed <= 90:
                    speed += 10           
            elif key == 'p':
                if speed >= 10:
                    speed -= 10
                if speed == 0:
                    status = 'stop'
            elif key in ('wsad'):
                if speed == 0:
                    speed = 10
                if key == 'w':
                    if status != 'forward' and speed > 60:  
                        speed = 60
                    status = 'forward'
                elif key == 'a':
                    status = 'turn left'
                elif key == 's':
                    if status != 'backward' and speed > 60:
                        speed = 60
                    status = 'backward'
                elif key == 'd':
                    status = 'turn right' 
            elif key == 'f':
                status = 'stop'

            move(status, speed)  
        
        # Take photo and analyze
        elif key == 't':
            take_photo_and_analyze()
        
        # Quit
        elif key == readchar.key.CTRL_C:
            print('\nExiting...')
            px.stop()
            Vilib.camera_close()
            break 

        sleep(0.1)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:    
        print(f"âŒ Error: {e}")
    finally:
        px.stop()
        Vilib.camera_close()
