
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

# Set page config
st.set_page_config(
    page_title="Face Detection App",
    page_icon="👤",
    layout="wide"
)

# Title and description
st.title("🎯 Face Detection with OpenCV")
st.markdown("Upload an image or use your webcam to detect faces in real-time!")

# Load the Haar Cascade classifier
@st.cache_resource
def load_face_classifier():
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

face_classifier = load_face_classifier()

# Face detection function
def detect_faces(image, scale_factor=1.1, min_neighbors=5, min_size=(40, 40)):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_classifier.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size
    )
    
    # Draw rectangles around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
    
    return image, len(faces)

# Sidebar for parameters
st.sidebar.header("Detection Parameters")
scale_factor = st.sidebar.slider("Scale Factor", 1.01, 2.0, 1.1, 0.01)
min_neighbors = st.sidebar.slider("Min Neighbors", 1, 10, 5)
min_size = st.sidebar.slider("Min Size", 20, 100, 40)

# Main app sections
tab1, tab2, tab3 = st.tabs(["📷 Image Upload", "🎥 Webcam", "ℹ️ About"])

with tab1:
    st.header("Upload Image for Face Detection")
    
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Read image
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        if len(image_np.shape) == 3:
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image_np
        
        # Create columns for before/after
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_column_width=True)
        
        with col2:
            st.subheader("Face Detection Result")
            
            # Detect faces
            result_image, num_faces = detect_faces(
                image_bgr.copy(), 
                scale_factor, 
                min_neighbors, 
                (min_size, min_size)
            )
            
            # Convert back to RGB for display
            result_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
            st.image(result_rgb, use_column_width=True)
            
            # Display results
            if num_faces > 0:
                st.success(f"✅ Detected {num_faces} face(s)!")
            else:
                st.warning("⚠️ No faces detected. Try adjusting the parameters.")

with tab2:
    st.header("Real-time Face Detection")
    st.markdown("Click 'Start Webcam' to begin real-time face detection")
    
    # Webcam controls
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        start_webcam = st.button("Start Webcam", type="primary")
    
    with col2:
        stop_webcam = st.button("Stop Webcam")
    
    # Placeholder for webcam feed
    webcam_placeholder = st.empty()
    stats_placeholder = st.empty()
    
    if start_webcam:
        # Initialize session state
        if 'webcam_running' not in st.session_state:
            st.session_state.webcam_running = True
        
        # Try to access webcam
        try:
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                st.error("❌ Could not access webcam. Please check your camera permissions.")
            else:
                st.success("📹 Webcam started successfully!")
                
                # Create a container for the video stream
                frame_count = 0
                
                while st.session_state.get('webcam_running', True) and not stop_webcam:
                    ret, frame = cap.read()
                    
                    if not ret:
                        st.error("Failed to read from webcam")
                        break
                    
                    # Detect faces in the frame
                    processed_frame, num_faces = detect_faces(
                        frame.copy(),
                        scale_factor,
                        min_neighbors,
                        (min_size, min_size)
                    )
                    
                    # Convert BGR to RGB for Streamlit
                    frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    
                    # Display the frame
                    webcam_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                    
                    # Update stats
                    frame_count += 1
                    stats_placeholder.metric("Faces Detected", num_faces)
                    
                    # Break if stop button is pressed
                    if stop_webcam:
                        break
                
                cap.release()
                st.session_state.webcam_running = False
                
        except Exception as e:
            st.error(f"Error accessing webcam: {str(e)}")
    
    if stop_webcam:
        st.session_state.webcam_running = False
        webcam_placeholder.empty()
        stats_placeholder.empty()
        st.info("📴 Webcam stopped")

with tab3:
    st.header("About Face Detection")
    
    st.markdown("""
    ### 🔍 How it Works
    
    This application uses **Haar Cascade Classifiers** from OpenCV to detect faces in images and video streams.
    
    **Key Features:**
    - 📸 **Image Upload**: Upload any image to detect faces
    - 🎥 **Real-time Detection**: Use your webcam for live face detection
    - ⚙️ **Adjustable Parameters**: Fine-tune detection sensitivity
    
    ### 🛠️ Parameters Explanation
    
    - **Scale Factor**: How much the image size is reduced at each scale (1.1 = 10% reduction)
    - **Min Neighbors**: How many neighbors each candidate rectangle should retain
    - **Min Size**: Minimum possible face size (smaller faces are ignored)
    
    ### 🚀 Technology Stack
    - **Streamlit**: Web app framework
    - **OpenCV**: Computer vision library
    - **Haar Cascades**: Pre-trained face detection model
    
    ### 📝 Usage Tips
    - Ensure good lighting for better detection
    - Face should be relatively frontal
    - Adjust parameters if detection is too sensitive or not sensitive enough
    """)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and OpenCV")
