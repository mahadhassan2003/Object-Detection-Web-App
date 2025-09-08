
# Object Detection Web App

![128 Technologies](static/logo-128tech.png)

A sophisticated web application for real-time object detection using YOLO models with similarity-based classification. Built with FastAPI, PyTorch, and modern web technologies.

## 🚀 Features

- **Real-time Object Detection**: Live webcam feed processing with YOLO models
- **Multiple YOLO Models**: Support for YOLOv8n, YOLOv8s, YOLOv8m, YOLOv8l, and YOLOv8x
- **Similarity-based Classification**: Uses MobileNet embeddings for object similarity comparison
- **User Authentication**: Secure login/registration system with JWT tokens
- **Interactive UI**: Modern, responsive interface with dark/light theme support
- **Metrics Dashboard**: Real-time accuracy, precision, and mAP metrics
- **Fullscreen Mode**: Enhanced viewing experience for webcam feed

## 🛠️ Technology Stack

- **Backend**: FastAPI, Python 3.12+
- **Machine Learning**: PyTorch, Ultralytics YOLO, OpenCV
- **Database**: PostgreSQL
- **Frontend**: Vanilla JavaScript, HTML5, CSS3
- **Authentication**: JWT, bcrypt
- **Image Processing**: PIL, NumPy

## 📋 Prerequisites

- Python 3.12 or higher
- PostgreSQL database
- Webcam (for real-time detection)

## 🔧 Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd 128
   ```

2. **Install dependencies**:
   ```bash
   pip install fastapi uvicorn torch torchvision ultralytics opencv-python pillow numpy scikit-learn psycopg2-extras bcrypt pyjwt python-multipart
   ```

3. **Database Setup**:
   - Install PostgreSQL
   - Create a database named "Data"
   - Update database credentials in `app.py`:
     ```python
     DB_CONFIG = {
         "host": "localhost",
         "database": "Data", 
         "user": "your_username",
         "password": "your_password",
         "port": "5432"
     }
     ```

4. **Download YOLO Model**:
   The app will automatically download `yolov8n.pt` on first run, or you can download it manually.

## 🚀 Running the Application

1. **Start the server**:
   ```bash
   cd 128
   python app.py
   ```
   or
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000 --reload
   ```

2. **Access the application**:
   - Open your browser and navigate to `http://localhost:8000`
   - You'll be redirected to the login page
   - Register a new account or login with existing credentials

## 📖 Usage

### 1. Authentication
- Register a new account at `/register`
- Login at `/login`
- The app uses JWT tokens for secure authentication

### 2. Object Detection Setup
1. **Select Model**: Choose from YOLOv8 variants (n, s, m, l, x)
2. **Upload Reference Images**: Images containing the object you want to detect
3. **Upload Validation Images**: Images for testing accuracy
4. **Set Validation Labels**: Comma-separated 0s and 1s indicating object presence
5. **Choose Target Class**: Select from available YOLO classes
6. **Configure Parameters**:
   - Similarity threshold (0.0-1.0)
   - Image resize dimensions
   - Normalization parameters

### 3. Real-time Detection
- Click "Start Webcam" to begin live detection
- The system will highlight detected objects with bounding boxes
- Green boxes indicate "known" objects (above threshold)
- Red boxes indicate "unknown" objects (below threshold)
- Use fullscreen mode for better visualization

## 🎯 API Endpoints

### Authentication
- `POST /api/register` - Register new user
- `POST /api/login` - User login
- `GET /logout` - User logout

### Object Detection
- `POST /process_images` - Process reference and validation images
- `POST /process_webcam_frame` - Process single webcam frame
- `GET /class_names` - Get available YOLO classes

### Static Files
- `GET /` - Main application (redirects to auth)
- `GET /login` - Login page
- `GET /register` - Registration page

## 🏗️ Project Structure

```
128/
├── app.py                 # Main FastAPI application
├── static/               # Frontend assets
│   ├── index.html        # Main application interface
│   ├── login.html        # Login page
│   ├── register.html     # Registration page
│   ├── auth-redirect.html # Authentication redirect
│   ├── styles.css        # Main application styles
│   ├── auth.css          # Authentication page styles
│   ├── script.js         # Frontend JavaScript
│   └── logo-128tech.png  # Company logo
├── pyproject.toml        # Python project configuration
├── yolov8n.pt           # YOLO model weights
└── README.md            # Project documentation
```

## ⚙️ Configuration

### Database Configuration
Update the `DB_CONFIG` dictionary in `app.py`:
```python
DB_CONFIG = {
    "host": "your_host",
    "database": "your_database", 
    "user": "your_username",
    "password": "your_password",
    "port": "5432"
}
```

### Security Configuration
Change the JWT secret key in production:
```python
SECRET_KEY = "your_secure_secret_key_here"
```

### Model Configuration
The app supports different YOLO models. Larger models offer better accuracy but require more computational resources:
- `yolov8n`: Nano (fastest, least accurate)
- `yolov8s`: Small
- `yolov8m`: Medium  
- `yolov8l`: Large
- `yolov8x`: Extra Large (slowest, most accurate)

## 🎨 Features Overview

### Theme Support
- Light and dark theme toggle
- Automatic theme persistence
- Modern glassmorphism design

### Responsive Design
- Mobile-friendly interface
- Adaptive layouts for different screen sizes
- Touch-friendly controls

### Real-time Processing
- Live webcam feed processing
- Bounding box overlays
- Similarity score display
- Status indicators

## 🔒 Security Features

- Password hashing with bcrypt
- JWT token authentication
- SQL injection protection
- Input validation and sanitization

## 🚀 Deployment on Replit

This project is ready for deployment on Replit:

1. Import the project to Replit
2. Configure the database (Replit provides PostgreSQL)
3. Update the database configuration
4. Run the application

## 👥 Development Team

**Developed by:**
- Mahad Hassan
- Saad
- Tauqeer
- Samia
- Ayesha

**Company:** 128 Technologies

## 📧 Contact

- **Email**: info@128technologies.com.pk
- **LinkedIn**: [128 Technologies](https://www.linkedin.com/company/128-technologies/posts/?feedView=all)

## 📄 License

© Copyright 128 Technologies

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 🐛 Troubleshooting

### Common Issues

1. **Database Connection Error**:
   - Verify PostgreSQL is running
   - Check database credentials
   - Ensure database exists

2. **Webcam Not Working**:
   - Check browser permissions
   - Ensure webcam is not used by other applications
   - Try refreshing the page

3. **Model Loading Issues**:
   - Check internet connection for model download
   - Verify sufficient disk space
   - Ensure PyTorch is properly installed

### Performance Tips

- Use smaller YOLO models (yolov8n/yolov8s) for real-time performance
- Reduce image resolution for faster processing
- Close other applications to free up system resources

---

*Built with ❤️ by 128 Technologies*
