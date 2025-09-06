let stream = null;
let webcamInterval = null;
let isFullscreen = false;
let authToken = null;

// Authentication token management
function getAuthToken() {
    return localStorage.getItem('auth_token');
}

function setAuthToken(token) {
    localStorage.setItem('auth_token', token);
    authToken = token;
}

function removeAuthToken() {
    localStorage.removeItem('auth_token');
    authToken = null;
}

// Check if user is authenticated
function checkAuth() {
    const token = getAuthToken();
    if (!token) {
        window.location.href = '/login';
        return false;
    }
    authToken = token;
    return true;
}

// Add auth headers to requests
function getAuthHeaders() {
    const token = getAuthToken();
    return token ? { 'Authorization': `Bearer ${token}` } : {};
}

// Theme toggle functionality
function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';

    document.documentElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
}

// Initialize theme on page load
document.addEventListener('DOMContentLoaded', () => {
    // Check authentication first
    if (!checkAuth()) {
        return;
    }
    
    const savedTheme = localStorage.getItem('theme') || 'light';
    document.documentElement.setAttribute('data-theme', savedTheme);

    // Fetch class names with retry
    async function fetchClassNames(attempt = 1, maxAttempts = 2) {
        try {
            const response = await fetch('/class_names', {
                headers: getAuthHeaders()
            });
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}, ${response.statusText}`);
            }
            const data = await response.json();
            console.log('Class names response:', data); // Debug: Log raw response
            if (!data.class_names || !Array.isArray(data.class_names)) {
                throw new Error('Invalid response format: class_names not found or not an array');
            }
            const classSelect = document.getElementById('class_name');
            // Clear existing options except the default
            while (classSelect.options.length > 1) {
                classSelect.remove(1);
            }
            data.class_names.forEach(name => {
                const option = document.createElement('option');
                option.value = name;
                option.textContent = name;
                classSelect.appendChild(option);
            });
            document.getElementById('debug-info').textContent = `Loaded ${data.class_names.length} class names`;
            console.log(`Populated ${data.class_names.length} class names:`, data.class_names);
        } catch (error) {
            console.error(`Attempt ${attempt} - Error fetching class names:`, error);
            document.getElementById('logs').textContent = `Error fetching class names: ${error.message}`;
            document.getElementById('debug-info').textContent = `Error fetching class names: ${error.message}`;
            if (attempt < maxAttempts) {
                console.log(`Retrying fetchClassNames (Attempt ${attempt + 1})`);
                setTimeout(() => fetchClassNames(attempt + 1, maxAttempts), 1000);
            }
        }
    }

    fetchClassNames();
});

async function startWebcam() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        const video = document.getElementById('webcam');
        video.srcObject = stream;
        document.getElementById('webcam-status').textContent = 'Running';
        document.getElementById('startWebcam').disabled = true;
        document.getElementById('stopWebcam').disabled = false;
        document.getElementById('fullscreenWebcam').disabled = false;
        document.getElementById('annotated-frame').style.display = 'none';
        document.getElementById('webcam').style.display = 'block';
        document.getElementById('debug-info').textContent = 'Webcam started';

        // Start sending frames
        webcamInterval = setInterval(sendWebcamFrame, 500);
    } catch (error) {
        document.getElementById('logs').textContent = 'Error starting webcam: ' + error.message;
        document.getElementById('debug-info').textContent = 'Error: ' + error.message;
    }
}

function stopWebcam() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
        document.getElementById('webcam').srcObject = null;
        document.getElementById('webcam').style.display = 'block';
        document.getElementById('annotated-frame').style.display = 'none';
        document.getElementById('webcam-status').textContent = 'Stopped';
        document.getElementById('startWebcam').disabled = false;
        document.getElementById('stopWebcam').disabled = true;
        document.getElementById('fullscreenWebcam').disabled = true;
        document.getElementById('debug-info').textContent = 'Webcam stopped';
        clearInterval(webcamInterval);
        
        // Exit fullscreen if active
        if (isFullscreen) {
            exitFullscreen();
        }
    }
}

async function sendWebcamFrame() {
    const video = document.getElementById('webcam');
    const canvas = document.getElementById('canvas');
    const overlayCanvas = document.getElementById('overlay-canvas');
    const debugInfo = document.getElementById('debug-info');
    const threshold = parseFloat(document.getElementById('threshold').value);

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Set overlay canvas size to match video display size
    const videoRect = video.getBoundingClientRect();
    overlayCanvas.width = video.offsetWidth;
    overlayCanvas.height = video.offsetHeight;
    overlayCanvas.style.width = video.offsetWidth + 'px';
    overlayCanvas.style.height = video.offsetHeight + 'px';
    overlayCanvas.style.position = 'absolute';
    overlayCanvas.style.top = '0';
    overlayCanvas.style.left = '0';
    overlayCanvas.style.pointerEvents = 'none';

    const formData = new FormData();
    canvas.toBlob(async blob => {
        formData.append('frame', blob, 'frame.jpg');
        formData.append('threshold', threshold);

        try {
            const response = await fetch('/process_webcam_frame', {
                method: 'POST',
                headers: getAuthHeaders(),
                body: formData
            });
            const result = await response.json();
            if (response.ok) {
                document.getElementById('logs').textContent = `${result.label} (Boxes drawn: ${result.boxes_drawn})`;
                debugInfo.textContent = `Received bounding boxes: ${result.boxes_drawn}`;
                
                // Draw bounding boxes on overlay canvas if we have detection data
                if (result.bounding_boxes && result.bounding_boxes.length > 0) {
                    const overlayCtx = overlayCanvas.getContext('2d');
                    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
                    
                    // Calculate scaling factors
                    const scaleX = video.offsetWidth / video.videoWidth;
                    const scaleY = video.offsetHeight / video.videoHeight;
                    
                    result.bounding_boxes.forEach(box => {
                        // Scale coordinates to match video display size
                        const x1 = box.x1 * scaleX;
                        const y1 = box.y1 * scaleY;
                        const x2 = box.x2 * scaleX;
                        const y2 = box.y2 * scaleY;
                        
                        overlayCtx.strokeStyle = box.color;
                        overlayCtx.lineWidth = 3;
                        overlayCtx.strokeRect(x1, y1, x2 - x1, y2 - y1);
                        
                        // Draw label
                        overlayCtx.fillStyle = box.color;
                        overlayCtx.font = '16px Arial';
                        overlayCtx.fillText(box.label, x1, y1 - 10);
                    });
                } else {
                    // Clear overlay if no boxes
                    const overlayCtx = overlayCanvas.getContext('2d');
                    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
                }
                
                // Keep video visible
                video.style.display = 'block';
                document.getElementById('annotated-frame').style.display = 'none';
                
                // Update fullscreen display if active
                if (isFullscreen) {
                    const fullscreenVideo = document.getElementById('fullscreen-webcam');
                    const fullscreenOverlay = document.getElementById('fullscreen-overlay-canvas');
                    
                    if (fullscreenOverlay) {
                        fullscreenOverlay.width = fullscreenVideo.videoWidth;
                        fullscreenOverlay.height = fullscreenVideo.videoHeight;
                        fullscreenOverlay.style.width = fullscreenVideo.offsetWidth + 'px';
                        fullscreenOverlay.style.height = fullscreenVideo.offsetHeight + 'px';
                        
                        if (result.bounding_boxes && result.bounding_boxes.length > 0) {
                            const fullscreenCtx = fullscreenOverlay.getContext('2d');
                            fullscreenCtx.clearRect(0, 0, fullscreenOverlay.width, fullscreenOverlay.height);
                            
                            result.bounding_boxes.forEach(box => {
                                fullscreenCtx.strokeStyle = box.color;
                                fullscreenCtx.lineWidth = 3;
                                fullscreenCtx.strokeRect(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1);
                                
                                fullscreenCtx.fillStyle = box.color;
                                fullscreenCtx.font = '16px Arial';
                                fullscreenCtx.fillText(box.label, box.x1, box.y1 - 10);
                            });
                        }
                    }
                    
                    fullscreenVideo.style.display = 'block';
                    document.getElementById('fullscreen-annotated-frame').style.display = 'none';
                }
            } else {
                document.getElementById('logs').textContent = 'Error: ' + result.detail;
                debugInfo.textContent = 'Error: ' + result.detail;
                
                // Clear overlay on error
                const overlayCtx = overlayCanvas.getContext('2d');
                overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
                
                video.style.display = 'block';
                document.getElementById('annotated-frame').style.display = 'none';
                
                // Update fullscreen display if active
                if (isFullscreen) {
                    const fullscreenVideo = document.getElementById('fullscreen-webcam');
                    const fullscreenOverlay = document.getElementById('fullscreen-overlay-canvas');
                    
                    if (fullscreenOverlay) {
                        const fullscreenCtx = fullscreenOverlay.getContext('2d');
                        fullscreenCtx.clearRect(0, 0, fullscreenOverlay.width, fullscreenOverlay.height);
                    }
                    
                    fullscreenVideo.style.display = 'block';
                    document.getElementById('fullscreen-annotated-frame').style.display = 'none';
                }
            }
        } catch (error) {
            document.getElementById('logs').textContent = 'Error processing frame: ' + error.message;
            debugInfo.textContent = 'Error: ' + error.message;
            
            // Clear overlay on error
            const overlayCtx = overlayCanvas.getContext('2d');
            overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
            
            video.style.display = 'block';
            document.getElementById('annotated-frame').style.display = 'none';
            
            // Update fullscreen display if active
            if (isFullscreen) {
                const fullscreenVideo = document.getElementById('fullscreen-webcam');
                const fullscreenOverlay = document.getElementById('fullscreen-overlay-canvas');
                
                if (fullscreenOverlay) {
                    const fullscreenCtx = fullscreenOverlay.getContext('2d');
                    fullscreenCtx.clearRect(0, 0, fullscreenOverlay.width, fullscreenOverlay.height);
                }
                
                fullscreenVideo.style.display = 'block';
                document.getElementById('fullscreen-annotated-frame').style.display = 'none';
            }
        }
    }, 'image/jpeg', 0.9); // Use quality 0.9 for better performance
}

function toggleFullscreen() {
    if (!isFullscreen) {
        enterFullscreen();
    } else {
        exitFullscreen();
    }
}

function enterFullscreen() {
    const overlay = document.getElementById('fullscreen-overlay');
    const normalVideo = document.getElementById('webcam');
    const normalAnnotatedFrame = document.getElementById('annotated-frame');
    const fullscreenVideo = document.getElementById('fullscreen-webcam');
    const fullscreenAnnotatedFrame = document.getElementById('fullscreen-annotated-frame');
    
    // Copy stream to fullscreen video
    if (stream) {
        fullscreenVideo.srcObject = stream;
    }
    
    // Copy annotated frame if visible
    if (normalAnnotatedFrame.style.display === 'block') {
        fullscreenAnnotatedFrame.src = normalAnnotatedFrame.src;
        fullscreenAnnotatedFrame.style.display = 'block';
        fullscreenVideo.style.display = 'none';
    } else {
        fullscreenVideo.style.display = 'block';
        fullscreenAnnotatedFrame.style.display = 'none';
    }
    
    overlay.style.display = 'flex';
    isFullscreen = true;
    document.getElementById('fullscreenWebcam').textContent = 'Exit Fullscreen';
    
    // Prevent body scroll
    document.body.style.overflow = 'hidden';
}

function exitFullscreen() {
    const overlay = document.getElementById('fullscreen-overlay');
    const fullscreenVideo = document.getElementById('fullscreen-webcam');
    
    overlay.style.display = 'none';
    fullscreenVideo.srcObject = null;
    isFullscreen = false;
    document.getElementById('fullscreenWebcam').textContent = 'Fullscreen';
    
    // Restore body scroll
    document.body.style.overflow = 'auto';
}

// Handle escape key to exit fullscreen
document.addEventListener('keydown', function(event) {
    if (event.key === 'Escape' && isFullscreen) {
        exitFullscreen();
    }
});

// Logout function
function logout() {
    removeAuthToken();
    window.location.href = '/login';
}

async function processImages() {
    const modelName = document.getElementById('model_name').value;
    const refImages = document.getElementById('ref_images').files;
    const valImages = document.getElementById('val_images').files;
    const valLabels = document.getElementById('val_labels').value;
    const className = document.getElementById('class_name').value;
    const threshold = parseFloat(document.getElementById('threshold').value);
    const resizeSize = parseInt(document.getElementById('resize_size').value);
    const normMean = document.getElementById('norm_mean').value;
    const normStd = document.getElementById('norm_std').value;

    if (!modelName || refImages.length === 0 || valImages.length === 0 || !valLabels || !className || isNaN(threshold) || isNaN(resizeSize) || !normMean || !normStd) {
        document.getElementById('logs').textContent = 'Please fill all fields correctly.';
        return;
    }

    const formData = new FormData();
    formData.append('model_name', modelName);
    for (let i = 0; i < refImages.length; i++) {
        formData.append('ref_images', refImages[i]);
    }
    for (let i = 0; i < valImages.length; i++) {
        formData.append('val_images', valImages[i]);
    }
    formData.append('val_labels', valLabels);
    formData.append('target_class_name', className);
    formData.append('threshold', threshold);
    formData.append('resize_size', resizeSize);
    formData.append('norm_mean', normMean);
    formData.append('norm_std', normStd);

    // Reset progress and outputs
    document.getElementById('progress').value = 0;
    document.getElementById('progress-text').textContent = '0%';
    document.getElementById('logs').textContent = 'Processing...';
    document.getElementById('map').textContent = 'N/A';
    document.getElementById('precision').textContent = 'N/A';
    document.getElementById('accuracy').textContent = 'N/A';
    document.getElementById('debug-info').textContent = 'Processing images...';

    // Simulate progress
    let progress = 0;
    const progressInterval = setInterval(() => {
        progress += 10;
        document.getElementById('progress').value = progress;
        document.getElementById('progress-text').textContent = `${progress}%`;
        if (progress >= 100) clearInterval(progressInterval);
    }, 500);

    try {
        const response = await fetch('/process_images', {
            method: 'POST',
            headers: getAuthHeaders(),
            body: formData
        });
        clearInterval(progressInterval);
        document.getElementById('progress').value = 100;
        document.getElementById('progress-text').textContent = '100%';

        const result = await response.json();
        if (response.ok) {
            document.getElementById('logs').textContent = result.logs;
            document.getElementById('map').textContent = result.metrics.mAP.toFixed(4);
            document.getElementById('precision').textContent = result.metrics.Precision.toFixed(4);
            document.getElementById('accuracy').textContent = result.metrics.Accuracy.toFixed(4);
            document.getElementById('debug-info').textContent = 'Image processing complete';
        } else {
            document.getElementById('logs').textContent = 'Error: ' + result.detail;
            document.getElementById('debug-info').textContent = 'Error: ' + result.detail;
        }
    } catch (error) {
        clearInterval(progressInterval);
        document.getElementById('logs').textContent = 'Error: ' + error.message;
        document.getElementById('debug-info').textContent = 'Error: ' + error.message;
    }
}