// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const cameraInput = document.getElementById('cameraInput');
const galleryBtn = document.getElementById('galleryBtn');
const cameraBtn = document.getElementById('cameraBtn');
const previewArea = document.getElementById('previewArea');
const previewImage = document.getElementById('previewImage');
const analyzeBtn = document.getElementById('analyzeBtn');
const loading = document.getElementById('loading');
const resultArea = document.getElementById('resultArea');
const resultContent = document.getElementById('resultContent');
const apiUrlInput = document.getElementById('apiUrl');
const errorMessage = document.getElementById('errorMessage');
const cameraModal = document.getElementById('cameraModal');
const cameraVideo = document.getElementById('cameraVideo');
const captureBtn = document.getElementById('captureBtn');
const closeCameraBtn = document.getElementById('closeCameraBtn');
const switchCameraBtn = document.getElementById('switchCameraBtn');

let selectedFile = null;
let cameraStream = null;
let currentFacingMode = 'environment'; // 'user' for front, 'environment' for back

// Upload area click
uploadArea.addEventListener('click', () => {
    fileInput.click();
});

// Gallery button
galleryBtn.addEventListener('click', () => {
    fileInput.click();
});

// Camera button
cameraBtn.addEventListener('click', async () => {
    await startCamera();
});

// File input change
fileInput.addEventListener('change', handleFileSelect);
cameraInput.addEventListener('change', handleFileSelect);

// Drag and drop
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');

    const files = e.dataTransfer.files;
    if (files.length > 0 && files[0].type.startsWith('image/')) {
        handleFile(files[0]);
    }
});

// Handle file selection
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file && file.type.startsWith('image/')) {
        handleFile(file);
    }
}

// Handle file
function handleFile(file) {
    selectedFile = file;

    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewArea.classList.add('active');
        resultArea.classList.remove('active');
        errorMessage.classList.remove('active');
    };
    reader.readAsDataURL(file);
}

// Start camera
async function startCamera() {
    try {
        const constraints = {
            video: {
                facingMode: currentFacingMode,
                width: { ideal: 1280 },
                height: { ideal: 720 }
            }
        };

        cameraStream = await navigator.mediaDevices.getUserMedia(constraints);
        cameraVideo.srcObject = cameraStream;
        cameraModal.classList.add('active');
        errorMessage.classList.remove('active');
    } catch (error) {
        console.error('Error accessing camera:', error);
        showError('Tidak dapat mengakses kamera. Pastikan Anda memberikan izin akses kamera.');
    }
}

// Stop camera
function stopCamera() {
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
    }
    cameraModal.classList.remove('active');
}

// Switch camera
switchCameraBtn.addEventListener('click', async () => {
    currentFacingMode = currentFacingMode === 'environment' ? 'user' : 'environment';
    stopCamera();
    await startCamera();
});

// Capture photo from camera
captureBtn.addEventListener('click', () => {
    const canvas = document.createElement('canvas');
    canvas.width = cameraVideo.videoWidth;
    canvas.height = cameraVideo.videoHeight;

    const ctx = canvas.getContext('2d');
    ctx.drawImage(cameraVideo, 0, 0);

    canvas.toBlob((blob) => {
        const file = new File([blob], 'camera-capture.jpg', { type: 'image/jpeg' });
        handleFile(file);
        stopCamera();
    }, 'image/jpeg', 0.95);
});

// Close camera
closeCameraBtn.addEventListener('click', () => {
    stopCamera();
});

// Analyze button
analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) {
        showError('Silakan pilih foto terlebih dahulu');
        return;
    }

    const apiUrl = apiUrlInput.value.trim();
    // if (!apiUrl) {
    //     showError('Silakan masukkan URL API endpoint terlebih dahulu');
    //     return;
    // }

    await analyzeImage();
});

// Analyze image
// async function analyzeImage() {
//     try {
//         loading.classList.add('active');
//         resultArea.classList.remove('active');
//         errorMessage.classList.remove('active');
//         analyzeBtn.disabled = true;

//         const formData = new FormData();
//         formData.append('image', selectedFile);

//         const apiUrl = apiUrlInput.value.trim();

//         const response = await fetch(apiUrl, {
//             method: 'POST',
//             body: formData
//         });

//         if (!response.ok) {
//             throw new Error(`HTTP error! status: ${response.status}`);
//         }

//         const result = await response.json();
//         displayResult(result);

//     } catch (error) {
//         console.error('Error:', error);
//         showError('Gagal menganalisis foto: ' + error.message);
//     } finally {
//         loading.classList.remove('active');
//         analyzeBtn.disabled = false;
//     }
// }

// Display result
function displayResult(data) {
    resultContent.innerHTML = '';

    // Customize based on your API response structure
    if (data.diagnosis) {
        resultContent.innerHTML += `
                    <div class="result-item">
                        <strong>🩺 Diagnosis:</strong>
                        ${data.diagnosis}
                    </div>
                `;
    }

    if (data.condition) {
        resultContent.innerHTML += `
                    <div class="result-item">
                        <strong>📋 Kondisi:</strong>
                        ${data.condition}
                    </div>
                `;
    }

    if (data.recommendation) {
        resultContent.innerHTML += `
                    <div class="result-item">
                        <strong>💡 Rekomendasi:</strong>
                        ${data.recommendation}
                    </div>
                `;
    }

    if (data.severity) {
        resultContent.innerHTML += `
                    <div class="result-item">
                        <strong>⚠️ Tingkat Keparahan:</strong>
                        ${data.severity}
                    </div>
                `;
    }

    if (data.confidence) {
        resultContent.innerHTML += `
                    <div class="result-item">
                        <strong>📊 Tingkat Kepercayaan:</strong>
                        ${(data.confidence * 100).toFixed(1)}%
                    </div>
                `;
    }

    // Display all data if no specific fields found
    if (resultContent.innerHTML === '') {
        resultContent.innerHTML = `
                    <div class="result-item">
                        <strong>📄 Hasil Lengkap:</strong>
                        <pre style="white-space: pre-wrap; word-wrap: break-word; margin-top: 10px;">${JSON.stringify(data, null, 2)}</pre>
                    </div>
                `;
    }

    resultArea.classList.add('active');
}

// Show error
function showError(message) {
    errorMessage.textContent = message;
    errorMessage.classList.add('active');
}

// Demo mode - uncomment to test without real API

async function analyzeImage() {
    try {
        loading.classList.add('active');
        resultArea.classList.remove('active');
        errorMessage.classList.remove('active');
        analyzeBtn.disabled = true;

        // Simulate API call
        await new Promise(resolve => setTimeout(resolve, 2000));

        // Mock result
        const mockResult = {
            diagnosis: "Kondisi kulit normal",
            condition: "Tidak ditemukan kelainan yang signifikan",
            recommendation: "Tetap jaga kebersihan dan pola hidup sehat",
            severity: "Rendah",
            confidence: 0.92
        };

        displayResult(mockResult);

    } catch (error) {
        showError('Gagal menganalisis foto: ' + error.message);
    } finally {
        loading.classList.remove('active');
        analyzeBtn.disabled = false;
    }
}