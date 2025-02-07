async function decodeCaptcha() {
    const fileInput = document.getElementById('fileInput');
    const preview = document.getElementById('preview');
    const result = document.getElementById('result');
    
    if (!fileInput.files[0]) {
        alert('Please select an image');
        return;
    }

    // Show preview
    preview.style.display = 'inline';
    preview.src = URL.createObjectURL(fileInput.files[0]);

    // Create form data
    const formData = new FormData();
    formData.append('image', fileInput.files[0]);
    
    try {
        result.textContent = 'Decoding...';
        
        const response = await fetch('http://localhost:5000/api/decode', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        if (data.success) {
            result.textContent = `Decoded CAPTCHA: ${data.result}`;
        } else {
            result.textContent = `Error: ${data.error}`;
        }
    } catch (error) {
        result.textContent = 'Error: Could not connect to server';
        console.error(error);
    }
}

// Preview image when selected
document.getElementById('fileInput').addEventListener('change', function(e) {
    const preview = document.getElementById('preview');
    const result = document.getElementById('result');
    
    if (this.files && this.files[0]) {
        preview.style.display = 'inline';
        preview.src = URL.createObjectURL(this.files[0]);
        result.textContent = 'Click Decode CAPTCHA to process';
    }
});
