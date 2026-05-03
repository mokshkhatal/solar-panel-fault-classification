document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const previewArea = document.getElementById('preview-area');
    const imagePreview = document.getElementById('image-preview');
    const btnCancel = document.getElementById('btn-cancel');
    const btnAnalyze = document.getElementById('btn-analyze');
    const loader = document.getElementById('loader');
    const resultsSection = document.getElementById('results-section');
    
    // Result elements
    const resClass = document.getElementById('res-class');
    const resConf = document.getElementById('res-conf');
    const confBar = document.getElementById('conf-bar');
    const resSeverity = document.getElementById('res-severity');
    const resDelta = document.getElementById('res-delta');
    const resEntropy = document.getElementById('res-entropy');
    const entropyFill = document.getElementById('entropy-fill');
    const resHeatmap = document.getElementById('res-heatmap');

    let currentFile = null;

    // Drag and drop events
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    });

    dropZone.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', function() {
        handleFiles(this.files);
    });

    function handleFiles(files) {
        if (files.length > 0) {
            const file = files[0];
            if (file.type.startsWith('image/')) {
                currentFile = file;
                const reader = new FileReader();
                reader.onload = (e) => {
                    imagePreview.src = e.target.result;
                    dropZone.classList.add('hidden');
                    previewArea.classList.remove('hidden');
                    resultsSection.classList.add('hidden'); // Hide previous results
                    
                    // Reset UI
                    resHeatmap.classList.remove('loaded');
                    confBar.style.width = '0%';
                    entropyFill.style.strokeDashoffset = 126;
                }
                reader.readAsDataURL(file);
            } else {
                alert('Please upload an image file.');
            }
        }
    }

    btnCancel.addEventListener('click', () => {
        currentFile = null;
        fileInput.value = '';
        previewArea.classList.add('hidden');
        dropZone.classList.remove('hidden');
        resultsSection.classList.add('hidden');
    });

    btnAnalyze.addEventListener('click', async () => {
        if (!currentFile) return;

        // Show loader
        loader.classList.add('active');
        btnAnalyze.disabled = true;
        btnCancel.disabled = true;
        resultsSection.classList.add('hidden');

        const formData = new FormData();
        formData.append('image', currentFile);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.success) {
                displayResults(data.prediction, data.heatmap_url);
            } else {
                alert('Error: ' + data.error);
            }
        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred during analysis.');
        } finally {
            // Hide loader
            loader.classList.remove('active');
            btnAnalyze.disabled = false;
            btnCancel.disabled = false;
        }
    });

    function displayResults(prediction, heatmapUrl) {
        resultsSection.classList.remove('hidden');

        // Animate class
        resClass.textContent = prediction.class;

        // Animate confidence
        const confPercent = (prediction.confidence * 100).toFixed(1);
        resConf.textContent = `${confPercent}%`;
        setTimeout(() => {
            confBar.style.width = `${confPercent}%`;
        }, 100);

        // Severity
        resSeverity.textContent = prediction.severity;
        resSeverity.className = `severity-badge ${prediction.severity}`;

        // Delta
        resDelta.textContent = prediction.delta.toFixed(2);

        // Entropy
        const entropyVal = prediction.entropy.toFixed(3);
        resEntropy.textContent = entropyVal;
        
        // Max dashoffset is 126. 
        // 0 entropy = offset 126 (empty)
        // 1 entropy = offset 0 (full)
        // Actually, let's reverse it: lower entropy is better.
        // Let's just fill the gauge according to entropy.
        setTimeout(() => {
            const maxOffset = 126;
            const targetOffset = maxOffset - (prediction.entropy * maxOffset);
            entropyFill.style.strokeDashoffset = Math.max(0, targetOffset);
            
            // Color code entropy
            if (prediction.entropy > 0.6) {
                entropyFill.style.stroke = 'var(--severity-high)';
            } else if (prediction.entropy > 0.3) {
                entropyFill.style.stroke = 'var(--severity-medium)';
            } else {
                entropyFill.style.stroke = 'var(--severity-low)';
            }
        }, 100);

        // Load Heatmap
        resHeatmap.onload = () => {
            resHeatmap.classList.add('loaded');
        };
        resHeatmap.src = heatmapUrl;
        
        // Scroll to results
        setTimeout(() => {
            resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 100);
    }
});
