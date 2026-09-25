const API_URL = 'http://127.0.0.1:8000';
const input = document.querySelector('#image-input');
const dropzone = document.querySelector('#dropzone');
const previewWrap = document.querySelector('#preview-wrap');
const preview = document.querySelector('#preview');
const analyzeButton = document.querySelector('#analyze-button');
const removeButton = document.querySelector('#remove-button');
const errorMessage = document.querySelector('#error-message');
const resultState = document.querySelector('#result-state');
const emptyState = document.querySelector('#empty-state');
const resultContent = document.querySelector('#result-content');
const explanationSection = document.querySelector('#explanation-section');
const heatmap = document.querySelector('#heatmap');
let selectedFile = null;

function setFile(file) {
  if (!file || !file.type.startsWith('image/')) {
    showError('Please choose a JPG, PNG, or WEBP image.');
    return;
  }
  selectedFile = file;
  preview.src = URL.createObjectURL(file);
  dropzone.classList.add('hidden');
  previewWrap.classList.remove('hidden');
  analyzeButton.disabled = false;
  errorMessage.classList.add('hidden');
}

function showError(message) {
  errorMessage.textContent = message;
  errorMessage.classList.remove('hidden');
}

function reset() {
  selectedFile = null;
  input.value = '';
  preview.src = '';
  dropzone.classList.remove('hidden');
  previewWrap.classList.add('hidden');
  analyzeButton.disabled = true;
}

function renderScores(scores) {
  const scoresElement = document.querySelector('#scores');
  scoresElement.innerHTML = Object.entries(scores)
    .sort(([, first], [, second]) => second - first)
    .map(([label, score]) => `<div class="score-row"><div>${label}<div class="score-track"><span style="width:${score * 100}%"></span></div></div><b>${(score * 100).toFixed(0)}%</b></div>`)
    .join('');
}

async function analyze() {
  if (!selectedFile) return;
  analyzeButton.disabled = true;
  analyzeButton.querySelector('span').textContent = 'Analyzing...';
  resultState.textContent = 'Working';
  errorMessage.classList.add('hidden');
  const formData = new FormData();
  formData.append('file', selectedFile);
  try {
    const response = await fetch(`${API_URL}/predict`, { method: 'POST', body: formData });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || 'The API could not analyze this image.');
    document.querySelector('#disease-name').textContent = result.disease;
    document.querySelector('#confidence-value').textContent = `${(result.confidence * 100).toFixed(1)}%`;
    document.querySelector('#confidence-meter').style.width = `${result.confidence * 100}%`;
    document.querySelector('#severity-value').textContent = result.severity || 'Unavailable';
    document.querySelector('#treatment-value').textContent = result.treatment?.text || 'Consult an aquatic veterinarian.';
    renderScores(result.scores);
    if (result.gradcam_png_base64) {
      heatmap.src = `data:image/png;base64,${result.gradcam_png_base64}`;
      explanationSection.classList.remove('hidden');
    } else {
      explanationSection.classList.add('hidden');
    }
    emptyState.classList.add('hidden');
    resultContent.classList.remove('hidden');
    resultState.textContent = 'Complete';
  } catch (error) {
    resultState.textContent = 'Error';
    showError(error.message);
  } finally {
    analyzeButton.disabled = false;
    analyzeButton.querySelector('span').textContent = 'Analyze image';
  }
}

input.addEventListener('change', () => setFile(input.files[0]));
removeButton.addEventListener('click', reset);
analyzeButton.addEventListener('click', analyze);
['dragenter', 'dragover'].forEach((eventName) => dropzone.addEventListener(eventName, (event) => { event.preventDefault(); dropzone.classList.add('dragging'); }));
['dragleave', 'drop'].forEach((eventName) => dropzone.addEventListener(eventName, (event) => { event.preventDefault(); dropzone.classList.remove('dragging'); }));
dropzone.addEventListener('drop', (event) => setFile(event.dataTransfer.files[0]));