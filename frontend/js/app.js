// Main Application Logic

class BiasDetectionApp {
  constructor() {
    this.sessionId = null;
    this.columns = [];
    this.analysisResult = null;
    this.init();
  }

  init() {
    // Bind event listeners
    document.getElementById('upload-btn').addEventListener('click', () => this.uploadDataset());
    document.getElementById('analyze-btn').addEventListener('click', () => this.runAnalysis());
    document.getElementById('mitigate-btn').addEventListener('click', () => this.runMitigation());
    document.getElementById('clear-btn').addEventListener('click', () => this.clearResults());
    document.getElementById('model-mitigate-btn').addEventListener('click', () => this.modelMitigation());

    // Display current API URL
    document.getElementById('api-url').textContent = API.getBaseUrl();
  }

  async uploadDataset() {
    const fileInput = document.getElementById('dataset-file');
    const file = fileInput.files[0];

    if (!file) {
      Utils.showError('Please select a CSV file');
      return;
    }

    try {
      this.setLoading(true);
      const response = await API.uploadDataset(file);

      this.sessionId = response.session_id;
      this.columns = response.columns || [];

      // Update UI
      document.getElementById('session-id').textContent = this.sessionId;
      document.getElementById('session-info').classList.remove('hidden');
      document.getElementById('analysis-section').classList.remove('hidden');
      document.getElementById('mitigation-section').classList.remove('hidden');

      // Populate dropdowns
      this.populateDropdowns();

      Utils.showSuccess('Dataset uploaded successfully!');
    } catch (error) {
      Utils.showError(error);
    } finally {
      this.setLoading(false);
    }
  }

  populateDropdowns() {
    const protectedAttrSelect = document.getElementById('protected-attr');
    const labelColSelect = document.getElementById('label-col');

    // Clear existing options
    protectedAttrSelect.innerHTML = '';
    labelColSelect.innerHTML = '';

    // Add options
    this.columns.forEach(col => {
      const option1 = document.createElement('option');
      option1.value = col;
      option1.textContent = col;
      protectedAttrSelect.appendChild(option1);

      const option2 = document.createElement('option');
      option2.value = col;
      option2.textContent = col;
      labelColSelect.appendChild(option2);
    });
  }

  async runAnalysis() {
    if (!this.sessionId) {
      Utils.showError('Please upload a dataset first');
      return;
    }

    const payload = {
      session_id: this.sessionId,
      protected_attr: document.getElementById('protected-attr').value,
      label_column: document.getElementById('label-col').value,
      privileged_value: document.getElementById('priv-val').value,
      unprivileged_value: document.getElementById('unpriv-val').value,
      selected_metrics: document.getElementById('metrics').value.split(',').map(s => s.trim()).filter(Boolean),
    };

    if (!payload.protected_attr || !payload.label_column) {
      Utils.showError('Please select protected attribute and label column');
      return;
    }

    if (!payload.privileged_value || !payload.unprivileged_value) {
      Utils.showError('Please enter privileged and unprivileged values');
      return;
    }

    try {
      this.setLoading(true);
      const response = await API.analyze(payload);

      this.analysisResult = response.results;
      this.displayAnalysisResults();

      Utils.showSuccess('Analysis completed!');
    } catch (error) {
      Utils.showError(error);
    } finally {
      this.setLoading(false);
    }
  }

  displayAnalysisResults() {
    const container = document.getElementById('analysis-results');
    container.classList.remove('hidden');

    // Display raw results
    document.getElementById('results-json').textContent = JSON.stringify(this.analysisResult, null, 2);

    // Display suggestion
    const suggestion = Utils.computeSuggestion(this.analysisResult);
    if (suggestion) {
      const suggestionHtml = `
        <strong>Suggested Mitigation Level:</strong> ${suggestion.level}<br>
        <strong>Recommended Methods:</strong> ${suggestion.methods.join(', ')}<br>
        <strong>Reason:</strong> ${suggestion.reason}
      `;
      document.getElementById('suggestion-text').innerHTML = suggestionHtml;
      document.getElementById('suggestion-box').classList.remove('hidden');
    }
  }

  async runMitigation() {
    if (!this.sessionId) {
      Utils.showError('Please upload a dataset first');
      return;
    }

    const method = document.getElementById('mitigation-method').value;
    const payload = {
      session_id: this.sessionId,
      method: method,
      kwargs: {},
    };

    // Add repair_level for disparate_impact_remover
    if (method === 'disparate_impact_remover') {
      payload.kwargs.repair_level = parseFloat(document.getElementById('repair-level').value);
    }

    try {
      this.setLoading(true);
      const response = await API.mitigate(payload);

      // Display mitigation results
      document.getElementById('mitigation-results').classList.remove('hidden');
      document.getElementById('mitigation-json').textContent = JSON.stringify(response.new_results, null, 2);

      // Show download link if available
      if (response.download_endpoint) {
        const downloadUrl = API.getDownloadUrl(response.download_endpoint);
        document.getElementById('download-link').href = downloadUrl;
        document.getElementById('download-section').classList.remove('hidden');
      }

      Utils.showSuccess('Mitigation applied successfully!');
    } catch (error) {
      Utils.showError(error);
    } finally {
      this.setLoading(false);
    }
  }

  clearResults() {
    document.getElementById('mitigation-results').classList.add('hidden');
    document.getElementById('download-section').classList.add('hidden');
    Utils.showSuccess('Results cleared');
  }

  modelMitigation() {
    alert('Model mitigation is a frontend-only feature. Backend logic not yet implemented.');
  }

  setLoading(isLoading) {
    const buttons = document.querySelectorAll('button');
    buttons.forEach(btn => {
      btn.disabled = isLoading;
    });
  }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
  new BiasDetectionApp();
});
