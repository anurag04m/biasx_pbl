// API Configuration and Utilities

const API = {
  // Get the API base URL - defaults to localhost:5000 but can be overridden via URL param
  getBaseUrl() {
    // 1. Check URL param ?api=
    try {
      const params = new URLSearchParams(window.location.search);
      const apiParam = params.get('api');
      if (apiParam) {
        localStorage.setItem('biasx_api_url', apiParam);
        return apiParam.replace(/\/+$|\/$/, '').replace(/\s+/g, '');
      }
    } catch (e) { }

    // 2. Check localStorage
    const saved = localStorage.getItem('biasx_api_url');
    if (saved) return saved;

    // 3. Sensible defaults
    try {
      const loc = window.location;
      const hostname = loc.hostname;

      // If we are on localhost, use the current protocol (usually http)
      if (hostname === 'localhost' || hostname === '127.0.0.1' || hostname === '') {
        return `${loc.protocol}//localhost:5000`;
      }
    } catch (e) { }

    // Default fallback for GitHub Pages or other hosting
    // We default to http://localhost:5000 as that's where the backend usually is
    return 'http://localhost:5000';
  },

  setBaseUrl(url) {
    if (url) {
      localStorage.setItem('biasx_api_url', url.replace(/\/+$|\/$/, '').replace(/\s+/g, ''));
      window.location.reload();
    }
  },

  // Make an API request
  async request(endpoint, options = {}) {
    const base = this.getBaseUrl();
    const url = `${base}${endpoint}`;

    try {
      const response = await fetch(url, {
        ...options,
        headers: {
          ...options.headers,
        },
      });

      if (!response.ok) {
        const text = await response.text().catch(() => null);
        throw new Error(`API Error ${response.status}: ${text || response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('API Request failed:', error);
      throw error;
    }
  },

  // Upload dataset
  async uploadDataset(file) {
    const formData = new FormData();
    formData.append('dataset', file);

    return this.request('/upload_dataset', {
      method: 'POST',
      body: formData,
    });
  },

  // Fetch available metrics
  async getMetrics() {
    return this.request('/metrics', {
      method: 'GET',
    });
  },

  // Run bias analysis
  async analyze(payload) {
    return this.request('/analyze', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });
  },

  // Apply mitigation
  async mitigate(payload) {
    return this.request('/mitigate', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });
  },

  // Get download URL
  getDownloadUrl(endpoint) {
    return `${this.getBaseUrl()}${endpoint}`;
  },
};

// Utility functions
const Utils = {
  showMessage(message, type = 'info') {
    const container = document.getElementById('message-container');
    const messageEl = document.createElement('div');
    messageEl.className = `message ${type}`;
    messageEl.textContent = message;
    container.appendChild(messageEl);

    setTimeout(() => {
      messageEl.remove();
    }, 5000);
  },

  showError(error) {
    this.showMessage(error.message || error, 'error');
  },

  showSuccess(message) {
    this.showMessage(message, 'success');
  },

  // Compute mitigation suggestion based on analysis results
  computeSuggestion(results) {
    if (!results || !results.metrics) return null;

    let severity = 0; // 0=low, 1=medium, 2=high
    const reasons = [];

    // Check positive_rate_ratio (formerly disparate_impact)
    if (results.metrics.positive_rate_ratio) {
      const val = results.metrics.positive_rate_ratio.value;
      if (val !== null && !isNaN(val)) {
        if (val < 0.8) {
          severity = Math.max(severity, 2);
          reasons.push(`Positive Rate Ratio ${val.toFixed(3)} < 0.8 (strong bias)`);
        } else if (val < 0.9) {
          severity = Math.max(severity, 1);
          reasons.push(`Positive Rate Ratio ${val.toFixed(3)} between 0.8-0.9 (moderate bias)`);
        }
      }
    }
    // Fallback to old name if present
    else if (results.metrics.disparate_impact) {
      const val = results.metrics.disparate_impact.value;
      if (val !== null && !isNaN(val)) {
        if (val < 0.8) {
          severity = Math.max(severity, 2);
          reasons.push(`Disparate Impact ${val.toFixed(3)} < 0.8 (strong bias)`);
        } else if (val < 0.9) {
          severity = Math.max(severity, 1);
          reasons.push(`Disparate Impact ${val.toFixed(3)} between 0.8-0.9 (moderate bias)`);
        }
      }
    }

    // Check selection_rate (formerly statistical_parity_difference)
    if (results.metrics.selection_rate) {
      const val = Math.abs(results.metrics.selection_rate.value);
      if (val !== null && !isNaN(val)) {
        if (val > 0.2) {
          severity = Math.max(severity, 2);
          reasons.push(`Selection Rate difference ${val.toFixed(3)} > 0.2 (strong bias)`);
        } else if (val > 0.1) {
          severity = Math.max(severity, 1);
          reasons.push(`Selection Rate difference ${val.toFixed(3)} > 0.1 (moderate bias)`);
        }
      }
    }
    // Fallback to old name if present
    else if (results.metrics.statistical_parity_difference) {
      const val = Math.abs(results.metrics.statistical_parity_difference.value);
      if (val !== null && !isNaN(val)) {
        if (val > 0.2) {
          severity = Math.max(severity, 2);
          reasons.push(`Statistical parity difference ${val.toFixed(3)} > 0.2 (strong bias)`);
        } else if (val > 0.1) {
          severity = Math.max(severity, 1);
          reasons.push(`Statistical parity difference ${val.toFixed(3)} > 0.1 (moderate bias)`);
        }
      }
    }

    if (reasons.length === 0) {
      reasons.push('No significant bias detected by the heuristics');
    }

    const levels = ['low', 'medium', 'high'];
    let methods = [];

    if (severity === 0) {
      methods = ['reweighing'];
    } else if (severity === 1) {
      methods = ['reweighing', 'optimized_preprocessing'];
    } else {
      methods = ['reweighing', 'disparate_impact_remover', 'optimized_preprocessing'];
    }

    return {
      level: levels[severity],
      methods,
      reason: reasons.join('; '),
    };
  },
};
